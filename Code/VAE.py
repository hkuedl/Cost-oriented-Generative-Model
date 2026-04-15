import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import time
import scipy.stats as stats
from torch.utils.data import DataLoader, TensorDataset
from utils import *
from data_loader import *


def cvae_point_elbo_loss_condprior(mu_y, y, mu_q, logvar_q, mu_p, logvar_p, beta=1.0):
    recon = torch.mean((y - mu_y) ** 2)
    kl = kl_diag_gaussian_q_p(mu_q, logvar_q, mu_p, logvar_p)
    return recon + beta * kl, recon.detach(), kl.detach()


def kl_diag_gaussian_q_p(mu_q, logvar_q, mu_p, logvar_p):
    """
    KL( N(mu_q, sq) || N(mu_p, sp) ) for diagonal Gaussians.
    Returns scalar mean KL over batch.
    """
    # sp, sq
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)

    # KL per sample per dim:
    # 0.5 * [ log(var_p/var_q) + (var_q + (mu_q-mu_p)^2)/var_p - 1 ]
    kl = 0.5 * (
        (logvar_p - logvar_q)
        + (var_q + (mu_q - mu_p) ** 2) / (var_p + 1e-8)
        - 1.0
    )
    # mean over batch and sum over dims (or mean over dims; either OK if consistent)
    return kl.sum(dim=1).mean()


def kl_diag_gaussians_per_sample(mu_q, logvar_q, mu_p, logvar_p, sum_dim=-1):
    """
    mu_q/logvar_q/mu_p/logvar_p: same shape, first dim is batch
    returns: scalar (mean over batch) KL
    """
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    kl = 0.5 * (
        logvar_p - logvar_q
        + (var_q + (mu_q - mu_p) ** 2) / (var_p + 1e-8)
        - 1.0
    )
    kl_b = kl.sum(dim=sum_dim)
    return kl_b.mean()


class CVAE_Single(nn.Module):
    def __init__(self, input_dim, z_dim=16, hidden=256, dropout=0.0):
        super().__init__()
        self.input_dim = int(input_dim)
        self.z_dim = int(z_dim)

        x_dim = 24 * self.input_dim
        y_dim = 24

        # Encoder q(z|x,y)
        self.enc = nn.Sequential(
            nn.Linear(x_dim + y_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.q_mu = nn.Linear(hidden, z_dim)
        self.q_logvar = nn.Linear(hidden, z_dim)

        # Conditional prior p(z|x)
        self.prior = nn.Sequential(
            nn.Linear(x_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.p_mu = nn.Linear(hidden, z_dim)
        self.p_logvar = nn.Linear(hidden, z_dim)

        # Decoder p(y|x,z) point
        self.dec = nn.Sequential(
            nn.Linear(x_dim + z_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.dec_mu = nn.Linear(hidden, y_dim)

    @staticmethod
    def _reparam(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode_q(self, X, y):
        B = X.shape[0]
        x = X.reshape(B, -1)
        inp = torch.cat([x, y], dim=1)
        h = self.enc(inp)
        mu_q = self.q_mu(h)
        logvar_q = self.q_logvar(h)
        return mu_q, logvar_q

    def prior_p(self, X):
        B = X.shape[0]
        x = X.reshape(B, -1)
        h = self.prior(x)
        mu_p = self.p_mu(h)
        logvar_p = self.p_logvar(h)
        return mu_p, logvar_p

    def decode(self, X, z):
        B = X.shape[0]
        x = X.reshape(B, -1)
        inp = torch.cat([x, z], dim=1)
        h = self.dec(inp)
        mu_y = self.dec_mu(h)
        return mu_y

    def forward(self, X, y):
        mu_q, logvar_q = self.encode_q(X, y)
        mu_p, logvar_p = self.prior_p(X)
        z = self._reparam(mu_q, logvar_q)  # sample from posterior during training
        mu_y = self.decode(X, z)
        return mu_y, mu_q, logvar_q, mu_p, logvar_p

    def sample(self, X, n_samples=1, z_temp=1.0, grad_enabled=None):
        """
        Sample from p(y|X) using conditional prior.
        Returns: [S,B,24]
        """
        if grad_enabled is None:
            grad_enabled = torch.is_grad_enabled()

        S = int(n_samples)
        B = X.shape[0]
        device = X.device
        dtype = X.dtype

        if not grad_enabled:
            with torch.no_grad():
                was_training = self.training
                self.eval()

                mu_p, logvar_p = self.prior_p(X)
                std_p = torch.exp(0.5 * logvar_p) * float(z_temp)

                samples = []
                for _ in range(S):
                    eps = torch.randn(B, self.z_dim, device=device, dtype=dtype)
                    z = mu_p + std_p * eps
                    mu_y = self.decode(X, z)
                    samples.append(mu_y)

                if was_training:
                    self.train()

                return torch.stack(samples, dim=0)  # [S,B,24]

        # === 需要梯度：不强制 eval，不用 no_grad（可导），向量化实现 ===
        mu_p, logvar_p = self.prior_p(X)
        std_p = torch.exp(0.5 * logvar_p) * float(z_temp)

        eps = torch.randn(S, B, self.z_dim, device=device, dtype=mu_p.dtype)
        z = mu_p.unsqueeze(0) + std_p.unsqueeze(0) * eps  # [S,B,z]

        X_rep = X.unsqueeze(0).expand(S, *X.shape).reshape(S * B, *X.shape[1:])
        z_flat = z.reshape(S * B, self.z_dim)
        mu_y = self.decode(X_rep, z_flat).reshape(S, B, -1)

        return mu_y

        


class CVAE_Multi(nn.Module):
    """
    Hierarchical multi-node CVAE (shared decoder, joint output):
      - global latent z_g: [B, z_global]
      - node latents z_n:  [B, N, z_node]
    X: [B,24,F]
    Y: [B,24,N]
    """

    def __init__(self, input_dim, n_nodes, z_global=8, z_node=4, hidden=256, dropout=0.0):
        super().__init__()
        self.input_dim = int(input_dim)
        self.n_nodes = int(n_nodes)
        self.z_global = int(z_global)
        self.z_node = int(z_node)

        x_dim = 24 * self.input_dim
        y_dim = 24 * self.n_nodes
        z_total = self.z_global + self.n_nodes * self.z_node

        # Encoder q(z|x,y)
        self.enc = nn.Sequential(
            nn.Linear(x_dim + y_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.qg_mu = nn.Linear(hidden, self.z_global)
        self.qg_logvar = nn.Linear(hidden, self.z_global)
        self.qn_mu = nn.Linear(hidden, self.n_nodes * self.z_node)
        self.qn_logvar = nn.Linear(hidden, self.n_nodes * self.z_node)

        # Prior p(z|x)
        self.prior = nn.Sequential(
            nn.Linear(x_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.pg_mu = nn.Linear(hidden, self.z_global)
        self.pg_logvar = nn.Linear(hidden, self.z_global)
        self.pn_mu = nn.Linear(hidden, self.n_nodes * self.z_node)
        self.pn_logvar = nn.Linear(hidden, self.n_nodes * self.z_node)

        # Shared decoder p(y|x,z): outputs all nodes jointly (NO node-wise decoder)
        self.dec = nn.Sequential(
            nn.Linear(x_dim + z_total, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.dec_mu = nn.Linear(hidden, y_dim)

    @staticmethod
    def _reparam(mu, logvar, z_temp=1.0):
        std = torch.exp(0.5 * logvar) * float(z_temp)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode_q(self, X, Y):
        B = X.shape[0]
        x = X.reshape(B, -1)
        y = Y.reshape(B, -1)
        h = self.enc(torch.cat([x, y], dim=1))

        mu_g = self.qg_mu(h)
        logvar_g = self.qg_logvar(h)
        mu_n = self.qn_mu(h).view(B, self.n_nodes, self.z_node)
        logvar_n = self.qn_logvar(h).view(B, self.n_nodes, self.z_node)
        return mu_g, logvar_g, mu_n, logvar_n

    def prior_p(self, X):
        B = X.shape[0]
        x = X.reshape(B, -1)
        h = self.prior(x)

        mu_g = self.pg_mu(h)
        logvar_g = self.pg_logvar(h)
        mu_n = self.pn_mu(h).view(B, self.n_nodes, self.z_node)
        logvar_n = self.pn_logvar(h).view(B, self.n_nodes, self.z_node)
        return mu_g, logvar_g, mu_n, logvar_n

    def decode(self, X, z_g, z_n):
        B = X.shape[0]
        x = X.reshape(B, -1)
        z = torch.cat([z_g, z_n.reshape(B, -1)], dim=1)  # [B, zg + N*zn]
        h = self.dec(torch.cat([x, z], dim=1))
        y_flat = self.dec_mu(h)  # [B, 24*N]
        return y_flat.view(B, 24, self.n_nodes)

    def forward(self, X, Y):
        mu_g_q, logvar_g_q, mu_n_q, logvar_n_q = self.encode_q(X, Y)
        mu_g_p, logvar_g_p, mu_n_p, logvar_n_p = self.prior_p(X)

        z_g = self._reparam(mu_g_q, logvar_g_q, z_temp=1.0)
        z_n = self._reparam(mu_n_q, logvar_n_q, z_temp=1.0)

        mu_Y = self.decode(X, z_g, z_n)

        return (
            mu_Y,
            mu_g_q, logvar_g_q, mu_g_p, logvar_g_p,
            mu_n_q, logvar_n_q, mu_n_p, logvar_n_p
        )

    def sample(self, X, n_samples=50, z_temp=1.0, grad_enabled=None):
        """
        Sample from p(Y|X) using conditional prior.
        Returns: [S,B,24,N]
        """
        if grad_enabled is None:
            grad_enabled = torch.is_grad_enabled()

        S = int(n_samples)
        B = X.shape[0]
        device = X.device
        dtype = X.dtype

        if not grad_enabled:
            # === 与之前一致：eval + no_grad + loop ===
            with torch.no_grad():
                was_training = self.training
                self.eval()

                mu_g_p, logvar_g_p, mu_n_p, logvar_n_p = self.prior_p(X)  # [B,zg], [B,zg], [B,N,zn], [B,N,zn]
                std_g = torch.exp(0.5 * logvar_g_p) * float(z_temp)
                std_n = torch.exp(0.5 * logvar_n_p) * float(z_temp)

                outs = []
                for _ in range(S):
                    eps_g = torch.randn(B, self.z_global, device=device, dtype=dtype)
                    eps_n = torch.randn(B, self.n_nodes, self.z_node, device=device, dtype=dtype)
                    z_g = mu_g_p + std_g * eps_g
                    z_n = mu_n_p + std_n * eps_n
                    outs.append(self.decode(X, z_g, z_n))

                if was_training:
                    self.train()

                return torch.stack(outs, dim=0)  # [S,B,24,N]

        # === 需要梯度：可导 + 向量化 ===
        mu_g_p, logvar_g_p, mu_n_p, logvar_n_p = self.prior_p(X)
        std_g = torch.exp(0.5 * logvar_g_p) * float(z_temp)                 # [B,zg]
        std_n = torch.exp(0.5 * logvar_n_p) * float(z_temp)                 # [B,N,zn]

        eps_g = torch.randn(S, B, self.z_global, device=device, dtype=mu_g_p.dtype)
        eps_n = torch.randn(S, B, self.n_nodes, self.z_node, device=device, dtype=mu_n_p.dtype)

        z_g = mu_g_p.unsqueeze(0) + std_g.unsqueeze(0) * eps_g              # [S,B,zg]
        z_n = mu_n_p.unsqueeze(0) + std_n.unsqueeze(0) * eps_n              # [S,B,N,zn]

        X_rep  = X.unsqueeze(0).expand(S, *X.shape).reshape(S * B, *X.shape[1:])        # [S*B,24,F]
        z_g_fb = z_g.reshape(S * B, self.z_global)                                      # [S*B,zg]
        z_n_fb = z_n.reshape(S * B, self.n_nodes, self.z_node)                          # [S*B,N,zn]

        Y = self.decode(X_rep, z_g_fb, z_n_fb).reshape(S, B, 24, self.n_nodes)          # [S,B,24,N]
        return Y


# class CVAE_Multi(nn.Module):
#     """
#     Same as your CVAE_Single, but:
#       Y: [B,24,N], mu_Y: [B,24,N]
#     """
#     def __init__(self, input_dim, n_nodes, z_dim=16, hidden=256, dropout=0.0):
#         super().__init__()
#         self.input_dim = int(input_dim)
#         self.n_nodes = int(n_nodes)
#         self.z_dim = int(z_dim)

#         x_dim = 24 * self.input_dim
#         y_dim = 24 * self.n_nodes

#         # Encoder q(z|x,y)
#         self.enc = nn.Sequential(
#             nn.Linear(x_dim + y_dim, hidden),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden, hidden),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#         )
#         self.q_mu = nn.Linear(hidden, z_dim)
#         self.q_logvar = nn.Linear(hidden, z_dim)

#         # Conditional prior p(z|x)
#         self.prior = nn.Sequential(
#             nn.Linear(x_dim, hidden),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden, hidden),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#         )
#         self.p_mu = nn.Linear(hidden, z_dim)
#         self.p_logvar = nn.Linear(hidden, z_dim)

#         # Decoder p(y|x,z) point
#         self.dec = nn.Sequential(
#             nn.Linear(x_dim + z_dim, hidden),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden, hidden),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#         )
#         self.dec_mu = nn.Linear(hidden, y_dim)

#     @staticmethod
#     def _reparam(mu, logvar, z_temp=1.0):
#         std = torch.exp(0.5 * logvar) * float(z_temp)
#         eps = torch.randn_like(std)
#         return mu + eps * std

#     def encode_q(self, X, Y):
#         B = X.shape[0]
#         x = X.reshape(B, -1)
#         y = Y.reshape(B, -1)
#         h = self.enc(torch.cat([x, y], dim=1))
#         return self.q_mu(h), self.q_logvar(h)

#     def prior_p(self, X):
#         B = X.shape[0]
#         x = X.reshape(B, -1)
#         h = self.prior(x)
#         return self.p_mu(h), self.p_logvar(h)

#     def decode(self, X, z):
#         B = X.shape[0]
#         x = X.reshape(B, -1)
#         h = self.dec(torch.cat([x, z], dim=1))
#         y_flat = self.dec_mu(h)
#         return y_flat.view(B, 24, self.n_nodes)

#     def forward(self, X, Y):
#         mu_q, logvar_q = self.encode_q(X, Y)
#         mu_p, logvar_p = self.prior_p(X)
#         z = self._reparam(mu_q, logvar_q, z_temp=1.0)  # posterior sample (train)
#         mu_Y = self.decode(X, z)
#         return mu_Y, mu_q, logvar_q, mu_p, logvar_p

#     @torch.no_grad()
#     def sample(self, X, n_samples=50, z_temp=1.0):
#         """
#         Sample from conditional prior p(z|X), decode to Y.
#         returns: [S,B,24,N]
#         """
#         self.eval()
#         B = X.shape[0]
#         mu_p, logvar_p = self.prior_p(X)
#         std_p = torch.exp(0.5 * logvar_p) * float(z_temp)

#         outs = []
#         for _ in range(int(n_samples)):
#             eps = torch.randn(B, self.z_dim, device=X.device)
#             z = mu_p + std_p * eps
#             outs.append(self.decode(X, z))
#         return torch.stack(outs, dim=0)



# class CVAE_Multi(nn.Module):
#     """
#     Same as your CVAE_Single, but:
#       Y: [B,24,N], mu_Y: [B,24,N]
#     """
#     def __init__(self, input_dim, n_nodes, z_dim=16, hidden=256, dropout=0.0):
#         super().__init__()
#         self.input_dim = int(input_dim)
#         self.n_nodes = int(n_nodes)
#         self.z_dim = int(z_dim)

#         x_dim = 24 * self.input_dim
#         y_dim = 24 * self.n_nodes

#         # Encoder q(z|x,y)
#         self.enc = nn.Sequential(
#             nn.Linear(x_dim + y_dim, hidden),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden, hidden),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#         )
#         self.q_mu = nn.Linear(hidden, z_dim)
#         self.q_logvar = nn.Linear(hidden, z_dim)

#         # Conditional prior p(z|x)
#         self.prior = nn.Sequential(
#             nn.Linear(x_dim, hidden),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden, hidden),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#         )
#         self.p_mu = nn.Linear(hidden, z_dim)
#         self.p_logvar = nn.Linear(hidden, z_dim)

#         # Decoder p(y|x,z) point
#         self.dec = nn.Sequential(
#             nn.Linear(x_dim + z_dim, hidden),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden, hidden),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#         )
#         self.dec_mu = nn.Linear(hidden, y_dim)

#     @staticmethod
#     def _reparam(mu, logvar, z_temp=1.0):
#         std = torch.exp(0.5 * logvar) * float(z_temp)
#         eps = torch.randn_like(std)
#         return mu + eps * std

#     def encode_q(self, X, Y):
#         B = X.shape[0]
#         x = X.reshape(B, -1)
#         y = Y.reshape(B, -1)
#         h = self.enc(torch.cat([x, y], dim=1))
#         return self.q_mu(h), self.q_logvar(h)

#     def prior_p(self, X):
#         B = X.shape[0]
#         x = X.reshape(B, -1)
#         h = self.prior(x)
#         return self.p_mu(h), self.p_logvar(h)

#     def decode(self, X, z):
#         B = X.shape[0]
#         x = X.reshape(B, -1)
#         h = self.dec(torch.cat([x, z], dim=1))
#         y_flat = self.dec_mu(h)
#         return y_flat.view(B, 24, self.n_nodes)

#     def forward(self, X, Y):
#         mu_q, logvar_q = self.encode_q(X, Y)
#         mu_p, logvar_p = self.prior_p(X)
#         z = self._reparam(mu_q, logvar_q, z_temp=1.0)  # posterior sample (train)
#         mu_Y = self.decode(X, z)
#         return mu_Y, mu_q, logvar_q, mu_p, logvar_p

#     @torch.no_grad()
#     def sample(self, X, n_samples=50, z_temp=1.0):
#         """
#         Sample from conditional prior p(z|X), decode to Y.
#         returns: [S,B,24,N]
#         """
#         self.eval()
#         B = X.shape[0]
#         mu_p, logvar_p = self.prior_p(X)
#         std_p = torch.exp(0.5 * logvar_p) * float(z_temp)

#         outs = []
#         for _ in range(int(n_samples)):
#             eps = torch.randn(B, self.z_dim, device=X.device)
#             z = mu_p + std_p * eps
#             outs.append(self.decode(X, z))
#         return torch.stack(outs, dim=0)
