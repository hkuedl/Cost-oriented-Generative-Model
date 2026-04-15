import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import *

def d_hinge_loss(real_logits, fake_logits):
    d_loss_real = torch.mean(torch.relu(1.0 - real_logits))
    d_loss_fake = torch.mean(torch.relu(1.0 + fake_logits))
    return d_loss_real + d_loss_fake

def g_hinge_loss(fake_logits):
    return -torch.mean(fake_logits)


class ResBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + 0.1 * self.block(x)


class CGAN_Generator_Multi(nn.Module):
    """
    X: [B,T,F]
    z_global: [B,zg]
    z_node:   [B,N,zn]
    -> Y: [B,T,N]
    """
    def __init__(
        self,
        input_dim,
        n_nodes,
        seq_len=24,
        z_global_dim=16,
        z_node_dim=8,
        trunk_hidden=256,
        trunk_blocks=2,
        head_hidden=256,
        head_blocks=2,
        dropout=0.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.n_nodes = int(n_nodes)
        self.seq_len = int(seq_len)

        self.zg = int(z_global_dim)
        self.zn = int(z_node_dim)

        x_flat = self.seq_len * self.input_dim

        self.trunk_head = nn.Sequential(
            nn.Linear(x_flat, trunk_hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
        )
        self.trunk_body = nn.Sequential(*[ResBlock(trunk_hidden, dropout) for _ in range(int(trunk_blocks))])

        head_in = trunk_hidden + self.zg + self.n_nodes * self.zn
        layers = [
            nn.Linear(head_in, head_hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
        ]
        layers += [ResBlock(head_hidden, dropout) for _ in range(int(head_blocks))]
        layers += [nn.Linear(head_hidden, self.seq_len * self.n_nodes)]
        self.head = nn.Sequential(*layers)

    def forward(self, X, z_global, z_node):
        B = X.size(0)
        x = X.reshape(B, -1)
        h = self.trunk_body(self.trunk_head(x))          # [B,trunk_hidden]
        zn_flat = z_node.reshape(B, -1)                  # [B,N*zn]
        hin = torch.cat([h, z_global, zn_flat], dim=-1)  # [B, head_in]
        yflat = self.head(hin)                           # [B,T*N]
        return yflat.view(B, self.seq_len, self.n_nodes)

    def sample(self, X, n_samples=50, z_temp=1.0, share_znode=False, grad_enabled=None):
        """
        X: [B,T,F] -> Y: [S,B,T,N]

        - grad_enabled=None : follow torch.is_grad_enabled()
        - grad_enabled=False: keep old behavior (eval + no_grad)
        - grad_enabled=True : differentiable, vectorized sampling (no @torch.no_grad)
        """
        if grad_enabled is None:
            grad_enabled = torch.is_grad_enabled()

        S = int(n_samples)
        B = X.size(0)
        device = X.device
        dtype = X.dtype

        if not grad_enabled:
            # === old behavior (non-differentiable) ===
            with torch.no_grad():
                was_training = self.training
                self.eval()

                X_rep = X.repeat_interleave(S, dim=0)  # [B*S,T,F]
                zg = torch.randn(B * S, self.zg, device=device, dtype=dtype) * float(z_temp)

                if not share_znode:
                    zn = torch.randn(B * S, self.n_nodes, self.zn, device=device, dtype=dtype) * float(z_temp)
                else:
                    zn0 = torch.randn(B * S, 1, self.zn, device=device, dtype=dtype) * float(z_temp)
                    zn = zn0.expand(B * S, self.n_nodes, self.zn).contiguous()

                Y = self.forward(X_rep, zg, zn).view(B, S, self.seq_len, self.n_nodes).permute(1, 0, 2, 3)

                if was_training:
                    self.train()
                return Y  # [S,B,T,N]

        X_rep = X.unsqueeze(0).expand(S, *X.shape).reshape(S * B, *X.shape[1:])

        zg = torch.randn(S, B, self.zg, device=device, dtype=dtype) * float(z_temp)
        zg = zg.reshape(S * B, self.zg)

        if not share_znode:
            zn = torch.randn(S, B, self.n_nodes, self.zn, device=device, dtype=dtype) * float(z_temp)
        else:
            zn0 = torch.randn(S, B, 1, self.zn, device=device, dtype=dtype) * float(z_temp)
            zn = zn0.expand(S, B, self.n_nodes, self.zn).contiguous()
        zn = zn.reshape(S * B, self.n_nodes, self.zn)

        Y = self.forward(X_rep, zg, zn).reshape(S, B, self.seq_len, self.n_nodes)
        return Y

# =========================================================
# Discriminator (joint MLP)
# =========================================================
class CGAN_Discriminator_Multi(nn.Module):
    """
    X: [B,T,F]
    Y: [B,T,N]
    -> score [B,1]
    """
    def __init__(self, input_dim, n_nodes, seq_len=24, hidden=256, dropout=0.0):
        super().__init__()
        self.input_dim = int(input_dim)
        self.n_nodes = int(n_nodes)
        self.seq_len = int(seq_len)

        x_flat = self.seq_len * self.input_dim
        y_flat = self.seq_len * self.n_nodes
        in_dim = x_flat + y_flat

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),

            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
        )
        self.tail = nn.Linear(hidden, 1)

    def forward(self, X, Y):
        B = X.size(0)
        x = X.reshape(B, -1)
        y = Y.reshape(B, -1)
        h = torch.cat([x, y], dim=1)
        feat = self.net(h)      # [B,hidden]
        out = self.tail(feat)   # [B,1]
        return out



class CGAN_Generator_Single(nn.Module):
    """
    x: [B,T,F], z:[B,Z] -> y:[B,T]
    sample(x[1,T,F]) -> [S,1,T] (caller can .squeeze(1)->[S,T])
    """
    def __init__(self, input_dim, seq_len=24, z_dim=32, hidden=256, n_blocks=2, dropout=0.0):
        super().__init__()
        self.seq_len = int(seq_len)
        self.input_dim = int(input_dim)
        self.z_dim = int(z_dim)
        self.in_dim = self.seq_len * self.input_dim + self.z_dim

        self.head = nn.Sequential(
            nn.Linear(self.in_dim, hidden),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.body = nn.Sequential(*[ResBlock(hidden, dropout) for _ in range(n_blocks)])
        self.tail = nn.Linear(hidden, self.seq_len)

    def forward(self, x, z):
        B = x.size(0)
        x_flat = x.reshape(B, -1)
        inp = torch.cat([x_flat, z], dim=1)
        h = self.body(self.head(inp))
        y = self.tail(h)  # [B,T]
        return y

    def sample(self, x, n_samples=50, z_temp=1.0, grad_enabled=None):
        assert x.dim() == 3, "x must be [B,T,F]"

        if grad_enabled is None:
            grad_enabled = torch.is_grad_enabled()

        S = int(n_samples)
        B = x.size(0)
        device = x.device
        dtype = x.dtype

        if not grad_enabled:
            with torch.no_grad():
                was_training = self.training
                self.eval()

                x_rep = x.repeat_interleave(S, dim=0)  # [B*S,T,F]
                z = torch.randn(B * S, self.z_dim, device=device, dtype=dtype) * float(z_temp)
                y = self.forward(x_rep, z).view(B, S, self.seq_len).permute(1, 0, 2)  # [S,B,T]

                if was_training:
                    self.train()
                return y

        # differentiable path
        x_rep = x.unsqueeze(0).expand(S, *x.shape).reshape(S * B, *x.shape[1:])  # [S*B,T,F]
        z = torch.randn(S, B, self.z_dim, device=device, dtype=dtype) * float(z_temp)
        z = z.reshape(S * B, self.z_dim)

        y = self.forward(x_rep, z).reshape(S, B, self.seq_len)  # [S,B,T]
        return y

class CGAN_Discriminator_Single(nn.Module):
    """
    x:[B,T,F], y:[B,T] -> score:[B,1]
    Support feature matching: forward(..., return_feat=True) -> (score, feat)
    """
    def __init__(self, input_dim, seq_len=24, hidden=256, n_blocks=2, dropout=0.0):
        super().__init__()
        self.seq_len = int(seq_len)
        self.input_dim = int(input_dim)
        self.in_dim = self.seq_len * self.input_dim + self.seq_len

        self.head = nn.Sequential(
            nn.Linear(self.in_dim, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
        )
        self.body = nn.Sequential(*[ResBlock(hidden, dropout) for _ in range(n_blocks)])
        self.tail = nn.Linear(hidden, 1)

    def forward(self, x, y, return_feat=False):
        if y.dim() == 3 and y.shape[-1] == 1:
            y = y[..., 0]
        B = x.size(0)
        x_flat = x.reshape(B, -1)
        inp = torch.cat([x_flat, y], dim=1)
        feat = self.body(self.head(inp))
        out = self.tail(feat)
        if return_feat:
            return out, feat
        return out