# diffusion_pkg.py
import math
import copy
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt

# =================================================
# 3) Noise schedule + helpers
# =================================================
def cosine_beta_schedule(T, s=0.008):
    """
    Cosine schedule from Nichol & Dhariwal 2021.
    returns betas: [T] in (0,1)
    """
    steps = T + 1
    x = torch.linspace(0, T, steps)
    alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 1e-5, 0.999)

def extract(a, t, x_shape):
    """
    a: [T], t: [B] long
    return a_t reshaped to [B,1,1...]
    """
    B = t.shape[0]
    out = a.gather(-1, t).float()
    return out.view(B, *([1] * (len(x_shape) - 1)))

def default(val, d):
    return d if val is None else val

# =================================================
# 4) Time embedding
# =================================================
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = int(dim)

    def forward(self, t):
        """
        t: [B] int/long
        out: [B, dim]
        """
        half = self.dim // 2
        device = t.device
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

# =================================================
# 5) Denoiser (UNet-1D over 24 steps)
# =================================================
class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, time_hidden, dropout=0.0, groups=8):
        super().__init__()
        assert in_ch % groups == 0 and out_ch % groups == 0, "channels must be divisible by groups"
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)

        self.time_proj = nn.Linear(time_hidden, out_ch)
        self.skip = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x, t_emb):
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.time_proj(t_emb).unsqueeze(-1)

        h = self.conv2(self.drop(self.act(self.norm2(h))))
        return h + self.skip(x)

class Down1D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.down = nn.Conv1d(ch, ch, kernel_size=4, stride=2, padding=1)  # /2

    def forward(self, x):
        return self.down(x)

class Up1D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.up = nn.ConvTranspose1d(ch, ch, kernel_size=4, stride=2, padding=1)  # *2

    def forward(self, x):
        return self.up(x)

import torch
import torch.nn as nn
import torch.nn.functional as F

class DenoiserUNet_Single(nn.Module):
    """
    Inputs:
      y_t: [B,24]
      X  : [B,24,F]
      t  : [B]
    Output:
      v_hat: [B,24]
    """
    def __init__(self, feat_dim, y_dim=24, time_dim=64, hidden_ch=128, dropout=0.0, groups=8):
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.y_dim = int(y_dim)

        self.time_emb = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_ch),
            nn.SiLU(),
            nn.Linear(hidden_ch, hidden_ch),
        )
        self.time_to_channel = nn.Linear(hidden_ch, 1)

        in_ch = 1 + self.feat_dim + 1  # y + X + time_channel
        self.in_conv = nn.Conv1d(in_ch, hidden_ch, kernel_size=3, padding=1)

        # encoder (24 -> 12 -> 6)
        self.enc1 = ResBlock1D(hidden_ch, hidden_ch, hidden_ch, dropout=dropout, groups=groups)
        self.down1 = Down1D(hidden_ch)
        self.enc2 = ResBlock1D(hidden_ch, hidden_ch, hidden_ch, dropout=dropout, groups=groups)
        self.down2 = Down1D(hidden_ch)

        # bottleneck @ 6
        self.mid1 = ResBlock1D(hidden_ch, hidden_ch, hidden_ch, dropout=dropout, groups=groups)
        self.mid2 = ResBlock1D(hidden_ch, hidden_ch, hidden_ch, dropout=dropout, groups=groups)

        # decoder (6 -> 12 -> 24)
        self.up2 = Up1D(hidden_ch)
        self.dec2 = ResBlock1D(hidden_ch * 2, hidden_ch, hidden_ch, dropout=dropout, groups=groups)
        self.up1 = Up1D(hidden_ch)
        self.dec1 = ResBlock1D(hidden_ch * 2, hidden_ch, hidden_ch, dropout=dropout, groups=groups)

        self.out_norm = nn.GroupNorm(groups, hidden_ch)
        self.out_conv = nn.Conv1d(hidden_ch, 1, kernel_size=3, padding=1)
        self.act = nn.SiLU()

    def forward(self, y_t, t, X):
        B, L = y_t.shape
        assert L == self.y_dim
        assert X.shape[0] == B and X.shape[1] == L and X.shape[2] == self.feat_dim

        ht = self.time_mlp(self.time_emb(t))                         # [B,h]
        tc = self.time_to_channel(ht).unsqueeze(-1).expand(B, 1, L)  # [B,1,24]

        ych = y_t.unsqueeze(1)                        # [B,1,24]
        xch = X.permute(0, 2, 1).contiguous()         # [B,F,24]
        x = torch.cat([ych, xch, tc], dim=1)          # [B,1+F+1,24]

        x = self.in_conv(x)                           # [B,h,24]
        e1 = self.enc1(x, ht)                         # [B,h,24]
        d1 = self.down1(e1)                           # [B,h,12]
        e2 = self.enc2(d1, ht)                        # [B,h,12]
        d2 = self.down2(e2)                           # [B,h,6]

        m = self.mid2(self.mid1(d2, ht), ht)          # [B,h,6]

        u2 = self.up2(m)                              # [B,h,12]
        u2 = torch.cat([u2, e2], dim=1)               # [B,2h,12]
        u2 = self.dec2(u2, ht)                        # [B,h,12]

        u1 = self.up1(u2)                             # [B,h,24]
        u1 = torch.cat([u1, e1], dim=1)               # [B,2h,24]
        u1 = self.dec1(u1, ht)                        # [B,h,24]

        out = self.out_conv(self.act(self.out_norm(u1)))  # [B,1,24]
        return out.squeeze(1)                             # [B,24]

# =================================================
# 6) Conditional DDPM for y|X (v-parameterization) + DDIM
# =================================================

class ConditionalDDPM_Single(nn.Module):
    def __init__(
        self,
        feat_dim,
        T=200,
        y_dim=24,
        time_dim=64,
        hidden_ch=128,
        n_layers=6,
        dropout=0.0,
        beta_schedule="cosine",
    ):
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.y_dim = int(y_dim)
        self.T = int(T)

        if beta_schedule == "cosine":
            betas = cosine_beta_schedule(self.T)
        elif beta_schedule == "linear":
            betas = torch.linspace(1e-4, 0.02, self.T)
        else:
            raise ValueError("beta_schedule must be 'cosine' or 'linear'")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], dtype=torch.float32), alphas_cumprod[:-1]], dim=0
        )

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance.clamp(min=1e-20))
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        # denoiser: UNet over 24 (predicting v)
        self.denoiser = DenoiserUNet_Single(
            feat_dim=self.feat_dim,
            y_dim=self.y_dim,
            time_dim=time_dim,
            hidden_ch=hidden_ch,
            dropout=dropout,
            groups=8,
        )

    def q_sample(self, y0, t, noise=None):
        noise = default(noise, torch.randn_like(y0))
        sqrt_ab = extract(self.sqrt_alphas_cumprod, t, y0.shape)
        sqrt_omab = extract(self.sqrt_one_minus_alphas_cumprod, t, y0.shape)
        return sqrt_ab * y0 + sqrt_omab * noise

    def predict_v(self, y_t, t, X):
        return self.denoiser(y_t, t, X)

    def p_mean_variance(self, y_t, t, X):
        v = self.predict_v(y_t, t, X)  # [B,24]

        sqrt_ab = extract(self.sqrt_alphas_cumprod, t, y_t.shape)
        sqrt_omab = extract(self.sqrt_one_minus_alphas_cumprod, t, y_t.shape)

        # y0_hat = sqrt(ab)*y_t - sqrt(1-ab)*v
        y0_hat = sqrt_ab * y_t - sqrt_omab * v
        y0_hat = torch.clamp(y0_hat, -5.0, 5.0)

        coef1 = extract(self.posterior_mean_coef1, t, y_t.shape)
        coef2 = extract(self.posterior_mean_coef2, t, y_t.shape)
        mean = coef1 * y0_hat + coef2 * y_t

        logvar = extract(self.posterior_log_variance_clipped, t, y_t.shape)
        return mean, logvar

    def p_sample(self, y_t, t, X, grad_enabled=None):
        if grad_enabled is None:
            grad_enabled = torch.is_grad_enabled()

        with torch.set_grad_enabled(grad_enabled):
            mean, logvar = self.p_mean_variance(y_t, t, X)
            if (t == 0).all():
                return mean
            noise = torch.randn_like(y_t)
            return mean + torch.exp(0.5 * logvar) * noise

    def loss(self, X, y0):
        B = X.shape[0]
        device = X.device
        t = torch.randint(0, self.T, (B,), device=device).long()
        noise = torch.randn_like(y0)
        y_t = self.q_sample(y0, t, noise=noise)

        sqrt_ab = extract(self.sqrt_alphas_cumprod, t, y0.shape)
        sqrt_omab = extract(self.sqrt_one_minus_alphas_cumprod, t, y0.shape)

        # v target: v = sqrt(ab)*eps - sqrt(1-ab)*y0
        v = sqrt_ab * noise - sqrt_omab * y0

        v_pred = self.predict_v(y_t, t, X)
        return F.mse_loss(v_pred, v, reduction="mean")

    # =========================
    # DDIM helpers + sampling
    # =========================
    def _predict_x0_from_v(self, y_t, t, v):
        sqrt_ab = extract(self.sqrt_alphas_cumprod, t, y_t.shape)
        sqrt_omab = extract(self.sqrt_one_minus_alphas_cumprod, t, y_t.shape)
        y0 = sqrt_ab * y_t - sqrt_omab * v
        return torch.clamp(y0, -5.0, 5.0)

    def _predict_eps_from_x0(self, y_t, t, y0):
        sqrt_ab = extract(self.sqrt_alphas_cumprod, t, y_t.shape)
        sqrt_omab = extract(self.sqrt_one_minus_alphas_cumprod, t, y_t.shape)
        return (y_t - sqrt_ab * y0) / (sqrt_omab + 1e-8)

    def ddim_step(self, y_t, t, t_prev, X, eta=0.0, grad_enabled=None):
        """
        One DDIM step: t -> t_prev (t_prev < t)
        eta=0 deterministic, eta>0 stochastic
        """
        if grad_enabled is None:
            grad_enabled = torch.is_grad_enabled()

        with torch.set_grad_enabled(grad_enabled):
            v = self.predict_v(y_t, t, X)
            y0 = self._predict_x0_from_v(y_t, t, v)
            eps = self._predict_eps_from_x0(y_t, t, y0)

            ab_t = extract(self.alphas_cumprod, t, y_t.shape)          # alpha_bar_t
            ab_prev = extract(self.alphas_cumprod, t_prev, y_t.shape)  # alpha_bar_prev

            eta = float(eta)
            sigma = eta * torch.sqrt(
                (1.0 - ab_prev) / (1.0 - ab_t) * (1.0 - ab_t / ab_prev)
            )
            sigma = torch.clamp(sigma, min=0.0)

            dir_part = torch.sqrt(torch.clamp(1.0 - ab_prev - sigma**2, min=0.0)) * eps
            y_prev = torch.sqrt(ab_prev) * y0 + dir_part

            if eta > 0.0:
                y_prev = y_prev + sigma * torch.randn_like(y_t)

            return y_prev

    @torch.no_grad()
    def ddim_sample(self, X, n_steps=50, eta=0.0):
        """
        DDIM sampling (batch):
          X: [B,24,F]
          return: [B,24]
        """
        self.eval()
        B = X.shape[0]
        device = X.device

        # build timestep schedule (descending), ensure starts at T-1 and ends at 0
        T = int(self.T)
        if n_steps is None or int(n_steps) >= T:
            ts = list(range(T - 1, -1, -1))
        else:
            n_steps = max(1, int(n_steps))
            idx = torch.linspace(0, T - 1, n_steps, device=device).round().long().tolist()
            ts = list(sorted(set(idx), reverse=True))
            if ts[0] != T - 1:
                ts = [T - 1] + ts
            if ts[-1] != 0:
                ts = ts + [0]

        y_t = torch.randn(B, self.y_dim, device=device)

        for k, ti in enumerate(ts):
            t = torch.full((B,), ti, device=device, dtype=torch.long)

            if ti == 0:
                v = self.predict_v(y_t, t, X)
                y_t = self._predict_x0_from_v(y_t, t, v)
                break

            ti_prev = ts[k + 1]
            t_prev = torch.full((B,), ti_prev, device=device, dtype=torch.long)
            y_t = self.ddim_step(y_t, t, t_prev, X, eta=eta, grad_enabled=False)

        return y_t


class NodeWiseLagGating(nn.Module):
    """
    Node-wise gating for lag_all.
    Given lag_all [B,24,N], common [B,24,Fc], and current y_t [B,24,N],
    produce a gate g [B,24,N] in [0,1] to reweight lag_all without collapsing node dimension.

    lag_used = lag_all * (1 + alpha * (2g - 1))   # scale in [1-alpha, 1+alpha]
    """
    def __init__(self, n_nodes, common_dim, hidden=64, alpha=0.5):
        super().__init__()
        self.n_nodes = int(n_nodes)
        self.common_dim = int(common_dim)
        self.alpha = float(alpha)

        # score per node per time step
        self.mlp = nn.Sequential(
            nn.Linear(1 + 1 + self.common_dim, hidden),  # [lag, y_t, common]
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, lag_all, common, y_t):
        """
        lag_all: [B,24,N]
        common : [B,24,Fc]
        y_t    : [B,24,N]
        return:
          lag_used: [B,24,N]
          gate    : [B,24,N] (sigmoid)
        """
        B, L, N = lag_all.shape
        assert N == self.n_nodes
        assert common.shape[:2] == (B, L)
        assert y_t.shape == (B, L, N)

        # [B,24,N,Fc]
        common_exp = common.unsqueeze(2).expand(B, L, N, self.common_dim)
        # [B,24,N,1]
        lag_exp = lag_all.unsqueeze(-1)
        yt_exp = y_t.unsqueeze(-1)

        z = torch.cat([lag_exp, yt_exp, common_exp], dim=-1)   # [B,24,N,2+Fc]
        gate = torch.sigmoid(self.mlp(z).squeeze(-1))          # [B,24,N] in (0,1)

        # scale around 1.0 to avoid killing signal early
        scale = 1.0 + self.alpha * (2.0 * gate - 1.0)          # [B,24,N] in [1-a, 1+a]
        lag_used = lag_all * scale
        return lag_used, gate



class NodeWiseBottleneckFiLM(nn.Module):
    """
    Node-wise FiLM at bottleneck.

    Input:
      m      : [B, H, Lb]  (Lb=6)
      ht     : [B, Ht]
      common : [B, 24, Fc]
    Steps:
      m -> expand to per-node: mn [B, N, H, Lb]
      produce node-wise gamma/beta: [B,N,H]
      FiLM on mn
      aggregate back to: [B,H,Lb]
    """
    def __init__(self, hidden_ch, time_hidden, common_dim, n_nodes,
                 node_emb_dim=32, film_scale=0.1):
        super().__init__()
        self.hidden_ch = int(hidden_ch)
        self.time_hidden = int(time_hidden)
        self.common_dim = int(common_dim)
        self.n_nodes = int(n_nodes)
        self.node_emb_dim = int(node_emb_dim)
        self.film_scale = float(film_scale)

        self.node_emb = nn.Embedding(self.n_nodes, self.node_emb_dim)

        self.to_node = nn.Conv1d(self.hidden_ch, self.hidden_ch * self.n_nodes, kernel_size=1)

        in_dim = self.time_hidden + self.common_dim + self.node_emb_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, self.hidden_ch * 2),
            nn.SiLU(),
            nn.Linear(self.hidden_ch * 2, self.hidden_ch * 2),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

        self.from_node = nn.Conv1d(self.hidden_ch * self.n_nodes, self.hidden_ch, kernel_size=1)

    def forward(self, m, ht, common):
        B, H, Lb = m.shape
        assert H == self.hidden_ch
        common_pool = common.mean(dim=1)  # [B,Fc]

        mn = self.to_node(m)  # [B, H*N, Lb]
        mn = mn.view(B, self.n_nodes, self.hidden_ch, Lb)  # [B,N,H,Lb]

        node_ids = torch.arange(self.n_nodes, device=m.device)  # [N]
        ne = self.node_emb(node_ids).unsqueeze(0).expand(B, -1, -1)  # [B,N,Emb]

        ht_exp = ht.unsqueeze(1).expand(B, self.n_nodes, self.time_hidden)          # [B,N,Ht]
        cc_exp = common_pool.unsqueeze(1).expand(B, self.n_nodes, self.common_dim)  # [B,N,Fc]
        z = torch.cat([ht_exp, cc_exp, ne], dim=-1)  # [B,N,in_dim]

        gb = self.mlp(z)  # [B,N,2H]
        gamma, beta = gb.chunk(2, dim=-1)  # [B,N,H], [B,N,H]

        mn = mn * (1.0 + self.film_scale * gamma.unsqueeze(-1)) + self.film_scale * beta.unsqueeze(-1)

        mn_flat = mn.view(B, self.n_nodes * self.hidden_ch, Lb)  # [B,N*H,Lb]
        m_out = self.from_node(mn_flat)  # [B,H,Lb]
        return m_out


class DenoiserUNet_Joint(nn.Module):
    """
    Inputs:
      y_t  : [B,24,N]
      cond : [B,24,Fcond] where Fcond = Fc + N
      t    : [B]
    Output:
      v_hat: [B,24,N]
    """
    def __init__(self, n_nodes, cond_dim, time_dim=128, hidden_ch=256, dropout=0.0, groups=8,
                 film_node_emb_dim=32, film_scale=0.1):
        super().__init__()
        self.n_nodes = int(n_nodes)
        self.cond_dim = int(cond_dim)

        self.Fc = 6
        assert self.cond_dim == self.Fc + self.n_nodes, \
            f"Expected cond_dim == Fc + N == {self.Fc}+{self.n_nodes}={self.Fc+self.n_nodes}, got {self.cond_dim}"

        self.lag_gate = NodeWiseLagGating(
            n_nodes=self.n_nodes,
            common_dim=self.Fc,
            hidden=64,
            alpha=0.5,
        )
        self.last_lag_gate = None

        self.time_emb = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_ch),
            nn.SiLU(),
            nn.Linear(hidden_ch, hidden_ch),
        )
        self.time_to_channel = nn.Linear(hidden_ch, 1)

        in_ch = self.n_nodes + self.cond_dim + 1
        self.in_conv = nn.Conv1d(in_ch, hidden_ch, kernel_size=3, padding=1)

        # encoder: 24 -> 12 -> 6
        self.enc1 = ResBlock1D(hidden_ch, hidden_ch, hidden_ch, dropout=dropout, groups=groups)
        self.down1 = Down1D(hidden_ch)
        self.enc2 = ResBlock1D(hidden_ch, hidden_ch, hidden_ch, dropout=dropout, groups=groups)
        self.down2 = Down1D(hidden_ch)

        # bottleneck @ 6
        self.mid1 = ResBlock1D(hidden_ch, hidden_ch, hidden_ch, dropout=dropout, groups=groups)
        self.mid2 = ResBlock1D(hidden_ch, hidden_ch, hidden_ch, dropout=dropout, groups=groups)

        self.node_film = NodeWiseBottleneckFiLM(
            hidden_ch=hidden_ch,
            time_hidden=hidden_ch,
            common_dim=self.Fc,
            n_nodes=self.n_nodes,
            node_emb_dim=film_node_emb_dim,
            film_scale=film_scale,
        )

        # decoder: 6 -> 12 -> 24
        self.up2 = Up1D(hidden_ch)
        self.dec2 = ResBlock1D(hidden_ch * 2, hidden_ch, hidden_ch, dropout=dropout, groups=groups)
        self.up1 = Up1D(hidden_ch)
        self.dec1 = ResBlock1D(hidden_ch * 2, hidden_ch, hidden_ch, dropout=dropout, groups=groups)

        self.out_norm = nn.GroupNorm(groups, hidden_ch)
        self.out_conv = nn.Conv1d(hidden_ch, self.n_nodes, kernel_size=3, padding=1)
        self.act = nn.SiLU()

    def forward(self, y_t, t, cond):
        B, L, N = y_t.shape
        assert L == 24 and N == self.n_nodes
        assert cond.shape[0] == B and cond.shape[1] == L

        ht = self.time_mlp(self.time_emb(t))  # [B,hidden_ch]
        tc = self.time_to_channel(ht).unsqueeze(-1).expand(B, 1, L)  # [B,1,24]

        common = cond[:, :, :self.Fc]      # [B,24,6]
        lag_all = cond[:, :, self.Fc:]     # [B,24,N]

        lag_used, gate = self.lag_gate(lag_all, common, y_t)
        self.last_lag_gate = gate.detach()

        # lag_used = lag_all
        # self.last_lag_gate = None

        cond_used = torch.cat([common, lag_used], dim=-1)       # [B,24,Fc+N]

        ych = y_t.permute(0, 2, 1).contiguous()       # [B,N,24]
        cch = cond_used.permute(0, 2, 1).contiguous() # [B,Fcond,24]
        x = torch.cat([ych, cch, tc], dim=1)          # [B,in_ch,24]

        x = self.in_conv(x)                           # [B,H,24]
        e1 = self.enc1(x, ht)                         # [B,H,24]
        d1 = self.down1(e1)                           # [B,H,12]
        e2 = self.enc2(d1, ht)                        # [B,H,12]
        d2 = self.down2(e2)                           # [B,H,6]

        m = self.mid2(self.mid1(d2, ht), ht)          # [B,H,6]
        m = self.node_film(m, ht, common)             # [B,H,6]

        u2 = self.up2(m)                              # [B,H,12]
        u2 = torch.cat([u2, e2], dim=1)               # [B,2H,12]
        u2 = self.dec2(u2, ht)                        # [B,H,12]

        u1 = self.up1(u2)                             # [B,H,24]
        u1 = torch.cat([u1, e1], dim=1)               # [B,2H,24]
        u1 = self.dec1(u1, ht)                        # [B,H,24]

        out = self.out_conv(self.act(self.out_norm(u1)))  # [B,N,24]
        return out.permute(0, 2, 1).contiguous()           # [B,24,N]


class ConditionalDDPM_Joint(nn.Module):
    def __init__(self, n_nodes, cond_dim, T=200, time_dim=64, hidden_ch=128, n_layers=6, dropout=0.0,
                 beta_schedule="cosine"):
        super().__init__()
        self.n_nodes = int(n_nodes)
        self.y_dim = self.n_nodes
        self.cond_dim = int(cond_dim)
        self.T = int(T)

        if beta_schedule == "cosine":
            betas = cosine_beta_schedule(self.T)
        elif beta_schedule == "linear":
            betas = torch.linspace(1e-4, 0.02, self.T)
        else:
            raise ValueError("beta_schedule must be 'cosine' or 'linear'")

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.], dtype=torch.float32), alphas_cumprod[:-1]], dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod))

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance.clamp(min=1e-20))
        self.register_buffer("posterior_log_variance_clipped",
                             torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer("posterior_mean_coef1",
                             betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer("posterior_mean_coef2",
                             (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        self.denoiser = DenoiserUNet_Joint(
            n_nodes=self.n_nodes,
            cond_dim=self.cond_dim,
            time_dim=time_dim,
            hidden_ch=hidden_ch,
            dropout=dropout,
            groups=8,
            film_node_emb_dim=32,
            film_scale=0.01,
        )

    def q_sample(self, y0, t, noise=None):
        noise = default(noise, torch.randn_like(y0))
        sqrt_ab = extract(self.sqrt_alphas_cumprod, t, y0.shape)
        sqrt_omab = extract(self.sqrt_one_minus_alphas_cumprod, t, y0.shape)
        return sqrt_ab * y0 + sqrt_omab * noise

    def predict_v(self, y_t, t, cond):
        return self.denoiser(y_t, t, cond)

    def p_mean_variance(self, y_t, t, cond):
        v = self.predict_v(y_t, t, cond)

        sqrt_ab = extract(self.sqrt_alphas_cumprod, t, y_t.shape)
        sqrt_omab = extract(self.sqrt_one_minus_alphas_cumprod, t, y_t.shape)

        y0_hat = sqrt_ab * y_t - sqrt_omab * v
        y0_hat = torch.clamp(y0_hat, -5.0, 5.0)

        coef1 = extract(self.posterior_mean_coef1, t, y_t.shape)
        coef2 = extract(self.posterior_mean_coef2, t, y_t.shape)
        mean = coef1 * y0_hat + coef2 * y_t

        logvar = extract(self.posterior_log_variance_clipped, t, y_t.shape)
        return mean, logvar

    def p_sample(self, y_t, t, cond, grad_enabled=None):
        if grad_enabled is None:
            grad_enabled = torch.is_grad_enabled()
    
        with torch.set_grad_enabled(grad_enabled):
            mean, logvar = self.p_mean_variance(y_t, t, cond)
            if (t == 0).all():
                return mean
            noise = torch.randn_like(y_t)
            return mean + torch.exp(0.5 * logvar) * noise

    def loss(self, cond, y0):
        B = cond.shape[0]
        device = cond.device
        t = torch.randint(0, self.T, (B,), device=device).long()
        noise = torch.randn_like(y0)
        y_t = self.q_sample(y0, t, noise=noise)

        sqrt_ab = extract(self.sqrt_alphas_cumprod, t, y0.shape)
        sqrt_omab = extract(self.sqrt_one_minus_alphas_cumprod, t, y0.shape)

        v = sqrt_ab * noise - sqrt_omab * y0
        v_pred = self.predict_v(y_t, t, cond)
        #return F.mse_loss(v_pred, v, reduction="mean")
        return (v_pred - v).pow(2).mean(dim=(0, 1)).mean()

    # =================================================
    # DDIM additions (sampling only; training unchanged)
    # =================================================
    def _predict_x0_from_v(self, y_t, t, v):
        sqrt_ab = extract(self.sqrt_alphas_cumprod, t, y_t.shape)
        sqrt_omab = extract(self.sqrt_one_minus_alphas_cumprod, t, y_t.shape)
        x0 = sqrt_ab * y_t - sqrt_omab * v
        return torch.clamp(x0, -5.0, 5.0)

    def _predict_eps_from_x0(self, y_t, t, x0):
        sqrt_ab = extract(self.sqrt_alphas_cumprod, t, y_t.shape)
        sqrt_omab = extract(self.sqrt_one_minus_alphas_cumprod, t, y_t.shape)
        eps = (y_t - sqrt_ab * x0) / (sqrt_omab + 1e-8)
        return eps

    def ddim_step(self, y_t, t, t_prev, cond, eta=0.0, grad_enabled=None):
        if grad_enabled is None:
            grad_enabled = torch.is_grad_enabled()

        with torch.set_grad_enabled(grad_enabled):
            v = self.predict_v(y_t, t, cond)
            x0 = self._predict_x0_from_v(y_t, t, v)
            eps = self._predict_eps_from_x0(y_t, t, x0)

            ab_t = extract(self.alphas_cumprod, t, y_t.shape)
            ab_prev = extract(self.alphas_cumprod, t_prev, y_t.shape)

            eta = float(eta)
            sigma = eta * torch.sqrt((1 - ab_prev) / (1 - ab_t) * (1 - ab_t / ab_prev))
            sigma = torch.clamp(sigma, min=0.0)

            dir_part = torch.sqrt(torch.clamp(1 - ab_prev - sigma**2, min=0.0)) * eps
            y_prev = torch.sqrt(ab_prev) * x0 + dir_part

            if eta > 0.0:
                y_prev = y_prev + sigma * torch.randn_like(y_t)

            return y_prev

    def ddim_sample(self, cond, n_steps=None, eta=0.0, shared_noise=True, grad_enabled=None):
        self.eval()
        if grad_enabled is None:
            grad_enabled = torch.is_grad_enabled()
        device = cond.device
        B = cond.shape[0]
        N = self.n_nodes

        n_steps = self.T if n_steps is None else int(n_steps)
        n_steps = max(1, min(n_steps, self.T))

        if n_steps == self.T:
            ts = list(range(self.T - 1, -1, -1))
        else:
            idx = torch.linspace(0, self.T - 1, n_steps, device=device).round().long().tolist()
            ts = list(sorted(set(idx), reverse=True))
            if ts[0] != self.T - 1:
                ts = [self.T - 1] + ts
            if ts[-1] != 0:
                ts = ts + [0]

        with torch.set_grad_enabled(grad_enabled):
            if shared_noise:
                base = torch.randn(B, 24, 1, device=device)
                y = base.expand(B, 24, N).contiguous()
            else:
                y = torch.randn(B, 24, N, device=device)

            for k, ti in enumerate(ts):
                t = torch.full((B,), ti, device=device, dtype=torch.long)

                if ti == 0:
                    v = self.predict_v(y, t, cond)
                    y = self._predict_x0_from_v(y, t, v)
                    break

                ti_prev = ts[k + 1]
                t_prev = torch.full((B,), ti_prev, device=device, dtype=torch.long)
                y = self.ddim_step(y, t, t_prev, cond, eta=eta, grad_enabled=grad_enabled)

        return y
