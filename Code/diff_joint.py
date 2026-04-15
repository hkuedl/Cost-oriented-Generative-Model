import os
import math
import random

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from utils import *
from diffusion import *
from data_loader import *
from combined_data_loader import *
from tqdm import tqdm
from Optimization_multi_node import *
from Optimization_single_node import *
from scenarios_reduce import *

import os
import copy
import numpy as np
import torch
from torch.utils.data import DataLoader

@torch.no_grad()
def pinball_loss_multiq(y_true, y_samp, quantiles=(0.1, 0.5, 0.9)):
    """
    y_true: [B,T,N]
    y_samp: [S,B,T,N]
    return: scalar avg pinball loss
    """
    qs = torch.tensor(list(quantiles), device=y_true.device, dtype=y_true.dtype)  # [Q]
    y_q = torch.quantile(y_samp, qs, dim=0)  # [Q,B,T,N]

    diff = y_true.unsqueeze(0) - y_q  # [Q,B,T,N]
    q_view = qs.view(-1, 1, 1, 1)
    loss = torch.maximum(q_view * diff, (q_view - 1.0) * diff)
    return loss.mean().item()


class Runner_diffusion_joint:
    def __init__(
        self,
        train_set,
        val_set,
        test_set,
        T=200,
        time_dim=64,
        hidden_ch=128,
        n_layers=6,
        dropout=0.0,
        lr=2e-4,
        device=None,
    ):
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        n_nodes = int(train_set.y.shape[-1])
        cond_dim = int(train_set.cond.shape[-1])

        self.model = ConditionalDDPM_Joint(
            n_nodes=n_nodes,
            cond_dim=cond_dim,
            T=T,
            time_dim=time_dim,
            hidden_ch=hidden_ch,
            n_layers=n_layers,
            dropout=dropout,
            beta_schedule="cosine",
        ).to(self.device)

        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.best_z_temp = None
        self.best_val_pinball_real = None
        self.ztemp_table = None

    def fit(
        self,
        epochs=2000,
        batch_size=128,
        patience=80,
        best_path="best.pt",
        verbose=False,
        grad_clip=1.0,
    ):
        train_loader = DataLoader(self.train_set, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(self.val_set, batch_size=batch_size, shuffle=False, drop_last=False)

        best_val = float("inf")
        bad = 0

        for ep in range(1, epochs + 1):
            self.model.train()
            tr = []

            for cond, y in train_loader:
                cond = cond.to(self.device)
                y = y.to(self.device)

                loss = self.model.loss(cond, y)

                self.opt.zero_grad()
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(grad_clip))
                self.opt.step()

                tr.append(float(loss.item()))

            self.model.eval()
            va = []
            with torch.no_grad():
                for cond, y in val_loader:
                    cond = cond.to(self.device)
                    y = y.to(self.device)
                    va.append(float(self.model.loss(cond, y).item()))

            val_loss = float(np.mean(va)) if len(va) else float("inf")

            if verbose and (ep == 1 or ep % 10 == 0):
                print(f"Epoch {ep:4d} | train={np.mean(tr):.6f} | val={val_loss:.6f}")

            if val_loss < best_val - 1e-8:
                best_val = val_loss
                bad = 0
                torch.save(self.model.state_dict(), best_path)
            else:
                bad += 1
                if bad >= patience:
                    if verbose:
                        print(f"Early stop at epoch {ep}, best val={best_val:.6f}")
                    break

        self.model.load_state_dict(torch.load(best_path, map_location=self.device))
        self.model.eval()
        return self

    @torch.no_grad()
    def select_best_ztemp_by_pinball(
        self,
        ztemps,
        quantiles=(0.1, 0.5, 0.9),
        n_samples=50,
        batch_size=64,
        sample_chunk=20,
        mode="ddim",
        n_steps=50,
        eta=0.0,
        shared_noise=True,
        verbose=True,
    ):
        self.model.eval()
        loader = DataLoader(self.val_set, batch_size=batch_size, shuffle=False, drop_last=False)

        def inv_y(ds, y):
            try:
                out = ds.inverse_transform_y(y)
                if isinstance(out, torch.Tensor):
                    return out.to(self.device).float()
                else:
                    return torch.as_tensor(out, device=self.device, dtype=torch.float32)
            except Exception:
                if isinstance(y, torch.Tensor):
                    y_np = y.detach().cpu().numpy()
                else:
                    y_np = np.asarray(y)

                if y_np.ndim == 3:
                    out_np = ds.inverse_transform_y(y_np)
                    return torch.as_tensor(out_np, device=self.device, dtype=torch.float32)

                if y_np.ndim == 4:
                    outs = [ds.inverse_transform_y(y_np[s]) for s in range(y_np.shape[0])]
                    out_np = np.stack(outs, axis=0)
                    return torch.as_tensor(out_np, device=self.device, dtype=torch.float32)

                raise ValueError(f"Unsupported y ndim={y_np.ndim}, shape={y_np.shape}")

        best_t, best_score = None, float("inf")
        table = []

        for t in list(ztemps):
            scores = []

            for cond, y in loader:
                cond = cond.to(self.device).float()
                y = y.to(self.device).float()

                if mode == "ddpm":
                    samp = ddpm_sample_parallel_chunked_joint(
                        self.model,
                        cond,
                        n_samples=int(n_samples),
                        sample_chunk=int(sample_chunk),
                        shared_noise=bool(shared_noise),
                        z_temp=float(t),
                    )  # [S,B,N,24]
                elif mode == "ddim":
                    samp = ddim_sample_parallel_chunked_joint(
                        self.model,
                        cond,
                        n_samples=int(n_samples),
                        n_steps=int(n_steps),
                        eta=float(eta),
                        sample_chunk=int(sample_chunk),
                        shared_noise=bool(shared_noise),
                        z_temp=float(t),
                    )  # [S,B,N,24]
                else:
                    raise ValueError(f"Unknown mode: {mode}")

                samp = samp.permute(0, 1, 3, 2).contiguous()  # [S,B,24,N]

                Y_real = inv_y(self.val_set, y)
                samp_real = inv_y(self.val_set, samp)

                pb = pinball_loss_multiq(Y_real, samp_real, quantiles=quantiles)
                scores.append(pb)

            score = float(np.mean(scores))
            table.append((float(t), score))

            if verbose:
                print(f"[z_temp={t:>6}] val(real-scale) pinball{tuple(quantiles)} = {score:.6f}")

            if score < best_score:
                best_score = score
                best_t = float(t)

        self.best_z_temp = best_t
        self.best_val_pinball_real = best_score
        self.ztemp_table = table

        if verbose:
            print(f"best z_temp = {best_t} | best val(real-scale) pinball = {best_score:.6f}")

        return best_t, best_score, table

def run_diffusion_joint(
    data_path,
    node_cols,
    device=None,
    epochs=2000,
    batch_size=128,
    lr=2e-4,
    patience=80,
    ckpt_dir="../Model/Diffusion/ckpt_nodes_diffusion_joint_unet",
    verbose=True,
    train_length=4296,
    val_ratio=0.2,
    seed=42,
    T=200,
    time_dim=64,
    hidden_ch=128,
    n_layers=6,
    dropout=0.0,
    find_best_ztemp=True,
    z_temp_grid=(0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0),
    ztemp_search_quantiles=(0.1, 0.5, 0.9),
    ztemp_search_n_samples=50,
    ztemp_search_batch_size=64,
    ztemp_search_sample_chunk=20,
    ztemp_search_mode="ddim",
    ztemp_search_n_steps=50,
    ztemp_search_eta=0.0,
    ztemp_search_shared_noise=True,
):
    os.makedirs(ckpt_dir, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    train_set = Dataset_load_multi_node_diff(
        data_path=data_path, node_cols=node_cols, flag="train",
        train_length=train_length, val_ratio=val_ratio, seed=seed
    )
    val_set = Dataset_load_multi_node_diff(
        data_path=data_path, node_cols=node_cols, flag="val",
        train_length=train_length, val_ratio=val_ratio, seed=seed
    )
    test_set = Dataset_load_multi_node_diff(
        data_path=data_path, node_cols=node_cols, flag="test",
        train_length=train_length, val_ratio=val_ratio, seed=seed
    )

    runner = Runner_diffusion_joint(
        train_set=train_set,
        val_set=val_set,
        test_set=test_set,
        T=T,
        time_dim=time_dim,
        hidden_ch=hidden_ch,
        n_layers=n_layers,
        dropout=dropout,
        lr=lr,
        device=device,
    )

    best_path = os.path.join(ckpt_dir, f"best_joint_{len(node_cols)}nodes.pt")
    runner.fit(
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        best_path=best_path,
        verbose=verbose,
    )

    best_z_temp = 1.0
    best_val_pinball_real = None
    ztemp_table = None

    if bool(find_best_ztemp):
        best_z_temp, best_val_pinball_real, ztemp_table = runner.select_best_ztemp_by_pinball(
            ztemps=z_temp_grid,
            quantiles=ztemp_search_quantiles,
            n_samples=int(ztemp_search_n_samples),
            batch_size=int(ztemp_search_batch_size),
            sample_chunk=int(ztemp_search_sample_chunk),
            mode=str(ztemp_search_mode),
            n_steps=int(ztemp_search_n_steps),
            eta=float(ztemp_search_eta),
            shared_noise=bool(ztemp_search_shared_noise),
            verbose=verbose,
        )
        try:
            runner.best_val_pinball_real = best_val_pinball_real
            runner.ztemp_table = ztemp_table
        except Exception:
            pass

    joint_model = runner.model.to(device).eval()

    def _mk_splits(train_set, val_set, test_set):
        return {
            "train": {"cond": train_set.cond, "y": train_set.y},
            "val":   {"cond": val_set.cond,   "y": val_set.y},
            "test":  {"cond": test_set.cond,  "y": test_set.y},
        }

    models_s = {node: joint_model for node in node_cols}
    handlers_s = {node: {"train": train_set, "val": val_set, "test": test_set} for node in node_cols}
    pack_data_s = {node: {"splits": _mk_splits(train_set, val_set, test_set)} for node in node_cols}

    models_s["_JOINT_"] = joint_model
    handlers_s["_JOINT_"] = {"train": train_set, "val": val_set, "test": test_set}
    pack_data_s["_JOINT_"] = {
        "splits": _mk_splits(train_set, val_set, test_set),
        "node_cols": list(node_cols),
        "runner": runner,
        "best_path": best_path,
        "best_z_temp": (None if best_z_temp is None else float(best_z_temp)),
        "best_val_pinball_real": (None if best_val_pinball_real is None else float(best_val_pinball_real)),
        "ztemp_table": ztemp_table,
        "ztemp_quantiles": tuple(ztemp_search_quantiles),
    }

    return models_s, handlers_s, pack_data_s

def ddpm_sample_parallel_chunked_joint(
    model,
    cond,
    n_samples=50,
    sample_chunk=20,
    shared_noise=True,
    grad_enabled=None,
    cpu_offload=False,
    trunc_steps=10,
    force_fp32=True,
    return_BNT=True,
    z_temp=1.0,
):
    if grad_enabled is None:
        grad_enabled = torch.is_grad_enabled()
    if grad_enabled and cpu_offload:
        raise ValueError("Training (grad_enabled=True) requires cpu_offload=False.")

    model.eval()
    device = cond.device
    B, T, F = cond.shape
    assert T == 24
    S = int(n_samples)
    N = int(model.n_nodes)

    if force_fp32 and cond.dtype != torch.float32:
        cond = cond.float()

    z_temp = float(z_temp)
    K = max(0, min(int(trunc_steps), int(model.T)))

    outs = []
    for s0 in range(0, S, int(sample_chunk)):
        s1 = min(S, s0 + int(sample_chunk))
        sc = s1 - s0

        cond_rep = cond.repeat(sc, 1, 1)
        dtype = cond_rep.dtype

        if shared_noise:
            base = torch.randn(sc * B, 24, 1, device=device, dtype=dtype) * z_temp
            y_t = base.expand(sc * B, 24, N).contiguous()
        else:
            y_t = torch.randn(sc * B, 24, N, device=device, dtype=dtype) * z_temp

        with torch.no_grad():
            for ti in range(model.T - 1, K - 1, -1):
                t = torch.full((sc * B,), ti, device=device, dtype=torch.long)
                y_t = model.p_sample(y_t, t, cond_rep, grad_enabled=False, z_temp=z_temp)

        with torch.set_grad_enabled(grad_enabled):
            for ti in range(K - 1, -1, -1):
                t = torch.full((sc * B,), ti, device=device, dtype=torch.long)
                y_t = model.p_sample(y_t, t, cond_rep, grad_enabled=grad_enabled, z_temp=z_temp)

        y_out = y_t.view(sc, B, 24, N)

        if return_BNT:
            y_out = y_out.permute(0, 1, 3, 2).contiguous()
        else:
            y_out = y_out.contiguous()

        if cpu_offload:
            y_out = y_out.cpu()

        outs.append(y_out)

        del cond_rep, y_t, y_out
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return torch.cat(outs, dim=0)

def ddim_sample_parallel_chunked_joint(
    model,
    cond,
    n_samples=50,
    n_steps=50,
    eta=0.0,
    sample_chunk=20,
    shared_noise=True,
    grad_enabled=None,
    cpu_offload=False,
    trunc_steps=10,
    force_fp32=True,
    z_temp=1.0,
    return_BNT=True,
):
    if grad_enabled is None:
        grad_enabled = torch.is_grad_enabled()
    if grad_enabled and cpu_offload:
        raise ValueError("Training (grad_enabled=True) requires cpu_offload=False.")

    model.eval()
    device = cond.device
    B, T, F = cond.shape
    assert T == 24
    S = int(n_samples)
    N = int(model.n_nodes)
    Tdiff = int(model.T)

    if force_fp32 and cond.dtype != torch.float32:
        cond = cond.float()

    z_temp = float(z_temp)

    if n_steps is None or int(n_steps) >= Tdiff:
        ts = list(range(Tdiff - 1, -1, -1))
    else:
        n_steps = max(1, int(n_steps))
        idx = torch.linspace(0, Tdiff - 1, n_steps, device=device).round().long().tolist()
        ts = list(sorted(set(idx), reverse=True))
        if len(ts) == 0:
            ts = [Tdiff - 1]
        if ts[0] != Tdiff - 1:
            ts = [Tdiff - 1] + ts
        if ts[-1] != 0:
            ts = ts + [0]

    if len(ts) < 2:
        raise ValueError(f"Bad DDIM schedule: ts={ts} (len<2). T={Tdiff}, n_steps={n_steps}")

    n_pairs = len(ts) - 1
    K = max(0, min(int(trunc_steps), n_pairs))

    outs = []
    for s0 in range(0, S, int(sample_chunk)):
        s1 = min(S, s0 + int(sample_chunk))
        sc = s1 - s0

        cond_rep = cond.repeat(sc, 1, 1)
        dtype = cond_rep.dtype

        if shared_noise:
            base = torch.randn(sc * B, 24, 1, device=device, dtype=dtype) * z_temp
            y = base.expand(sc * B, 24, N).contiguous()
        else:
            y = torch.randn(sc * B, 24, N, device=device, dtype=dtype) * z_temp

        with torch.no_grad():
            for k in range(0, n_pairs - K):
                ti, ti_prev = ts[k], ts[k + 1]
                t = torch.full((sc * B,), ti, device=device, dtype=torch.long)
                t_prev = torch.full((sc * B,), ti_prev, device=device, dtype=torch.long)
                y = model.ddim_step(
                    y, t, t_prev, cond_rep,
                    eta=float(eta),
                    grad_enabled=False,
                    z_temp=z_temp,
                )

        with torch.set_grad_enabled(grad_enabled):
            for k in range(n_pairs - K, n_pairs):
                ti, ti_prev = ts[k], ts[k + 1]
                t = torch.full((sc * B,), ti, device=device, dtype=torch.long)
                t_prev = torch.full((sc * B,), ti_prev, device=device, dtype=torch.long)
                y = model.ddim_step(
                    y, t, t_prev, cond_rep,
                    eta=float(eta),
                    grad_enabled=grad_enabled,
                    z_temp=z_temp,
                )

            t0 = torch.full((sc * B,), 0, device=device, dtype=torch.long)
            v = model.predict_v(y, t0, cond_rep)
            y = model._predict_x0_from_v(y, t0, v)

        y_out = y.view(sc, B, 24, N)

        if return_BNT:
            y_out = y_out.permute(0, 1, 3, 2).contiguous()
        else:
            y_out = y_out.contiguous()

        if cpu_offload:
            y_out = y_out.cpu()

        outs.append(y_out)

        del cond_rep, y, y_out
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return torch.cat(outs, dim=0)

@torch.no_grad()
def sample_window_diffusion_joint(
    models_m,
    handlers_m,
    pack_data_m,
    target_nodes,
    horizon_days=7,
    start_day=0,
    n_samples=50,
    seq_len=24,
    sample_chunk=20,
    mode="ddim",
    n_steps=50,
    eta=0.0,
    day_chunk=14,
    split="test",
    z_temp=None,
    shared_noise=True,
):
    node_names = list(target_nodes)
    N = len(node_names)
    split = str(split)

    if z_temp is None:
        if "_JOINT_" in pack_data_m:
            z_temp = pack_data_m["_JOINT_"].get("best_z_temp", None)
        z_temp = 1.0 if z_temp is None else float(z_temp)
    else:
        z_temp = float(z_temp)

    if "_JOINT_" in pack_data_m and "splits" in pack_data_m["_JOINT_"]:
        pack_joint = pack_data_m["_JOINT_"]
        if split not in pack_joint["splits"]:
            raise KeyError(f"split={split} not in pack_data_m['_JOINT_']['splits']")
        cond_all = pack_joint["splits"][split]["cond"]
        y_all = pack_joint["splits"][split]["y"]
        joint_model = models_m["_JOINT_"]
        handler = handlers_m["_JOINT_"][split]
    else:
        any_node = node_names[0]
        cond_all = pack_data_m[any_node]["cond_test"]
        y_all = pack_data_m[any_node]["y_test"]
        joint_model = models_m[any_node]
        handler = handlers_m[any_node]
        split = "test"

    D = int(cond_all.shape[0])
    if start_day < 0 or start_day >= D:
        raise ValueError(f"start_day={start_day} out of range, total_days({split})={D}")

    actual_days = min(int(horizon_days), D - int(start_day))
    L = actual_days * int(seq_len)

    device = next(joint_model.parameters()).device
    joint_model.eval()

    Y_true = np.zeros((N, L), dtype=np.float32)
    Y_pred = np.zeros((int(n_samples), N, L), dtype=np.float32)

    for d0 in range(0, actual_days, int(day_chunk)):
        d1 = min(actual_days, d0 + int(day_chunk))
        B_blk = d1 - d0

        g0 = int(start_day) + d0
        g1 = int(start_day) + d1

        cond_blk = cond_all[g0:g1].to(device, non_blocking=True)
        y_blk = y_all[g0:g1]

        y_real = np.asarray(handler.inverse_transform_y(y_blk))
        Y_true[:, d0*seq_len:d1*seq_len] = y_real.reshape(B_blk * seq_len, N).T.astype(np.float32)

        if mode == "ddpm":
            samples_norm = ddpm_sample_parallel_chunked_joint(
                joint_model,
                cond_blk,
                n_samples=int(n_samples),
                sample_chunk=int(sample_chunk),
                shared_noise=bool(shared_noise),
                z_temp=float(z_temp),
                return_BNT=True,
            )
        elif mode == "ddim":
            samples_norm = ddim_sample_parallel_chunked_joint(
                joint_model,
                cond_blk,
                n_samples=int(n_samples),
                n_steps=int(n_steps),
                eta=float(eta),
                sample_chunk=int(sample_chunk),
                shared_noise=bool(shared_noise),
                z_temp=float(z_temp),
                return_BNT=True,
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

        samples_norm = samples_norm.cpu().numpy()
        samples_norm = np.transpose(samples_norm, (0, 1, 3, 2))
        samples_real = np.asarray(handler.inverse_transform_y(samples_norm))

        block_pred = samples_real.reshape(int(n_samples), B_blk * seq_len, N).transpose(0, 2, 1).astype(np.float32)
        Y_pred[:, :, d0*seq_len:d1*seq_len] = block_pred

        del cond_blk
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    return dict(
        mode=mode,
        split=split,
        target_nodes=node_names,
        start_day=int(start_day),
        horizon_days=int(actual_days),
        seq_len=int(seq_len),
        n_samples=int(n_samples),
        z_temp=float(z_temp),
        Y_true=Y_true,
        Y_pred=Y_pred,
    )



def _resolve_diffusion_z_temp(z_temp=None, pack_data_m=None, args=None, default=1.0):
    if z_temp is not None:
        return float(z_temp)

    if pack_data_m is not None:
        joint = pack_data_m.get("_JOINT_", {})
        best_z = joint.get("best_z_temp", None)
        if best_z is not None:
            return float(best_z)

    if args is not None and hasattr(args, "z_temp"):
        return float(getattr(args, "z_temp"))

    return float(default)



class Joint_diffusion_predictor(nn.Module):
    def __init__(self, joint_model, dataset, device=None, default_z_temp=1.0):
        super().__init__()

        # 保持旧版逻辑：joint_model 传入的是整个 models_m
        self.model = joint_model["_JOINT_"]
        self.N = int(self.model.n_nodes)
        self.device = device
        self.default_z_temp = float(default_z_temp)

        # infer Fc
        if hasattr(self.model, "denoiser") and hasattr(self.model.denoiser, "Fc"):
            self.Fc = int(self.model.denoiser.Fc)
        else:
            self.Fc = int(self.model.cond_dim) - self.N

        # scaler buffers (fp32)
        sc = dataset.scaler_y
        mean = np.asarray(sc.mean_, dtype=np.float32).reshape(self.N)
        scale = np.asarray(sc.scale_, dtype=np.float32).reshape(self.N)
        self.register_buffer("y_mean", torch.from_numpy(mean).view(self.N, 1, 1))    # [N,1,1]
        self.register_buffer("y_scale", torch.from_numpy(scale).view(self.N, 1, 1))  # [N,1,1]

    def inverse_y(self, y_norm):
        # y_norm: [B,N,24] or [B,24,N] -> return [B,N,24]
        if y_norm.shape[1] == 24:
            y_norm = y_norm.permute(0, 2, 1).contiguous()
        mean = self.y_mean.to(y_norm.device).view(1, self.N, 1).to(y_norm.dtype)
        scale = self.y_scale.to(y_norm.device).view(1, self.N, 1).to(y_norm.dtype)
        return y_norm * scale + mean

    def sample(
        self,
        X_all,                   # [B,24,Fc+N]
        n_samples=50,
        sample_chunk=20,
        mode="ddim",
        n_steps=50,
        eta=0.0,
        trunc_steps=10,
        cpu_offload=False,
        grad_enabled=None,
        return_real_scale=True,
        # 兼容接口：有些实现会传 shared_noise；joint 这里接住
        shared_noise=True,
        # NEW
        z_temp=None,
        **kwargs,                # 兜底：避免 unexpected kw
    ):
        if grad_enabled is None:
            grad_enabled = torch.is_grad_enabled()

        if z_temp is None:
            z_temp = self.default_z_temp
        z_temp = float(z_temp)

        if mode == "ddpm":
            Y = ddpm_sample_parallel_chunked_joint(
                self.model,
                X_all,
                n_samples=n_samples,
                sample_chunk=sample_chunk,
                trunc_steps=trunc_steps,
                cpu_offload=cpu_offload,
                grad_enabled=grad_enabled,
                shared_noise=shared_noise,
                z_temp=z_temp,
            )  # [S,B,N,24]
        else:
            Y = ddim_sample_parallel_chunked_joint(
                self.model,
                X_all,
                n_samples=n_samples,
                n_steps=n_steps,
                eta=eta,
                sample_chunk=sample_chunk,
                trunc_steps=trunc_steps,
                cpu_offload=cpu_offload,
                grad_enabled=grad_enabled,
                shared_noise=shared_noise,
                z_temp=z_temp,
            )  # [S,B,N,24]

        if not return_real_scale:
            return Y

        mean = self.y_mean.to(Y.device).view(1, 1, self.N, 1).to(Y.dtype)
        scale = self.y_scale.to(Y.device).view(1, 1, self.N, 1).to(Y.dtype)
        return Y * scale + mean

    def forward(self, X_all, **kwargs):
        return self.sample(X_all, **kwargs)

class DFL_model_diffusion_Deterministic_SingleNode(nn.Module):
    def __init__(
        self,
        mgr,
        optnet_DA,
        optnet_RT,
        predictor,
        scenario_filter=None,
        n_scen=50,
        clamp_min=0.0,
        solver="ECOS",
    ):
        super().__init__()
        self.mgr = mgr
        self.optnet_DA = optnet_DA
        self.optnet_RT = optnet_RT
        self.predictor = predictor
        self.scenario_filter = scenario_filter
        self.n_scen = int(n_scen)
        self.clamp_min = clamp_min
        self.solver = solver
        self.optnet_dtype = torch.float64

        b_vals = [mgr.gen_info[g]["b"] for g in mgr.gen_bus_list]
        self.register_buffer("b_G", torch.tensor(b_vals, dtype=self.optnet_dtype))
        self.res_up_ratio = float(getattr(mgr.args, "reserve_up_ratio", 0.05))
        self.res_dn_ratio = float(getattr(mgr.args, "reserve_dn_ratio", 0.02))

    def _canon_y_true_11T(self, y_true, T=24):
        if y_true.dim() != 3:
            raise ValueError(f"y_true must be [B,11,T] or [B,T,11], got {tuple(y_true.shape)}")
        if y_true.shape[1] == 11 and y_true.shape[2] == T:
            return y_true
        if y_true.shape[1] == T and y_true.shape[2] == 11:
            return y_true.permute(0, 2, 1).contiguous()
        raise ValueError(f"y_true must be [B,11,{T}] or [B,{T},11], got {tuple(y_true.shape)}")

    def _realized_total_cost(self, P_DA_GT, R_up_GT, R_dn_GT, omega_true_T, solver):
        bG = self.b_G.to(device=P_DA_GT.device, dtype=self.optnet_dtype)

        stage1_energy_linear = (P_DA_GT * bG[:, None]).sum()
        stage1_reserve = (
            (R_up_GT * (self.res_up_ratio * bG)[:, None]).sum()
            + (R_dn_GT * (self.res_dn_ratio * bG)[:, None]).sum()
        )

        sol_rt = self.optnet_RT(
            R_up_GT.unsqueeze(0),
            R_dn_GT.unsqueeze(0),
            omega_true_T.unsqueeze(0),
            solver=solver,
        )

        rt_obj_true = sol_rt["rt_obj"][0]
        return stage1_energy_linear + stage1_reserve + rt_obj_true, sol_rt

    def forward(
        self,
        X_all,
        y_true,
        hourly_load_min_sys=None,
        hourly_load_max_sys=None,
        eps=None,
        solver=None,
        return_aux=False,
        return_filter_aux=False,
        predictor_n_samples=None,
        predictor_kwargs=None,
        filter_kwargs=None,
    ):
        del hourly_load_min_sys, hourly_load_max_sys, eps

        solver = solver or self.solver
        device = X_all.device
        predictor_kwargs = dict(predictor_kwargs or {})
        filter_kwargs = dict(filter_kwargs or {})

        S = int(predictor_n_samples) if predictor_n_samples is not None else self.n_scen

        Y_scen = self.predictor(
            X_all,
            n_samples=S,
            return_real_scale=True,
            grad_enabled=torch.is_grad_enabled(),
            **predictor_kwargs,
        ).to(device=device, dtype=self.optnet_dtype)

        if self.clamp_min is not None:
            Y_scen = torch.clamp(Y_scen, min=float(self.clamp_min))

        aux_filter = None
        if self.scenario_filter is not None:
            Y_scen, aux_filter = self.scenario_filter(
                Y_scen, is_train=self.training, **filter_kwargs
            )
            Y_scen = Y_scen.to(device=device, dtype=self.optnet_dtype)

        # deterministic forecast
        forecast_sys = Y_scen.sum(dim=2).mean(dim=0)  # [B,T]

        y_true_11 = self._canon_y_true_11T(y_true).to(device=device)
        y_true_real11 = self.predictor.inverse_y(y_true_11).to(device=device, dtype=self.optnet_dtype)
        omega_true = y_true_real11.sum(dim=1) - forecast_sys  # [B,T]

        B = X_all.shape[0]
        loss_list = []

        aux = None
        if return_aux:
            aux = {
                "forecast_sys": [],
                "omega_true": [],
                "det_obj": [],
                "realized_total": [],
                "rt_obj_true": [],
                "P_DA": [],
                "R_up": [],
                "R_dn": [],
            }

        for b in range(B):
            sol_da = self.optnet_DA(
                forecast_sys[b].unsqueeze(0),
                solver=solver,
            )

            P_DA = sol_da["P_DA"][0].to(dtype=self.optnet_dtype)
            R_up = sol_da["R_up"][0].to(dtype=self.optnet_dtype)
            R_dn = sol_da["R_dn"][0].to(dtype=self.optnet_dtype)

            realized_total, sol_rt = self._realized_total_cost(
                P_DA, R_up, R_dn, omega_true[b], solver=solver
            )
            loss_list.append(realized_total)

            if return_aux:
                aux["forecast_sys"].append(forecast_sys[b].detach())
                aux["omega_true"].append(omega_true[b].detach())
                obj = sol_da.get("obj", None)
                if isinstance(obj, torch.Tensor):
                    aux["det_obj"].append(obj[0].detach())
                else:
                    aux["det_obj"].append(torch.tensor(float("nan"), device=device, dtype=self.optnet_dtype))
                aux["realized_total"].append(realized_total.detach())
                aux["rt_obj_true"].append(sol_rt["rt_obj"][0].detach())
                aux["P_DA"].append(P_DA.detach())
                aux["R_up"].append(R_up.detach())
                aux["R_dn"].append(R_dn.detach())

        loss_vec = torch.stack(loss_list, dim=0)

        if return_aux:
            for k in aux:
                aux[k] = torch.stack(aux[k], dim=0)

        if return_aux and not return_filter_aux:
            return loss_vec, aux
        if return_filter_aux and not return_aux:
            return loss_vec, aux_filter
        if return_aux and return_filter_aux:
            return loss_vec, aux, aux_filter
        return loss_vec


class DFL_model_diffusion_Deterministic_MultiNode(nn.Module):
    def __init__(
        self,
        mgr,
        optnet_DA,
        optnet_RT,
        predictor,
        scenario_filter=None,
        n_scen=50,
        clamp_min=0.0,
        solver="ECOS",
    ):
        super().__init__()
        self.mgr = mgr
        self.optnet_DA = optnet_DA
        self.optnet_RT = optnet_RT
        self.predictor = predictor
        self.scenario_filter = scenario_filter
        self.n_scen = int(n_scen)
        self.clamp_min = clamp_min
        self.solver = solver
        self.optnet_dtype = torch.float64

        bus_sched_0 = [b - 1 for b in mgr.default_bus_order]
        self.register_buffer("bus_indices_11_to_14", torch.tensor(bus_sched_0, dtype=torch.long))

        b_vals = [mgr.gen_info[g]["b"] for g in mgr.gen_bus_list]
        self.register_buffer("b_G", torch.tensor(b_vals, dtype=self.optnet_dtype))
        self.res_up_ratio = float(getattr(mgr.args, "reserve_up_ratio", 0.05))
        self.res_dn_ratio = float(getattr(mgr.args, "reserve_dn_ratio", 0.02))

    def _canon_y_true_11T(self, y_true, T=24):
        if y_true.dim() != 3:
            raise ValueError(f"y_true must be [B,11,T] or [B,T,11], got {tuple(y_true.shape)}")
        if y_true.shape[1] == 11 and y_true.shape[2] == T:
            return y_true
        if y_true.shape[1] == T and y_true.shape[2] == 11:
            return y_true.permute(0, 2, 1).contiguous()
        raise ValueError(f"y_true must be [B,11,{T}] or [B,{T},11], got {tuple(y_true.shape)}")

    def _map_11_to_14(self, x_11):
        *prefix, n11, T = x_11.shape
        if n11 != 11:
            raise ValueError(f"expected 11 nodes, got {n11}")
        out = x_11.new_zeros((*prefix, 14, T))
        out[..., self.bus_indices_11_to_14, :] = x_11
        return out

    def forward(
        self,
        X_all,
        y_true,
        hourly_load_min_sys=None,
        hourly_load_max_sys=None,
        eps=None,
        solver=None,
        return_aux=False,
        return_filter_aux=False,
        predictor_n_samples=None,
        predictor_kwargs=None,
        filter_kwargs=None,
    ):
        del hourly_load_min_sys, hourly_load_max_sys, eps

        solver = solver or self.solver
        device, dtype = X_all.device, self.optnet_dtype
        predictor_kwargs = dict(predictor_kwargs or {})
        filter_kwargs = dict(filter_kwargs or {})

        S = int(predictor_n_samples) if predictor_n_samples is not None else self.n_scen

        Y_scen_11 = self.predictor(
            X_all,
            n_samples=S,
            return_real_scale=True,
            grad_enabled=torch.is_grad_enabled(),
            **predictor_kwargs,
        ).to(device=device, dtype=dtype)

        if self.clamp_min is not None:
            Y_scen_11 = torch.clamp(Y_scen_11, min=float(self.clamp_min))

        aux_filter = None
        if self.scenario_filter is not None:
            Y_scen_11, aux_filter = self.scenario_filter(
                Y_scen_11, is_train=self.training, **filter_kwargs
            )
            Y_scen_11 = Y_scen_11.to(device=device, dtype=dtype)

        forecast11 = Y_scen_11.mean(dim=0)  # [B,11,T]
        forecast14 = self._map_11_to_14(forecast11)

        sol_DA = self.optnet_DA(
            forecast14,
            solver=solver,
            return_cost=True,
        )
        P_DA = sol_DA["P_DA"].to(device=device, dtype=dtype)
        R_up = sol_DA["R_up"].to(device=device, dtype=dtype)
        R_dn = sol_DA["R_dn"].to(device=device, dtype=dtype)

        T_len = forecast14.shape[-1]
        y_true_11 = self._canon_y_true_11T(y_true, T=T_len).to(device=device)
        y_true_real11 = self.predictor.inverse_y(y_true_11).to(device=device, dtype=dtype)
        y_true_14 = self._map_11_to_14(y_true_real11)

        omega14_true = y_true_14 - forecast14
        sol_RT = self.optnet_RT(R_up, R_dn, omega14_true, solver=solver)
        rt_obj_true = sol_RT["rt_obj"].to(device=device, dtype=dtype)

        bG = self.b_G.to(device=device, dtype=dtype)
        cost_energy = (P_DA * bG[None, :, None]).sum(dim=(1, 2))
        cost_reserve = (
            (R_up * (self.res_up_ratio * bG)[None, :, None]).sum(dim=(1, 2))
            + (R_dn * (self.res_dn_ratio * bG)[None, :, None]).sum(dim=(1, 2))
        )
        total_realized_cost = cost_energy + cost_reserve + rt_obj_true

        aux = None
        if return_aux:
            aux = {
                "forecast11": forecast11.detach(),
                "forecast14": forecast14.detach(),
                "P_DA": P_DA.detach(),
                "R_up": R_up.detach(),
                "R_dn": R_dn.detach(),
                "rt_obj_true": rt_obj_true.detach(),
                "cost_energy": cost_energy.detach(),
                "cost_reserve": cost_reserve.detach(),
            }
            obj = sol_DA.get("obj", None)
            if isinstance(obj, torch.Tensor):
                aux["det_obj"] = obj.detach().to(device=device, dtype=dtype)

        if return_aux and not return_filter_aux:
            return total_realized_cost, aux
        if return_filter_aux and not return_aux:
            return total_realized_cost, aux_filter
        if return_aux and return_filter_aux:
            return total_realized_cost, aux, aux_filter
        return total_realized_cost




class DFL_model_diffusion_MultiNode(nn.Module):
    def __init__(
        self,
        mgr,
        optnet_DA,
        optnet_RT,
        predictor,
        scenario_filter=None,   # <- added
        n_scen=50,
        clamp_min=0.0,
        solver="ECOS",
    ):
        super().__init__()
        self.mgr = mgr
        self.optnet_DA = optnet_DA
        self.optnet_RT = optnet_RT
        self.predictor = predictor
        self.scenario_filter = scenario_filter   # <- added

        self.n_scen = int(n_scen)
        self.clamp_min = clamp_min
        self.solver = solver
        
        self.optnet_dtype = torch.float64  # 统一数值类型，对齐 VAE

        # 11 -> 14 mapping
        bus_sched_0 = [b - 1 for b in mgr.default_bus_order]  # len=11, in [0,13]
        self.register_buffer("bus_indices_11_to_14", torch.tensor(bus_sched_0, dtype=torch.long))

        # prices 
        b_vals = [mgr.gen_info[g]["b"] for g in mgr.gen_bus_list]
        self.register_buffer("b_G", torch.tensor(b_vals, dtype=self.optnet_dtype))

        self.res_up_ratio = float(getattr(mgr.args, "reserve_up_ratio", 0.05))
        self.res_dn_ratio = float(getattr(mgr.args, "reserve_dn_ratio", 0.02))

    def _map_11_to_14(self, x_11):
        """
        x_11: [..., 11, T] -> [..., 14, T]
        """
        *prefix, n11, T = x_11.shape
        assert n11 == 11, f"expected 11 nodes, got {n11}"
        out = x_11.new_zeros((*prefix, 14, T))
        out[..., self.bus_indices_11_to_14, :] = x_11
        return out

    def forward(
        self,
        X_all,
        y_true,  # normalized, [B,11,24] or [B,24,11]
        solver=None,
        return_aux=False,
        return_filter_aux=False, # <- added
        predictor_n_samples=None,
        predictor_kwargs=None,   
        filter_kwargs=None,      # <- added
    ):
        solver = solver or self.solver
        device = X_all.device
        dtype = self.optnet_dtype
        predictor_kwargs = dict(predictor_kwargs or {})
        filter_kwargs = dict(filter_kwargs or {})

        S = int(predictor_n_samples) if predictor_n_samples is not None else self.n_scen

        # 1) diffusion scenarios: [S,B,11,T] on real scale
        Y_scen = self.predictor(
            X_all,
            n_samples=S,
            return_real_scale=True,
            grad_enabled=torch.is_grad_enabled(),
            **predictor_kwargs
        ).to(device=device, dtype=dtype) # 强制类型对齐

        if self.clamp_min is not None:
            Y_scen = torch.clamp(Y_scen, min=float(self.clamp_min))
            
        # 2) scenario filter hook
        aux_filter = None
        if self.scenario_filter is not None:
            Y_scen, aux_filter = self.scenario_filter(
                Y_scen, is_train=self.training, **filter_kwargs
            )
            # filter 后再次对齐，避免内部产生 device/dtype 漂移
            Y_scen = Y_scen.to(device=device, dtype=dtype)

        # 3) SAA inputs
        forecast11 = Y_scen.mean(dim=0)  # [B,11,T]
        scen11 = Y_scen.permute(1, 0, 2, 3).contiguous()  # [B,S,11,T]

        forecast14 = self._map_11_to_14(forecast11)  # [B,14,T]
        scen14 = self._map_11_to_14(scen11)          # [B,S,14,T]
        omega14 = scen14 - forecast14.unsqueeze(1)   # [B,S,14,T]

        # 4) DA
        sol_DA = self.optnet_DA(
            forecast14,
            omega14,
            solver=solver,
            return_rt=False,
            return_cost=True,
        )
        P_DA = sol_DA["P_DA"].to(device=device, dtype=dtype)
        R_up = sol_DA["R_up"].to(device=device, dtype=dtype)
        R_dn = sol_DA["R_dn"].to(device=device, dtype=dtype)

        # 5) RT realized cost with true load
        # y_true_11 需要像 VAE 那样做好 T 和 11 的维数 canon 操作
        T_len = forecast14.shape[-1]
        if y_true.dim() == 3 and y_true.shape[1] == T_len and y_true.shape[2] == 11:
            y_true_11 = y_true.permute(0, 2, 1).contiguous()
        else:
            y_true_11 = y_true

        y_true_11 = y_true_11.to(device=device)
        y_true_real11 = self.predictor.inverse_y(y_true_11).to(device=device, dtype=dtype)  # [B,11,T]
        y_true_14 = self._map_11_to_14(y_true_real11)  # [B,14,T]

        omega14_true = y_true_14 - forecast14
        sol_RT = self.optnet_RT(R_up, R_dn, omega14_true, solver=solver)
        rt_obj_true = sol_RT["rt_obj"].to(device=device, dtype=dtype)  # [B]

        # costs
        bG = self.b_G.to(device=device, dtype=dtype)  # [G]
        cost_energy = (P_DA * bG[None, :, None]).sum(dim=(1, 2))  # [B]

        cost_reserve = (
            (R_up * (self.res_up_ratio * bG)[None, :, None]).sum(dim=(1, 2))
            + (R_dn * (self.res_dn_ratio * bG)[None, :, None]).sum(dim=(1, 2))
        )

        total_realized_cost = cost_energy + cost_reserve + rt_obj_true

        if return_aux:
            aux = {
                "forecast11": forecast11.detach(),
                "forecast14": forecast14.detach(),
                "P_DA": P_DA.detach(),
                "R_up": R_up.detach(),
                "R_dn": R_dn.detach(),
                "rt_obj_true": rt_obj_true.detach(),
                "cost_energy": cost_energy.detach(),
                "cost_reserve": cost_reserve.detach(),
            }
        else:
            aux = None

        # Return Logic Match
        if return_aux and not return_filter_aux: return total_realized_cost, aux
        if return_filter_aux and not return_aux: return total_realized_cost, aux_filter
        if return_aux and return_filter_aux: return total_realized_cost, aux, aux_filter
        return total_realized_cost


class DFL_model_diffusion_SingleNode(nn.Module):
    def __init__(
        self,
        mgr,
        optnet_DA,
        optnet_RT,
        predictor,
        scenario_filter=None,   # <- added
        n_scen=50,
        clamp_min=0.0,
        solver="ECOS",
    ):
        super().__init__()
        self.mgr = mgr
        self.optnet_DA = optnet_DA
        self.optnet_RT = optnet_RT
        self.predictor = predictor
        self.scenario_filter = scenario_filter   # <- added

        self.n_scen = int(n_scen)
        self.clamp_min = clamp_min
        self.solver = solver
        
        self.optnet_dtype = torch.float64

        b_vals = [mgr.gen_info[g]["b"] for g in mgr.gen_bus_list]
        self.register_buffer("b_G", torch.tensor(b_vals, dtype=self.optnet_dtype))

        self.res_up_ratio = float(getattr(mgr.args, "reserve_up_ratio", 0.05))
        self.res_dn_ratio = float(getattr(mgr.args, "reserve_dn_ratio", 0.02))

    def _realized_total_cost(self, P_DA_GT, R_up_GT, R_dn_GT, omega_true_T, solver):
        bG = self.b_G.to(device=P_DA_GT.device, dtype=self.optnet_dtype)  

        stage1_energy_linear = (P_DA_GT * bG[:, None]).sum()
        stage1_reserve = (
            (R_up_GT * (self.res_up_ratio * bG)[:, None]).sum()
            + (R_dn_GT * (self.res_dn_ratio * bG)[:, None]).sum()
        )

        sol_rt = self.optnet_RT(
            R_up_GT.unsqueeze(0),
            R_dn_GT.unsqueeze(0),
            omega_true_T.unsqueeze(0),
            solver=solver,
        )
        rt_obj_true = sol_rt["rt_obj"][0]
        return stage1_energy_linear + stage1_reserve + rt_obj_true, sol_rt

    def forward(
        self,
        X_all,
        y_true,  
        solver=None,
        return_aux=False,
        return_filter_aux=False, # <- added
        predictor_n_samples=None,
        predictor_kwargs=None,
        filter_kwargs=None,      # <- added
    ):
        solver = solver or self.solver
        device = X_all.device
        dtype = self.optnet_dtype
        predictor_kwargs = dict(predictor_kwargs or {})
        filter_kwargs = dict(filter_kwargs or {})

        S = int(predictor_n_samples) if predictor_n_samples is not None else self.n_scen

        Y_scen = self.predictor(
            X_all,
            n_samples=S,
            return_real_scale=True,
            grad_enabled=torch.is_grad_enabled(),
            **predictor_kwargs
        ).to(device=device, dtype=dtype)

        if self.clamp_min is not None:
            Y_scen = torch.clamp(Y_scen, min=float(self.clamp_min))
            
        # Hook Filter
        aux_filter = None
        if self.scenario_filter is not None:
            Y_scen, aux_filter = self.scenario_filter(
                Y_scen, is_train=self.training, **filter_kwargs
            )
            Y_scen = Y_scen.to(device=device, dtype=dtype)

        # 聚合成系统单节点负荷
        scen_sys = Y_scen.sum(dim=2)
        forecast_sys = scen_sys.mean(dim=0)  # [B,T]
        omega_saa = (scen_sys - forecast_sys.unsqueeze(0)).permute(1, 0, 2).contiguous()  # [B,S,T]

        # y_true
        T_len = forecast_sys.shape[-1]
        if y_true.dim() == 3 and y_true.shape[1] == T_len and y_true.shape[2] == 11:
            y_true = y_true.permute(0, 2, 1).contiguous()
            
        y_true = y_true.to(device=device)
        y_true_real11 = self.predictor.inverse_y(y_true).to(dtype=dtype)  
        omega_true = y_true_real11.sum(dim=1) - forecast_sys  # [B,T]

        B = X_all.shape[0]
        loss_list = []

        aux = None
        if return_aux:
            aux = {
                "forecast_sys": [], "omega_true": [], "saa_obj": [],
                "realized_total": [], "rt_obj_true": [],
            }

        for b in range(B):
            sol_da = self.optnet_DA(
                forecast_sys[b].unsqueeze(0),
                omega_saa[b].unsqueeze(0),   
                solver=solver,
                return_rt=False,
            )

            P_DA = sol_da["P_DA"][0].to(dtype=dtype)  
            R_up = sol_da["R_up"][0].to(dtype=dtype)  
            R_dn = sol_da["R_dn"][0].to(dtype=dtype)  

            realized_total, sol_rt = self._realized_total_cost(
                P_DA, R_up, R_dn, omega_true[b], solver=solver
            )
            loss_list.append(realized_total)

            if return_aux:
                aux["forecast_sys"].append(forecast_sys[b].detach())
                aux["omega_true"].append(omega_true[b].detach())
                aux["saa_obj"].append(sol_da.get("obj", torch.tensor(float("nan"), device=device))[0].detach()
                                      if isinstance(sol_da.get("obj", None), torch.Tensor) else
                                      torch.tensor(float("nan"), device=device))
                aux["realized_total"].append(realized_total.detach())
                aux["rt_obj_true"].append(sol_rt["rt_obj"][0].detach())

        loss_vec = torch.stack(loss_list, dim=0)  # [B]

        if return_aux:
            for k in aux:
                aux[k] = torch.stack(aux[k], dim=0)

        # Return Logic Match
        if return_aux and not return_filter_aux: return loss_vec, aux
        if return_filter_aux and not return_aux: return loss_vec, aux_filter
        if return_aux and return_filter_aux: return loss_vec, aux, aux_filter
        return loss_vec


class DFL_model_diffusion_DRO_MultiNode(nn.Module):
    def __init__(
        self,
        mgr,
        optnet_DA,      
        optnet_RT,
        predictor,
        scenario_filter=None,   # <- added
        n_scen=50,
        clamp_min=0.0,
        solver="ECOS",
    ):
        super().__init__()
        self.mgr = mgr
        self.optnet_DA = optnet_DA
        self.optnet_RT = optnet_RT
        self.predictor = predictor
        self.scenario_filter = scenario_filter   # <- added

        self.n_scen = int(n_scen)
        self.clamp_min = clamp_min
        self.solver = solver

        bus_sched_0 = [b - 1 for b in mgr.default_bus_order]  
        self.register_buffer("bus_indices_11_to_14", torch.tensor(bus_sched_0, dtype=torch.long))

        self.optnet_dtype = torch.float64
        
        b_vals = [mgr.gen_info[g]["b"] for g in mgr.gen_bus_list]
        self.register_buffer("b_G", torch.tensor(b_vals, dtype=self.optnet_dtype))

        self.res_up_ratio = float(getattr(mgr.args, "reserve_up_ratio", 0.05))
        self.res_dn_ratio = float(getattr(mgr.args, "reserve_dn_ratio", 0.02))

    def _map_11_to_14(self, x_11):
        *prefix, n11, T = x_11.shape
        assert n11 == 11, f"expected 11 nodes, got {n11}"
        out = x_11.new_zeros((*prefix, 14, T))
        out[..., self.bus_indices_11_to_14, :] = x_11
        return out

    def forward(
        self,
        X_all,
        y_true,                 
        hourly_load_min_sys,    
        hourly_load_max_sys,    
        eps,                    
        solver=None,
        return_aux=False,
        return_filter_aux=False, # <- added
        predictor_n_samples=None,
        predictor_kwargs=None,
        filter_kwargs=None,      # <- added
    ):
        solver = solver or self.solver
        device = X_all.device
        dtype = self.optnet_dtype
        predictor_kwargs = dict(predictor_kwargs or {})
        filter_kwargs = dict(filter_kwargs or {})

        B = X_all.shape[0]
        S = int(predictor_n_samples) if predictor_n_samples is not None else self.n_scen

        Y_scen_11 = self.predictor(
            X_all,
            n_samples=S,
            return_real_scale=True,
            grad_enabled=torch.is_grad_enabled(),
            **predictor_kwargs
        ).to(device=device, dtype=dtype)

        if self.clamp_min is not None:
            Y_scen_11 = torch.clamp(Y_scen_11, min=float(self.clamp_min))

        # Hook Filter
        aux_filter = None
        if self.scenario_filter is not None:
            Y_scen_11, aux_filter = self.scenario_filter(
                Y_scen_11, is_train=self.training, **filter_kwargs
            )
            Y_scen_11 = Y_scen_11.to(device=device, dtype=dtype)

        forecast11 = Y_scen_11.mean(dim=0)               # [B,11,T]
        scen11 = Y_scen_11.permute(1, 0, 2, 3).contiguous() # [B,S,11,T]
        forecast14 = self._map_11_to_14(forecast11)  
        scen14 = self._map_11_to_14(scen11)          
        omega14 = scen14 - forecast14.unsqueeze(1)   

        Lmin14 = torch.as_tensor(hourly_load_min_sys, device=device, dtype=dtype)
        Lmax14 = torch.as_tensor(hourly_load_max_sys, device=device, dtype=dtype)

        if Lmin14.ndim == 2:
            Lmin14 = Lmin14.unsqueeze(0).expand(B, -1, -1).contiguous()
        if Lmax14.ndim == 2:
            Lmax14 = Lmax14.unsqueeze(0).expand(B, -1, -1).contiguous()

        om_min14 = Lmin14 - forecast14
        om_max14 = Lmax14 - forecast14
        om_min14, om_max14 = torch.minimum(om_min14, om_max14), torch.maximum(om_min14, om_max14)

        eps_t = torch.as_tensor(eps, device=device, dtype=dtype)
        if eps_t.ndim == 0:
            eps_t = eps_t.view(1).expand(B)

        sol_DA = self.optnet_DA(
            forecast14, omega14, om_min14, om_max14, eps_t, solver=solver,
        )
        P_DA, R_up, R_dn = sol_DA["P_DA"], sol_DA["R_up"], sol_DA["R_dn"]

        # true load 统一维度
        y_true = y_true.to(device=device)
        T_len = forecast14.shape[-1]
        if y_true.ndim == 3 and y_true.shape[1] == T_len and y_true.shape[2] == 11:
            y_true = y_true.permute(0, 2, 1).contiguous()

        y_true_real11 = self.predictor.inverse_y(y_true).to(device=device, dtype=dtype)
        y_true_14 = self._map_11_to_14(y_true_real11)

        omega14_true = y_true_14 - forecast14
        sol_RT = self.optnet_RT(R_up, R_dn, omega14_true, solver=solver)
        rt_obj_true = sol_RT["rt_obj"].to(device=device, dtype=dtype)

        bG = self.b_G.to(device=device, dtype=dtype)  
        cost_energy = (P_DA * bG[None, :, None]).sum(dim=(1, 2))

        cost_reserve = (
            (R_up * (self.res_up_ratio * bG)[None, :, None]).sum(dim=(1, 2))
            + (R_dn * (self.res_dn_ratio * bG)[None, :, None]).sum(dim=(1, 2))
        )

        total_realized_cost = cost_energy + cost_reserve + rt_obj_true

        if return_aux:
            aux = {
                "forecast11": forecast11.detach(), "forecast14": forecast14.detach(),
                "omega14": omega14.detach(), "om_min14": om_min14.detach(),
                "om_max14": om_max14.detach(), "eps": eps_t.detach(),
                "P_DA": P_DA.detach(), "R_up": R_up.detach(), "R_dn": R_dn.detach(),
                "rt_obj_true": rt_obj_true.detach(), "cost_energy": cost_energy.detach(),
                "cost_reserve": cost_reserve.detach(),
            }
        else:
            aux = None

        if return_aux and not return_filter_aux: return total_realized_cost, aux
        if return_filter_aux and not return_aux: return total_realized_cost, aux_filter
        if return_aux and return_filter_aux: return total_realized_cost, aux, aux_filter
        return total_realized_cost

class DFL_model_diffusion_DRO_SingleNode(nn.Module):
    def __init__(
        self,
        mgr,
        optnet_DA,      
        optnet_RT,
        predictor,
        scenario_filter=None,   # <- added
        n_scen=50,
        clamp_min=0.0,
        solver="ECOS",
    ):
        super().__init__()
        self.mgr = mgr
        self.optnet_DA = optnet_DA
        self.optnet_RT = optnet_RT
        self.predictor = predictor
        self.scenario_filter = scenario_filter   # <- added

        self.n_scen = int(n_scen)
        self.clamp_min = clamp_min
        self.solver = solver

        self.optnet_dtype = torch.float64

        b_vals = [mgr.gen_info[g]["b"] for g in mgr.gen_bus_list]
        self.register_buffer("b_G", torch.tensor(b_vals, dtype=self.optnet_dtype))

        self.res_up_ratio = float(getattr(mgr.args, "reserve_up_ratio", 0.05))
        self.res_dn_ratio = float(getattr(mgr.args, "reserve_dn_ratio", 0.02))

    def _to_BT(self, x, B, device, dtype):
        x = torch.as_tensor(x, device=device, dtype=dtype)
        if x.ndim == 1:
            x = x.unsqueeze(0).expand(B, -1)
        return x

    def _realized_total_cost(self, P_DA_GT, R_up_GT, R_dn_GT, omega_true_T, solver):
        bG = self.b_G.to(device=P_DA_GT.device, dtype=self.optnet_dtype)
        stage1_energy_linear = (P_DA_GT * bG[:, None]).sum()
        stage1_reserve = (
            (R_up_GT * (self.res_up_ratio * bG)[:, None]).sum()
            + (R_dn_GT * (self.res_dn_ratio * bG)[:, None]).sum()
        )
        sol_rt = self.optnet_RT(
            R_up_GT.unsqueeze(0),
            R_dn_GT.unsqueeze(0),
            omega_true_T.unsqueeze(0),
            solver=solver,
        )
        rt_obj_true = sol_rt["rt_obj"][0]
        return stage1_energy_linear + stage1_reserve + rt_obj_true, sol_rt

    def forward(
        self,
        X_all,
        y_true,                 
        hourly_load_min_sys,    
        hourly_load_max_sys,    
        eps,                    
        solver=None,
        return_aux=False,
        return_filter_aux=False, # <- added
        predictor_n_samples=None,
        predictor_kwargs=None,
        filter_kwargs=None,      # <- added
    ):
        solver = solver or self.solver
        device = X_all.device
        dtype = self.optnet_dtype
        predictor_kwargs = dict(predictor_kwargs or {})
        filter_kwargs = dict(filter_kwargs or {})

        B = X_all.shape[0]
        S = int(predictor_n_samples) if predictor_n_samples is not None else self.n_scen

        Y_scen = self.predictor(
            X_all,
            n_samples=S,
            return_real_scale=True,
            grad_enabled=torch.is_grad_enabled(),
            **predictor_kwargs
        ).to(device=device, dtype=dtype)

        if self.clamp_min is not None:
            Y_scen = torch.clamp(Y_scen, min=float(self.clamp_min))

        # Hook Filter
        aux_filter = None
        if self.scenario_filter is not None:
            Y_scen, aux_filter = self.scenario_filter(
                Y_scen, is_train=self.training, **filter_kwargs
            )
            Y_scen = Y_scen.to(device=device, dtype=dtype)

        scen_sys = Y_scen.sum(dim=2)           # [S,B,T]
        forecast_sys = scen_sys.mean(dim=0)    # [B,T]
        omega_scen = (scen_sys - forecast_sys.unsqueeze(0)).permute(1, 0, 2).contiguous()  # [B,S,T]

        Lmin_sys = self._to_BT(hourly_load_min_sys, B=B, device=device, dtype=dtype)
        Lmax_sys = self._to_BT(hourly_load_max_sys, B=B, device=device, dtype=dtype)

        om_min_h = Lmin_sys - forecast_sys
        om_max_h = Lmax_sys - forecast_sys
        om_min_h, om_max_h = torch.minimum(om_min_h, om_max_h), torch.maximum(om_min_h, om_max_h)

        om_min_s = omega_scen.min(dim=1).values
        om_max_s = omega_scen.max(dim=1).values

        om_min = torch.minimum(om_min_h, om_min_s)  # [B,T]
        om_max = torch.maximum(om_max_h, om_max_s)  # [B,T]

        eps_t = torch.as_tensor(eps, device=device, dtype=dtype)
        if eps_t.ndim == 0:
            eps_t = eps_t.view(1).expand(B)

        T_len = forecast_sys.shape[-1]
        if y_true.dim() == 3 and y_true.shape[1] == T_len and y_true.shape[2] == 11:
            y_true = y_true.permute(0, 2, 1).contiguous()

        y_true_real11 = self.predictor.inverse_y(y_true.to(device=device)).to(device=device, dtype=dtype)
        omega_true = y_true_real11.sum(dim=1) - forecast_sys  # [B,T]

        loss_list = []
        aux = None
        if return_aux:
            aux = {"realized_total": []}

        for b in range(B):
            sol_da = self.optnet_DA(
                forecast_sys[b].unsqueeze(0),
                omega_scen[b].unsqueeze(0),  
                om_min[b].unsqueeze(0),      
                om_max[b].unsqueeze(0),      
                eps_t[b].view(1),            
                solver=solver,
            )

            P_DA = sol_da["P_DA"][0].to(dtype=dtype)
            R_up = sol_da["R_up"][0].to(dtype=dtype)
            R_dn = sol_da["R_dn"][0].to(dtype=dtype)

            realized_total, sol_rt = self._realized_total_cost(P_DA, R_up, R_dn, omega_true[b], solver=solver)
            loss_list.append(realized_total)

            if return_aux:
                aux["realized_total"].append(realized_total.detach())

        loss_vec = torch.stack(loss_list, dim=0)  # [B]
        
        if return_aux:
            for k in aux:
                aux[k] = torch.stack(aux[k], dim=0)

        if return_aux and not return_filter_aux: return loss_vec, aux
        if return_filter_aux and not return_aux: return loss_vec, aux_filter
        if return_aux and return_filter_aux: return loss_vec, aux, aux_filter
        return loss_vec



def _build_predictor_kwargs(args, *, training: bool):
    return dict(
        sample_chunk=getattr(args, "sample_chunk", 50),
        cpu_offload=False if training else bool(getattr(args, "cpu_offload", False)),
        trunc_steps=getattr(args, "trunc_steps", 1 if training else 0),
        mode=getattr(args, "diffusion_mode", "ddpm"),
        n_steps=getattr(args, "diffusion_steps", 50),
        eta=float(getattr(args, "ddim_eta", 0.0)),
        shared_noise=bool(getattr(args, "shared_noise", True)),
        z_temp=float(getattr(args, "z_temp", 1.0)),
    )



def DFL_train(
    dfl,
    train_dataset,
    args,
    problem_mode="saa",
    train_mode="dfl",
    filter_kwargs=None,
    lambda_div=1e5,
):
    print(f"Diffusion new train function: {train_mode}")

    # ---- 与参考代码一致：统一 train batch size ----
    train_bs = int(
        getattr(
            args,
            "train_batch_size",
            getattr(args, "dfl_batch_size", getattr(args, "batch_size", 8)),
        )
    )
    args.train_batch_size = train_bs

    # ---- device / mode ----
    device = torch.device(
        args.device if (getattr(args, "device", "cpu") == "cpu" or torch.cuda.is_available()) else "cpu"
    )
    dfl = dfl.to(device).train()
    if hasattr(dfl, "predictor") and hasattr(dfl.predictor, "train"):
        dfl.predictor.train()

    # ---- ScenarioFilter eval config sync ----
    eval_mode = str(getattr(args, "eval_mode", "soft")).lower()
    avoid_rand_duplicate = bool(getattr(args, "avoid_rand_duplicate", False))
    if getattr(dfl, "scenario_filter", None) is not None:
        if hasattr(dfl.scenario_filter, "eval_mode"):
            dfl.scenario_filter.eval_mode = eval_mode
        if hasattr(dfl.scenario_filter, "avoid_rand_duplicate"):
            dfl.scenario_filter.avoid_rand_duplicate = avoid_rand_duplicate

    # ---- filter kwargs ----
    if filter_kwargs is None:
        filter_kwargs = {
            "tau_gumbel": float(getattr(args, "tau_gumbel", 1.0)),
            "eps_uniform": float(getattr(args, "eps_uniform", 0.1)),
        }
    else:
        filter_kwargs = dict(filter_kwargs)
        filter_kwargs.setdefault("tau_gumbel", float(getattr(args, "tau_gumbel", 1.0)))
        filter_kwargs.setdefault("eps_uniform", float(getattr(args, "eps_uniform", 0.1)))

    # ---- diversity regularizer config ----
    div_type = str(getattr(args, "div_type", "inner")).lower()  # inner|kl|entropy
    div_eps = float(getattr(args, "div_eps", 1e-8))

    scenario_filter = getattr(dfl, "scenario_filter", None)
    has_learnable_filter = (
        scenario_filter is not None
        and any(p.requires_grad for p in scenario_filter.parameters())
    )

    # ---- optimizer setup ----
    optim_params_filter, optim_params_predictor = [], []

    if train_mode == "filter_only" and scenario_filter is not None:
        for p in dfl.parameters():
            p.requires_grad = False
        for p in dfl.scenario_filter.parameters():
            p.requires_grad = True

        optim_params_filter = [p for p in dfl.scenario_filter.parameters() if p.requires_grad]
        optim_params_predictor = []

        if len(optim_params_filter) == 0:
            raise ValueError("train_mode='filter_only' but scenario_filter has no trainable parameters.")

        filter_lr = float(getattr(args, "lr", getattr(args, "filter_lr", 1e-3)))
        optim = torch.optim.Adam(
            [{"params": optim_params_filter, "lr": filter_lr, "name": "filter"}],
        )
    else:
        for p in dfl.parameters():
            p.requires_grad = True

        optim_params_predictor = [p for p in dfl.predictor.parameters() if p.requires_grad]

        if scenario_filter is not None and any(p.requires_grad for p in scenario_filter.parameters()):
            optim_params_filter = [p for p in dfl.scenario_filter.parameters() if p.requires_grad]
        else:
            optim_params_filter = []

        filter_lr = float(getattr(args, "filter_lr", 1e-3))
        predictor_lr = float(getattr(args, "dfl_lr", getattr(args, "lr", 1e-5)))

        param_groups = []
        if len(optim_params_filter) > 0:
            param_groups.append({"params": optim_params_filter, "lr": filter_lr, "name": "filter"})
        if len(optim_params_predictor) > 0:
            param_groups.append({"params": optim_params_predictor, "lr": predictor_lr, "name": "predictor"})
        if len(param_groups) == 0:
            raise ValueError("No trainable parameters found in DFL_train.")

        optim = torch.optim.Adam(param_groups)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_bs,
        shuffle=True,
        drop_last=False,
    )

    pred_kwargs = _build_predictor_kwargs(args, training=True)
    epochs = int(getattr(args, "epochs", 1))

    # keep your original per-step lr decay behavior
    lr_decay = float(getattr(args, "lr_decay", 1.0))
    filter_lr_decay = float(getattr(args, "filter_lr_decay", lr_decay))
    dfl_lr_decay = float(getattr(args, "dfl_lr_decay", lr_decay))

    print("train_mode =", train_mode)
    print("scenario_filter type =", type(scenario_filter).__name__ if scenario_filter is not None else None)
    print("has_learnable_filter =", has_learnable_filter)
    print("train_batch_size =", train_bs)
    print("lr_decay =", lr_decay, "filter_lr_decay =", filter_lr_decay, "dfl_lr_decay =", dfl_lr_decay)

    train_losses = []

    for ep in tqdm(range(epochs), desc=f"Train ({train_mode})", leave=True):
        pbar = tqdm(train_loader, desc=f"Ep {ep+1}/{epochs}", leave=False)
        epoch_task_loss, epoch_div_loss, samples_cnt = 0.0, 0.0, 0

        for X_all, y_true in pbar:
            X_all = X_all.to(device).float()
            y_true = y_true.to(device).float()
            optim.zero_grad(set_to_none=True)

            kwargs = dict(
                solver=getattr(args, "solver", None),
                return_aux=False,
                return_filter_aux=True,
                predictor_n_samples=getattr(
                    args,
                    "predictor_n_samples",
                    getattr(args, "S_full", getattr(args, "N_scen", 50)),
                ),
                predictor_kwargs=pred_kwargs,
                filter_kwargs=dict(filter_kwargs),
            )

            if str(problem_mode).lower() == "dro":
                dtype = getattr(dfl, "optnet_dtype", torch.float64)
                kwargs.update(
                    hourly_load_min_sys=torch.as_tensor(args.Lmin, device=device, dtype=dtype),
                    hourly_load_max_sys=torch.as_tensor(args.Lmax, device=device, dtype=dtype),
                    eps=torch.as_tensor(args.eps_value, device=device, dtype=dtype),
                )

            out = dfl(X_all, y_true, **kwargs)
            if isinstance(out, (tuple, list)):
                loss_vec = out[0]
                aux_filter = out[-1]
            else:
                loss_vec = out
                aux_filter = None

            task_loss_val = loss_vec.mean()

            # -------- diversity regularizer --------
            div_loss_val = torch.tensor(0.0, device=device)
            if aux_filter is not None and ("p" in aux_filter) and (lambda_div > 0):
                p = aux_filter["p"].clamp_min(div_eps)
                p = p / p.sum(dim=-1, keepdim=True).clamp_min(div_eps)

                B_curr, K_curr, _ = p.shape

                if div_type == "inner":
                    if K_curr > 1:
                        inner_product = torch.bmm(p, p.transpose(1, 2))  # [B,K,K]
                        eye = (
                            torch.eye(K_curr, device=device, dtype=torch.bool)
                            .unsqueeze(0)
                            .expand(B_curr, -1, -1)
                        )
                        off_diag = inner_product[~eye]
                        div_loss_val = (off_diag ** 2).mean()

                elif div_type == "kl":
                    if K_curr > 1:
                        p_i = p.unsqueeze(2)  # [B,K,1,S]
                        p_j = p.unsqueeze(1)  # [B,1,K,S]
                        kl_ij = (p_i * (p_i.log() - p_j.log())).sum(dim=-1)
                        kl_ji = (p_j * (p_j.log() - p_i.log())).sum(dim=-1)
                        skl = 0.5 * (kl_ij + kl_ji)
                        eye = (
                            torch.eye(K_curr, device=device, dtype=torch.bool)
                            .unsqueeze(0)
                            .expand(B_curr, -1, -1)
                        )
                        div_loss_val = -skl[~eye].mean()

                elif div_type == "entropy":
                    H = -(p * p.log()).sum(dim=-1).mean()
                    div_loss_val = -H
                else:
                    raise ValueError(f"Unknown div_type={div_type}, choose from inner|kl|entropy")
            # -------------------------------------

            loss = task_loss_val + float(lambda_div) * div_loss_val
            loss.backward()

            if len(optim_params_filter) > 0:
                torch.nn.utils.clip_grad_norm_(optim_params_filter, 1.0)
            if len(optim_params_predictor) > 0:
                torch.nn.utils.clip_grad_norm_(optim_params_predictor, 1.0)

            optim.step()

            # per-step lr decay (as in your original code)
            for pg in optim.param_groups:
                if pg.get("name") == "filter":
                    pg["lr"] *= filter_lr_decay
                elif pg.get("name") == "predictor":
                    pg["lr"] *= dfl_lr_decay

            B_size = X_all.shape[0]
            epoch_task_loss += float(task_loss_val.detach().cpu()) * B_size
            epoch_div_loss += float(div_loss_val.detach().cpu()) * B_size
            samples_cnt += B_size

            pbar.set_postfix(
                Task=float(task_loss_val.detach().cpu()),
                Div=float(div_loss_val.detach().cpu()),
                Type=div_type,
                BS=train_bs,
                LR=[pg["lr"] for pg in optim.param_groups],
            )

        train_losses.append(
            {
                "epoch": int(ep + 1),
                "task": epoch_task_loss / max(samples_cnt, 1),
                "div": epoch_div_loss / max(samples_cnt, 1),
                "div_type": div_type,
                "train_batch_size": train_bs,
                "lr": {pg.get("name", f"group_{i}"): pg["lr"] for i, pg in enumerate(optim.param_groups)},
            }
        )

    return dfl, train_losses



@torch.no_grad()
def DFL_test(
    dfl,
    test_dataset,
    args,
    problem_mode="saa",
    filter_kwargs=None,
):
    test_bs = int(
        getattr(
            args,
            "test_batch_size",
            getattr(args, "batch_size", 8),
        )
    )
    args.test_batch_size = test_bs

    device = torch.device(
        args.device if (getattr(args, "device", "cpu") == "cpu" or torch.cuda.is_available()) else "cpu"
    )
    dfl = dfl.to(device).eval()

    eval_mode = str(getattr(args, "eval_mode", "soft")).lower()
    avoid_rand_duplicate = bool(getattr(args, "avoid_rand_duplicate", False))
    if getattr(dfl, "scenario_filter", None) is not None:
        if hasattr(dfl.scenario_filter, "eval_mode"):
            dfl.scenario_filter.eval_mode = eval_mode
        if hasattr(dfl.scenario_filter, "avoid_rand_duplicate"):
            dfl.scenario_filter.avoid_rand_duplicate = avoid_rand_duplicate

    print(f"[DFL_test] eval_mode={eval_mode}, avoid_rand_duplicate={avoid_rand_duplicate}, test_batch_size={test_bs}")

    if filter_kwargs is None:
        filter_kwargs = {
            "tau_gumbel": float(getattr(args, "tau_gumbel", 1.0)),
            "eps_uniform": float(getattr(args, "eps_uniform", 0.0)),
        }
    else:
        filter_kwargs = dict(filter_kwargs)
        filter_kwargs.setdefault("tau_gumbel", float(getattr(args, "tau_gumbel", 1.0)))
        filter_kwargs.setdefault("eps_uniform", float(getattr(args, "eps_uniform", 0.0)))

    test_loader = DataLoader(
        test_dataset,
        batch_size=test_bs,
        shuffle=False,
        drop_last=False,
    )

    pred_kwargs = _build_predictor_kwargs(args, training=False)
    losses = []

    pbar = tqdm(test_loader, desc="Test", leave=True)
    for X_all, y_true in pbar:
        set_seed(0)
        X_all, y_true = X_all.to(device).float(), y_true.to(device).float()

        kwargs = dict(
            solver=getattr(args, "solver", None),
            return_aux=False,
            return_filter_aux=False,
            predictor_n_samples=int(
                getattr(
                    args,
                    "predictor_n_samples",
                    getattr(args, "S_full", getattr(args, "N_scen", 50)),
                )
            ),
            predictor_kwargs=pred_kwargs,
            filter_kwargs=dict(filter_kwargs),
        )

        if str(problem_mode).lower() == "dro":
            dtype = getattr(dfl, "optnet_dtype", torch.float64)
            kwargs.update(
                hourly_load_min_sys=torch.as_tensor(args.Lmin, device=device, dtype=dtype),
                hourly_load_max_sys=torch.as_tensor(args.Lmax, device=device, dtype=dtype),
                eps=torch.as_tensor(args.eps_value, device=device, dtype=dtype),
            )

        out = dfl(X_all, y_true, **kwargs)
        loss_vec = out[0] if isinstance(out, (tuple, list)) else out

        batch_mean = float(loss_vec.mean().detach().cpu())
        losses.append(loss_vec.detach().cpu())
        pbar.set_postfix(loss=batch_mean)

    return torch.cat(losses, dim=0)

def run_DFL_diffusion_joint(
    args,
    problem_mode,
    optimization_mode,
    data_path,
    target_nodes,
    pack_data_m,
    models_m,
    device,
    seed=0,
    eval_splits=("test",),
    eval_flags=(True, True, True, True, True),
    stage2_artifact=None,
):
    import copy
    import time
    import torch

    def is_so(x):
        return str(x).lower() in {"so", "saa"}

    def is_multi(x):
        return str(x).lower() in {"multi", "multinode"}

    # ===== NEW: resolve best z_temp once and write back to args =====
    resolved_z_temp = _resolve_diffusion_z_temp(
        z_temp=None,
        pack_data_m=pack_data_m,
        args=args,
        default=1.0,
    )
    args.z_temp = float(resolved_z_temp)
    print(f"\n[run_DFL_diffusion_joint] using z_temp = {args.z_temp:.4f}")

    eval_splits = tuple(s.lower() for s in (eval_splits or ("test",)))
    if not all(s in ("train", "test") for s in eval_splits):
        raise ValueError(f"eval_splits must be subset of ('train','test'), got {eval_splits}")

    if eval_flags is None or len(eval_flags) != 5:
        raise ValueError(
            "eval_flags must be a 5-tuple: "
            "(det_before, random_before, stage2_after, stage3_before, stage3_after)"
        )
    eval_det_before, eval_random_before, eval_stage2_after, eval_stage3_before, eval_stage3_after = map(bool, eval_flags)

    args.eval_mode = str(getattr(args, "eval_mode", "soft")).lower()
    args.avoid_rand_duplicate = bool(getattr(args, "avoid_rand_duplicate", False))

    train_bs = int(
        getattr(
            args,
            "train_batch_size",
            getattr(args, "dfl_batch_size", getattr(args, "batch_size", 8)),
        )
    )
    test_bs = int(getattr(args, "test_batch_size", getattr(args, "batch_size", 8)))

    args.train_batch_size = train_bs
    args.test_batch_size = test_bs

    print("\n[ScenarioFilter eval config]")
    print("eval_mode =", args.eval_mode)
    print("avoid_rand_duplicate =", args.avoid_rand_duplicate)
    print("train_batch_size =", args.train_batch_size)
    print("test_batch_size =", args.test_batch_size)

    reused_stage2 = stage2_artifact is not None
    multi = is_multi(optimization_mode)

    if multi:
        SO_Manager = IEEE14_Reserve_SO_Manager_MultiNode
        DRO_Manager = IEEE14_Reserve_DRO_Manager_MultiNode
        DET_Manager = IEEE14_Reserve_Deterministic_DA_Manager_MultiNode

        SAA_DA_OptNet = MultiNode_Reserve_SAA_DA_OptNet
        DRO_DA_OptNet = MultiNode_Reserve_DRO_DA_OptNet
        DET_DA_OptNet = MultiNode_Reserve_Deterministic_DA_OptNet

        RT_OptNet = MultiNode_Reserve_RT_OptNet

        DFL_SO_Class = DFL_model_diffusion_MultiNode
        DFL_DRO_Class = DFL_model_diffusion_DRO_MultiNode
        DET_Class = DFL_model_diffusion_Deterministic_MultiNode
    else:
        SO_Manager = IEEE14_Reserve_SO_Manager_SingleNode
        DRO_Manager = IEEE14_Reserve_DRO_Manager_SingleNode
        DET_Manager = IEEE14_Reserve_Deterministic_DA_Manager_SingleNode

        SAA_DA_OptNet = SingleNode_Reserve_SAA_DA_OptNet
        DRO_DA_OptNet = SingleNode_Reserve_DRO_DA_OptNet
        DET_DA_OptNet = SingleNode_Reserve_Deterministic_DA_OptNet

        RT_OptNet = SingleNode_Reserve_RT_OptNet

        DFL_SO_Class = DFL_model_diffusion_SingleNode
        DFL_DRO_Class = DFL_model_diffusion_DRO_SingleNode
        DET_Class = DFL_model_diffusion_Deterministic_SingleNode

    args.S_full = int(getattr(args, "S_full", 50))
    K = int(getattr(args, "N_scen", 50))
    K_rand = int(getattr(args, "K_rand", 0))
    if K_rand > K:
        raise ValueError(f"K_rand({K_rand}) must be <= N_scen({K}).")

    def print_mean_loss(title, eval_dict, key):
        if key in eval_dict and eval_dict[key] is not None:
            val = eval_dict[key]
            if hasattr(val, "detach"):
                mean_val = val.detach().float().mean().item()
            else:
                mean_val = float(torch.as_tensor(val).float().mean().item())
            print(f"\n[TEST] {title}:")
            print(mean_val)

    det_before_eval = {}
    stage1_eval = {}
    stage2_eval = {}
    stage3_before_eval = {}
    stage3_eval = {}

    train_logs_stage2, train_logs_stage3 = [], []
    time_stage2, time_stage3 = 0.0, 0.0

    set_seed(seed)
    t0_total = time.time()

    run_stage2 = bool(getattr(args, "run_stage2", True))
    run_stage3 = bool(getattr(args, "run_stage3", True))

    if not reused_stage2:

        def make_dataset(flag):
            return Dataset_load_multi_node_diff(
                data_path=data_path,
                node_cols=target_nodes,
                flag=flag,
                train_length=8760,
                val_ratio=0.2,
                seed=42,
            )

        train_data = make_dataset("train")
        test_data = make_dataset("test")

        def build_predictor(joint_model, dataset):
            return Joint_diffusion_predictor(
                joint_model=copy.deepcopy(joint_model),
                dataset=dataset,
                device=device,
                default_z_temp=args.z_temp,
            ).to(device)

        def make_filter():
            return ScenarioFilter(
                args=args,
                prob_type="multi" if multi else "single",
                T=24,
                N_nodes=11,
                K=K,
                K_rand=K_rand,
                hidden=128,
            ).to(device)

        predictor_before = build_predictor(models_m, train_data)
        predictor_train = build_predictor(models_m, train_data)

        if is_so(problem_mode):
            mgr_local = SO_Manager(args)
            optnet_DA = SAA_DA_OptNet(mgr=mgr_local, N_scen=args.N_scen, T=24).to(device)
            DFLClass = DFL_SO_Class
            mode_str = "SO"
        else:
            mgr_local = DRO_Manager(args)
            optnet_DA = DRO_DA_OptNet(mgr=mgr_local, N_scen=args.N_scen, T=24).to(device)
            DFLClass = DFL_DRO_Class
            mode_str = "DRO"

        mgr_det = DET_Manager(args)
        optnet_DA_det = DET_DA_OptNet(mgr=mgr_det, T=24).to(device)

        optnet_RT = RT_OptNet(mgr=mgr_local, T=24).to(device)

        filter_module = make_filter()

        dfl_det_before = DET_Class(
            mgr=mgr_det,
            optnet_DA=optnet_DA_det,
            optnet_RT=optnet_RT,
            predictor=copy.deepcopy(predictor_before),
            scenario_filter=copy.deepcopy(filter_module),
            n_scen=args.N_scen,
            solver=getattr(args, "solver", "ECOS"),
        ).to(device)

        dfl_before = DFLClass(
            mgr=mgr_local,
            optnet_DA=optnet_DA,
            optnet_RT=optnet_RT,
            predictor=predictor_before,
            scenario_filter=copy.deepcopy(filter_module),
            n_scen=args.N_scen,
            solver=getattr(args, "solver", "ECOS"),
        ).to(device)

        dfl = DFLClass(
            mgr=mgr_local,
            optnet_DA=optnet_DA,
            optnet_RT=optnet_RT,
            predictor=predictor_train,
            scenario_filter=filter_module,
            n_scen=args.N_scen,
            solver=getattr(args, "solver", "ECOS"),
        ).to(device)

        def eval_on_splits(model, stage_tag: str):
            out = {}
            if "test" in eval_splits:
                set_seed(seed)
                out[f"test_losses_{stage_tag}"] = DFL_test(model, test_data, args, problem_mode=problem_mode)
            if "train" in eval_splits:
                set_seed(seed)
                out[f"train_losses_{stage_tag}"] = DFL_test(model, train_data, args, problem_mode=problem_mode)
            return out

        def rebuild_model_like(src_model, kind: str):
            if src_model is None:
                return None

            new_predictor = build_predictor(models_m, train_data)
            if hasattr(src_model, "predictor"):
                new_predictor.load_state_dict(copy.deepcopy(src_model.predictor.state_dict()))

            new_filter = make_filter()
            if hasattr(src_model, "scenario_filter") and hasattr(new_filter, "load_state_dict"):
                try:
                    new_filter.load_state_dict(copy.deepcopy(src_model.scenario_filter.state_dict()))
                except Exception:
                    pass

            if kind == "dfl":
                return DFLClass(
                    mgr=mgr_local,
                    optnet_DA=optnet_DA,
                    optnet_RT=optnet_RT,
                    predictor=new_predictor,
                    scenario_filter=new_filter,
                    n_scen=args.N_scen,
                    solver=getattr(args, "solver", "ECOS"),
                ).to(device)
            elif kind == "det":
                return DET_Class(
                    mgr=mgr_det,
                    optnet_DA=optnet_DA_det,
                    optnet_RT=optnet_RT,
                    predictor=new_predictor,
                    scenario_filter=new_filter,
                    n_scen=args.N_scen,
                    solver=getattr(args, "solver", "ECOS"),
                ).to(device)
            else:
                raise ValueError(f"unknown kind={kind}")

        if eval_det_before:
            det_before_eval = eval_on_splits(dfl_det_before, "deterministic_before")
            if "test" in eval_splits:
                print_mean_loss(
                    "Deterministic baseline (before DFL training)",
                    det_before_eval,
                    "test_losses_deterministic_before",
                )

        if eval_random_before:
            random_filter = RandomScenarioSelector(n_scen=int(args.N_scen)).to(device)
            orig_filter_before = dfl_before.scenario_filter
            dfl_before.scenario_filter = random_filter
            stage1_eval = eval_on_splits(dfl_before, "stage1_after")
            dfl_before.scenario_filter = orig_filter_before

            if "test" in eval_splits:
                print_mean_loss(
                    "Random filter baseline (before DFL training)",
                    stage1_eval,
                    "test_losses_stage1_after",
                )

        args.epochs = int(getattr(args, "dfl_epochs", 2))
        args.dfl_lr = float(getattr(args, "dfl_lr", 1e-7))
        args.filter_lr = float(getattr(args, "filter_lr", 1e-3))

        if run_stage2:
            print(
                f"\n ---> Stage A: train Diffusion DFL with RANDOM selector "
                f"(epochs={args.epochs}, dfl_lr={args.dfl_lr}, train_bs={args.train_batch_size}, z_temp={args.z_temp})"
            )

            dfl.scenario_filter = RandomScenarioSelector(n_scen=int(args.N_scen)).to(device)
            for p in dfl.predictor.parameters():
                p.requires_grad_(True)

            t0_s2 = time.time()
            dfl, train_logs_stage2 = DFL_train(
                dfl,
                train_data,
                args,
                problem_mode=problem_mode,
                train_mode="dfl",
                lambda_div=float(getattr(args, "lambda_div", 0)),
            )
            time_stage2 = time.time() - t0_s2
            print(f" ---> [Stage A done] time: {time_stage2:.2f} sec")

            if eval_stage2_after:
                stage2_eval = eval_on_splits(dfl, "stage2_after")
                if "test" in eval_splits:
                    print_mean_loss(
                        "Random filter cost AFTER DFL training",
                        stage2_eval,
                        "test_losses_stage2_after",
                    )
        else:
            print("\n ---> [skip] Stage A disabled (run_stage2=False)")

        dfl_after_stage2_snapshot = rebuild_model_like(dfl, kind="dfl")
        dfl_det_before_snapshot = rebuild_model_like(dfl_det_before, kind="det") if dfl_det_before is not None else None
        dfl_before_snapshot = rebuild_model_like(dfl_before, kind="dfl") if dfl_before is not None else None

    else:
        print("\n ---> Reusing passed stage2_artifact, skip Stage A and jump to Stage B")

        train_data = stage2_artifact["train_data"]
        test_data = stage2_artifact["test_data"]
        mode_str = stage2_artifact["mode_str"]

        det_before_eval = copy.deepcopy(stage2_artifact.get("det_before_eval", {}))
        stage1_eval = copy.deepcopy(stage2_artifact.get("stage1_eval", {}))
        stage2_eval = copy.deepcopy(stage2_artifact.get("stage2_eval", {}))

        train_logs_stage2 = copy.deepcopy(stage2_artifact.get("train_logs_stage2", []))
        time_stage2 = float(stage2_artifact.get("time_stage2_sec", 0.0))

        if is_so(problem_mode):
            mgr_local = SO_Manager(args)
            optnet_DA = SAA_DA_OptNet(mgr=mgr_local, N_scen=args.N_scen, T=24).to(device)
            DFLClass = DFL_SO_Class
        else:
            mgr_local = DRO_Manager(args)
            optnet_DA = DRO_DA_OptNet(mgr=mgr_local, N_scen=args.N_scen, T=24).to(device)
            DFLClass = DFL_DRO_Class

        optnet_RT = RT_OptNet(mgr=mgr_local, T=24).to(device)

        def build_predictor(joint_model, dataset):
            return Joint_diffusion_predictor(
                joint_model=copy.deepcopy(joint_model),
                dataset=dataset,
                device=device,
                default_z_temp=args.z_temp,
            ).to(device)

        def make_filter():
            return ScenarioFilter(
                args=args,
                prob_type="multi" if multi else "single",
                T=24,
                N_nodes=11,
                K=K,
                K_rand=K_rand,
                hidden=128,
            ).to(device)

        def rebuild_model_like(src_model, kind: str):
            if src_model is None:
                return None
            if kind != "dfl":
                raise ValueError("reuse branch rebuild_model_like only supports kind='dfl' for Stage B training.")

            new_predictor = build_predictor(models_m, train_data)
            if hasattr(src_model, "predictor"):
                new_predictor.load_state_dict(copy.deepcopy(src_model.predictor.state_dict()))

            new_filter = make_filter()
            if hasattr(src_model, "scenario_filter") and hasattr(new_filter, "load_state_dict"):
                try:
                    new_filter.load_state_dict(copy.deepcopy(src_model.scenario_filter.state_dict()))
                except Exception:
                    pass

            return DFLClass(
                mgr=mgr_local,
                optnet_DA=optnet_DA,
                optnet_RT=optnet_RT,
                predictor=new_predictor,
                scenario_filter=new_filter,
                n_scen=args.N_scen,
                solver=getattr(args, "solver", "ECOS"),
            ).to(device)

        stage2_model = stage2_artifact["dfl_after_stage2"]
        dfl = rebuild_model_like(stage2_model, kind="dfl")

        dfl_after_stage2_snapshot = stage2_model
        dfl_det_before_snapshot = stage2_artifact.get("dfl_det_before", None)
        dfl_before_snapshot = stage2_artifact.get("dfl_before", None)

        dfl_det_before = stage2_artifact.get("dfl_det_before", None)
        dfl_before = stage2_artifact.get("dfl_before", None)

        def eval_on_splits(model, stage_tag: str):
            out = {}
            if "test" in eval_splits:
                set_seed(seed)
                out[f"test_losses_{stage_tag}"] = DFL_test(model, test_data, args, problem_mode=problem_mode)
            if "train" in eval_splits:
                set_seed(seed)
                out[f"train_losses_{stage_tag}"] = DFL_test(model, train_data, args, problem_mode=problem_mode)
            return out

    if run_stage3:
        args.epochs = int(getattr(args, "filter_epochs", 10))
        args.lr = float(getattr(args, "filter_lr", 1e-3))

        print(
            f"\n ---> Stage B: switch to learnable filter, train Filter only "
            f"(epochs={args.epochs}, lr={args.lr}, train_bs={args.train_batch_size}, z_temp={args.z_temp})"
        )

        dfl.scenario_filter = ScenarioFilter(
            args=args,
            prob_type="multi" if multi else "single",
            T=24,
            N_nodes=11,
            K=K,
            K_rand=K_rand,
            hidden=128,
        ).to(device)

        if eval_stage3_before:
            stage3_before_eval = eval_on_splits(dfl, "stage3_before")
            if "test" in eval_splits:
                print_mean_loss(
                    "Fresh ScenarioFilter BEFORE training",
                    stage3_before_eval,
                    "test_losses_stage3_before",
                )

        for p in dfl.predictor.parameters():
            p.requires_grad_(False)
        for p in dfl.scenario_filter.parameters():
            p.requires_grad_(True)

        t0_s3 = time.time()
        dfl, train_logs_stage3 = DFL_train(
            dfl,
            train_data,
            args,
            problem_mode=problem_mode,
            train_mode="filter_only",
            lambda_div=float(getattr(args, "lambda_div", 1e5)),
        )
        time_stage3 = time.time() - t0_s3
        print(f" ---> [Stage B done] time: {time_stage3:.2f} sec")
    else:
        print("\n ---> [skip] Stage B disabled (run_stage3=False)")

    dfl_trained = dfl
    train_time_sec_total = time.time() - t0_total
    print(f"\n === total train time: {train_time_sec_total:.2f} sec ===")

    if eval_stage3_after:
        stage3_eval = eval_on_splits(dfl_trained, "stage3_after")
        if "test" in eval_splits:
            print_mean_loss(
                "ScenarioFilter AFTER training",
                stage3_eval,
                "test_losses_stage3_after",
            )

    stage2_artifact_out = {
        "train_data": train_data,
        "test_data": test_data,
        "mode_str": mode_str,
        "dfl_after_stage2": dfl_after_stage2_snapshot,
        "dfl_det_before": dfl_det_before_snapshot,
        "dfl_before": dfl_before_snapshot,
        "det_before_eval": copy.deepcopy(det_before_eval),
        "stage1_eval": copy.deepcopy(stage1_eval),
        "stage2_eval": copy.deepcopy(stage2_eval),
        "train_logs_stage2": copy.deepcopy(train_logs_stage2),
        "time_stage2_sec": float(time_stage2),
        "z_temp": float(args.z_temp),
    }

    result = {
        "optimization_mode": "multi" if multi else "single",
        "problem_mode": mode_str,
        "eval_splits": eval_splits,
        "eval_flags": tuple(map(bool, eval_flags)),
        "run_stage2": bool(run_stage2),
        "run_stage3": bool(run_stage3),
        "reused_stage2": bool(reused_stage2),
        "dfl_det_before": locals().get("dfl_det_before", None),
        "dfl_before": locals().get("dfl_before", None),
        "dfl_trained": dfl_trained,
        "train_logs_stage2": train_logs_stage2,
        "train_logs_stage3": train_logs_stage3,
        "time_stage2_sec": float(time_stage2),
        "time_stage3_sec": float(time_stage3),
        "train_time_sec_total": float(train_time_sec_total),
        "train_batch_size_used": int(args.train_batch_size),
        "test_batch_size_used": int(args.test_batch_size),
        "N_scen": int(args.N_scen),
        "S_full": int(args.S_full),
        "K_rand": int(K_rand),
        "eval_mode": str(getattr(args, "eval_mode", "soft")).lower(),
        "avoid_rand_duplicate": bool(getattr(args, "avoid_rand_duplicate", False)),
        "z_temp": float(args.z_temp),
        "stage2_artifact": stage2_artifact_out,
    }

    result.update(det_before_eval)
    result.update(stage1_eval)
    result.update(stage2_eval)
    result.update(stage3_before_eval)
    result.update(stage3_eval)

    if "test_losses_deterministic_before" in result:
        result["test_losses_det_baseline"] = result["test_losses_deterministic_before"]
    if "train_losses_deterministic_before" in result:
        result["train_losses_det_baseline"] = result["train_losses_deterministic_before"]

    if "test_losses_stage1_after" in result:
        result["test_losses_random_baseline"] = result["test_losses_stage1_after"]
    if "train_losses_stage1_after" in result:
        result["train_losses_random_baseline"] = result["train_losses_stage1_after"]

    if "test_losses_stage2_after" in result:
        result["test_losses_random_filter_after_dfl"] = result["test_losses_stage2_after"]
    if "train_losses_stage2_after" in result:
        result["train_losses_random_filter_after_dfl"] = result["train_losses_stage2_after"]

    if "test_losses_stage3_before" in result:
        result["test_losses_scenario_filter_before_training"] = result["test_losses_stage3_before"]
    if "train_losses_stage3_before" in result:
        result["train_losses_scenario_filter_before_training"] = result["train_losses_stage3_before"]

    if "test_losses_stage3_after" in result:
        result["test_losses_scenario_filter_after_training"] = result["test_losses_stage3_after"]
    if "train_losses_stage3_after" in result:
        result["train_losses_scenario_filter_after_training"] = result["train_losses_stage3_after"]

    if "test_losses_stage2_after" in result:
        result["test_losses_before_filter_training"] = result["test_losses_stage2_after"]
    if "train_losses_stage2_after" in result:
        result["train_losses_before_filter_training"] = result["train_losses_stage2_after"]

    if "test_losses_stage3_after" in result:
        result["test_losses_after_filter_training"] = result["test_losses_stage3_after"]
    if "train_losses_stage3_after" in result:
        result["train_losses_after_filter_training"] = result["train_losses_stage3_after"]

    if "train" in eval_splits:
        if "train_losses_stage1_after" in result:
            result["train_losses_before"] = result["train_losses_stage1_after"]
        if "train_losses_stage3_after" in result:
            result["train_losses_after"] = result["train_losses_stage3_after"]

    if "test" in eval_splits:
        if "test_losses_stage1_after" in result:
            result["test_losses_before"] = result["test_losses_stage1_after"]
        if "test_losses_stage3_after" in result:
            result["test_losses_after"] = result["test_losses_stage3_after"]

    return result

def compare_scenario_filters_with_stage3_learned_diff(
    base_result,
    args,
    problem_mode,
    optimization_mode,
    models_m,
    target_nodes,
    device,
    eval_splits=("test",),
    method_names=("learned", "random", "kmeans", "kmedoids", "hierarchical"),
    seed=0,
    verbose=True,
):
    import copy
    import numpy as np
    import torch

    # ===== NEW: prefer z_temp from stage2 artifact / base_result =====
    if "stage2_artifact" in base_result and isinstance(base_result["stage2_artifact"], dict):
        if "z_temp" in base_result["stage2_artifact"]:
            args.z_temp = float(base_result["stage2_artifact"]["z_temp"])
    elif "z_temp" in base_result:
        args.z_temp = float(base_result["z_temp"])

    train_bs = int(
        getattr(
            args,
            "train_batch_size",
            getattr(args, "dfl_batch_size", getattr(args, "batch_size", 8)),
        )
    )
    test_bs = int(getattr(args, "test_batch_size", getattr(args, "batch_size", 8)))

    args.train_batch_size = train_bs
    args.test_batch_size = test_bs

    args.eval_mode = str(getattr(args, "eval_mode", "soft")).lower()
    args.avoid_rand_duplicate = bool(getattr(args, "avoid_rand_duplicate", False))

    if verbose:
        print(f"[compare_diff_joint] train_batch_size = {train_bs}")
        print(f"[compare_diff_joint] test_batch_size = {test_bs}")
        print(f"[compare_diff_joint] eval_mode = {args.eval_mode}")
        print(f"[compare_diff_joint] avoid_rand_duplicate = {args.avoid_rand_duplicate}")
        print(f"[compare_diff_joint] z_temp = {float(getattr(args, 'z_temp', 1.0))}")

    eval_splits = tuple(s.lower() for s in (eval_splits or ("test",)))
    if not all(s in ("train", "test") for s in eval_splits):
        raise ValueError(f"eval_splits must be subset of ('train','test'), got {eval_splits}")

    if "stage2_artifact" not in base_result:
        raise ValueError("base_result does not contain 'stage2_artifact'.")

    stage2_artifact = base_result["stage2_artifact"]
    stage2_model = stage2_artifact["dfl_after_stage2"]
    trained_model = base_result["dfl_trained"]

    learned_filter = copy.deepcopy(trained_model.scenario_filter)
    train_data = stage2_artifact["train_data"]
    test_data = stage2_artifact["test_data"]

    def _mean_of(x):
        if x is None:
            return None
        if hasattr(x, "detach"):
            return float(x.detach().float().mean().item())
        return float(torch.as_tensor(x).float().mean().item())

    def is_so(x):
        return str(x).lower() in {"so", "saa"}

    def is_multi(x):
        return str(x).lower() in {"multi", "multinode"}

    multi = is_multi(optimization_mode)

    if multi:
        SO_Manager = IEEE14_Reserve_SO_Manager_MultiNode
        DRO_Manager = IEEE14_Reserve_DRO_Manager_MultiNode
        SAA_DA_OptNet = MultiNode_Reserve_SAA_DA_OptNet
        DRO_DA_OptNet = MultiNode_Reserve_DRO_DA_OptNet
        RT_OptNet = MultiNode_Reserve_RT_OptNet
        DFL_SO_Class = DFL_model_diffusion_MultiNode
        DFL_DRO_Class = DFL_model_diffusion_DRO_MultiNode
    else:
        SO_Manager = IEEE14_Reserve_SO_Manager_SingleNode
        DRO_Manager = IEEE14_Reserve_DRO_Manager_SingleNode
        SAA_DA_OptNet = SingleNode_Reserve_SAA_DA_OptNet
        DRO_DA_OptNet = SingleNode_Reserve_DRO_DA_OptNet
        RT_OptNet = SingleNode_Reserve_RT_OptNet
        DFL_SO_Class = DFL_model_diffusion_SingleNode
        DFL_DRO_Class = DFL_model_diffusion_DRO_SingleNode

    compare_result = {
        "backbone_predictor_source": "stage2_artifact['dfl_after_stage2'].predictor",
        "learned_filter_source": "base_result['dfl_trained'].scenario_filter",
        "method_names": list(method_names),
        "eval_splits": list(eval_splits),
        "train_batch_size_used": int(train_bs),
        "test_batch_size_used": int(test_bs),
        "z_temp": float(getattr(args, "z_temp", 1.0)),
        "details": {},
        "summary_mean": {},
    }

    def build_predictor_fresh():
        pred = Joint_diffusion_predictor(
            joint_model=copy.deepcopy(models_m),
            dataset=train_data,
            device=device,
            default_z_temp=float(getattr(args, "z_temp", 1.0)),
        ).to(device)
        pred.load_state_dict(copy.deepcopy(stage2_model.predictor.state_dict()))
        return pred

    def rebuild_with_specific_filter(filter_module):
        predictor = build_predictor_fresh()

        if is_so(problem_mode):
            mgr_local = SO_Manager(args)
            optnet_DA = SAA_DA_OptNet(mgr=mgr_local, N_scen=args.N_scen, T=24).to(device)
            DFLClass = DFL_SO_Class
        else:
            mgr_local = DRO_Manager(args)
            optnet_DA = DRO_DA_OptNet(mgr=mgr_local, N_scen=args.N_scen, T=24).to(device)
            DFLClass = DFL_DRO_Class

        optnet_RT = RT_OptNet(mgr=mgr_local, T=24).to(device)

        model = DFLClass(
            mgr=mgr_local,
            optnet_DA=optnet_DA,
            optnet_RT=optnet_RT,
            predictor=predictor,
            scenario_filter=filter_module.to(device),
            n_scen=args.N_scen,
            solver=getattr(args, "solver", "ECOS"),
        ).to(device)
        return model

    def eval_on_splits(model, stage_tag):
        out = {}
        if "test" in eval_splits:
            set_seed(seed)
            out[f"test_losses_{stage_tag}"] = DFL_test(
                model,
                test_data,
                args,
                problem_mode=problem_mode,
            )
        if "train" in eval_splits:
            set_seed(seed)
            out[f"train_losses_{stage_tag}"] = DFL_test(
                model,
                train_data,
                args,
                problem_mode=problem_mode,
            )
        return out

    for name in method_names:
        if verbose:
            print(f"[compare_diff_joint] evaluating method: {name}")

        if name == "learned":
            filter_module = copy.deepcopy(learned_filter)
        elif name == "random":
            filter_module = RandomScenarioSelector(n_scen=int(args.N_scen)).to(device)
        else:
            filter_module = build_scenario_baseline_filter(name, args, device)

        if hasattr(filter_module, "eval_mode"):
            filter_module.eval_mode = args.eval_mode
        if hasattr(filter_module, "avoid_rand_duplicate"):
            filter_module.avoid_rand_duplicate = args.avoid_rand_duplicate

        model = rebuild_with_specific_filter(filter_module)
        stage_tag = f"compare_stage2backbone_{name}"
        eval_out = eval_on_splits(model, stage_tag)

        detail = {
            "method_name": name,
            "scenario_filter_class": (
                type(filter_module).__name__ if filter_module is not None else None
            ),
        }
        detail.update(eval_out)
        compare_result["details"][name] = detail

        mean_info = {}
        for split in eval_splits:
            key = f"{split}_losses_{stage_tag}"
            if key in eval_out:
                mean_info[split] = _mean_of(eval_out[key])

        compare_result["summary_mean"][name] = mean_info

        if verbose:
            print(f"[compare_diff_joint][{name}]")
            for split in eval_splits:
                if split in mean_info and mean_info[split] is not None:
                    print(f"  {split} mean loss: {mean_info[split]:.6f}")

    return compare_result