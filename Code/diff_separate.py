import os
import time
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from diffusion import *
from data_loader import *
from combined_data_loader import *
from utils import *
from tqdm import tqdm
from Optimization_multi_node import *
from Optimization_single_node import *
from scenarios_reduce import *

@torch.no_grad()
def pinball_loss_multiq(y_true, y_samp, quantiles=(0.1, 0.5, 0.9)):
    """
    y_true: [B,T] or [B,T,N]
    y_samp: [S,B,T] or [S,B,T,N]
    return: scalar avg pinball loss
    """
    qs = torch.tensor(list(quantiles), device=y_true.device, dtype=y_true.dtype)  # [Q]
    y_q = torch.quantile(y_samp, qs, dim=0)  # [Q,B,T] or [Q,B,T,N]

    diff = y_true.unsqueeze(0) - y_q

    if diff.ndim == 3:
        # [Q,B,T]
        q_view = qs.view(-1, 1, 1)
    elif diff.ndim == 4:
        # [Q,B,T,N]
        q_view = qs.view(-1, 1, 1, 1)
    else:
        raise ValueError(f"Unexpected diff ndim={diff.ndim}, shape={diff.shape}")

    loss = torch.maximum(q_view * diff, (q_view - 1.0) * diff)
    return loss.mean().item()


def _get_dfl_main_device(dfl):
    if hasattr(dfl, "predictor") and hasattr(dfl.predictor, "output_device"):
        return torch.device(dfl.predictor.output_device)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Runner_diffusion_single:
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

        feat_dim = int(train_set.X.shape[-1])
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = ConditionalDDPM_Single(
            feat_dim=feat_dim,
            T=T,
            y_dim=24,
            time_dim=time_dim,
            hidden_ch=hidden_ch,
            n_layers=n_layers,
            dropout=dropout,
            beta_schedule="cosine",
        ).to(self.device).float()

        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)

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

            for X, y in train_loader:
                X = X.to(self.device, dtype=torch.float32, non_blocking=True)  # [B,24,F]
                y = y.to(self.device, dtype=torch.float32, non_blocking=True)  # [B,24]

                loss = self.model.loss(X, y)

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(grad_clip))
                self.opt.step()

                tr.append(float(loss.item()))

            self.model.eval()
            va = []
            with torch.no_grad():
                for X, y in val_loader:
                    X = X.to(self.device, dtype=torch.float32, non_blocking=True)
                    y = y.to(self.device, dtype=torch.float32, non_blocking=True)
                    va.append(float(self.model.loss(X, y).item()))

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
        self.model = self.model.to(self.device, dtype=torch.float32)
        self.model.eval()

def run_diffusion_single(
    data_path,
    node_cols,
    device=None,
    epochs=2000,
    batch_size=128,
    lr=2e-4,
    patience=80,
    ckpt_dir="ckpt_nodes_diffusion_conv",
    verbose=True,
    train_length=4296,
    val_ratio=0.2,
    seed=42,
    # diffusion params
    T=200,
    time_dim=64,
    hidden_ch=128,
    n_layers=6,
    dropout=0.0,
    # NEW: z_temp selection
    find_best_ztemp=True,
    z_temp_grid=np.linspace(0.5, 3.0, 8),
    ztemp_n_samples=50,
    ztemp_batch_size=256,
    ztemp_quantiles=None,
    diffusion_mode="ddim",
    diffusion_steps=50,
    ddim_eta=0.0,
    ztemp_search_sample_chunk=50,
):
    os.makedirs(ckpt_dir, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    models_s, handlers_s, pack_data_s = {}, {}, {}

    for node in node_cols:
        train_set = Dataset_load_single_node_diff(
            data_path, node, flag="train", train_length=train_length, val_ratio=val_ratio, seed=seed
        )
        val_set = Dataset_load_single_node_diff(
            data_path, node, flag="val", train_length=train_length, val_ratio=val_ratio, seed=seed
        )
        test_set = Dataset_load_single_node_diff(
            data_path, node, flag="test", train_length=train_length, val_ratio=val_ratio, seed=seed
        )

        runner = Runner_diffusion_single(
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

        best_path = os.path.join(ckpt_dir, f"best_{node}.pt")
        runner.fit(
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            best_path=best_path,
            verbose=verbose,
        )

        model = runner.model.to(device=device, dtype=torch.float32).eval()

        if find_best_ztemp:
            best_t, best_score, table = select_best_ztemp_for_single_node_diffusion(
                model=model,
                val_set=val_set,
                device=torch.device(device) if isinstance(device, str) else device,
                ztemps=z_temp_grid,
                n_samples=ztemp_n_samples,
                batch_size=ztemp_batch_size,
                quantiles=ztemp_quantiles,
                mode=diffusion_mode,
                n_steps=diffusion_steps,
                eta=ddim_eta,
                sample_chunk=ztemp_search_sample_chunk,
                verbose=verbose,
            )
        else:
            best_t, best_score, table = None, None, None

        if verbose and find_best_ztemp:
            print(f"[{node}] best z_temp={best_t} | val pinball={best_score:.6f}")

        models_s[node] = model

        handlers_s[node] = {
            "train": train_set,
            "val": val_set,
            "test": test_set,
        }

        pack_data_s[node] = {
            "splits": {
                "train": {"X": train_set.X, "y": train_set.y},
                "val": {"X": val_set.X, "y": val_set.y},
                "test": {"X": test_set.X, "y": test_set.y},
            },
            "runner": runner,
            "best_path": best_path,
            "best_z_temp": (None if best_t is None else float(best_t)),
            "best_val_pinball": (None if best_score is None else float(best_score)),
            "ztemp_table": table,
            "ztemp_quantiles": (None if ztemp_quantiles is None else tuple(ztemp_quantiles)),
        }

    return models_s, handlers_s, pack_data_s


def ddpm_sample_parallel_chunked_single(
    model,
    cond,  # [B,24,F]
    n_samples=50,
    sample_chunk=10,
    shared_noise=True,
    grad_enabled=None,
    cpu_offload=False,
    trunc_steps=10,
    force_fp32=True,
    z_temp=1.0,
):
    if grad_enabled is None:
        grad_enabled = torch.is_grad_enabled()
    if grad_enabled and cpu_offload:
        raise ValueError("Training (grad_enabled=True) requires cpu_offload=False.")

    z_temp = float(z_temp)

    model_device = next(model.parameters()).device
    cond = cond.to(device=model_device, dtype=torch.float32, non_blocking=True)

    device = model_device
    B = cond.shape[0]
    S = int(n_samples)

    if force_fp32:
        model = model.float()
        if cond.dtype != torch.float32:
            cond = cond.float()

    N = int(getattr(model, "n_nodes", 1))
    is_single = (N == 1)

    K = int(trunc_steps)
    K = max(0, min(K, int(model.T)))

    outs = []
    for s0 in range(0, S, sample_chunk):
        s1 = min(S, s0 + sample_chunk)
        sc = s1 - s0

        cond_rep = cond.repeat(sc, 1, 1)
        dtype = cond_rep.dtype

        if is_single:
            y_t = torch.randn(sc * B, model.y_dim, device=device, dtype=dtype) * z_temp
        else:
            if shared_noise:
                base = torch.randn(sc * B, model.y_dim, 1, device=device, dtype=dtype) * z_temp
                y_t = base.expand(sc * B, model.y_dim, N).contiguous()
            else:
                y_t = torch.randn(sc * B, model.y_dim, N, device=device, dtype=dtype) * z_temp

        with torch.no_grad():
            for ti in range(model.T - 1, K - 1, -1):
                t = torch.full((sc * B,), ti, device=device, dtype=torch.long)
                y_t = model.p_sample(y_t, t, cond_rep, grad_enabled=False, z_temp=z_temp)

        with torch.set_grad_enabled(grad_enabled):
            for ti in range(K - 1, -1, -1):
                t = torch.full((sc * B,), ti, device=device, dtype=torch.long)
                y_t = model.p_sample(y_t, t, cond_rep, grad_enabled=grad_enabled, z_temp=z_temp)

        y_out = y_t.view(sc, B, model.y_dim) if is_single else y_t.view(sc, B, model.y_dim, N)

        if force_fp32 and y_out.dtype != torch.float32:
            y_out = y_out.float()

        if cpu_offload:
            y_out = y_out.cpu()

        outs.append(y_out)

        del cond_rep, y_t, y_out
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return torch.cat(outs, dim=0)

def ddim_sample_parallel_chunked_single(
    model,
    cond,  # [B,24,F]
    n_samples=50,
    n_steps=50,
    eta=0.0,
    sample_chunk=10,
    shared_noise=True,
    grad_enabled=None,
    cpu_offload=False,
    trunc_steps=10,
    force_fp32=True,
    z_temp=1.0,
):
    if grad_enabled is None:
        grad_enabled = torch.is_grad_enabled()
    if grad_enabled and cpu_offload:
        raise ValueError("Training (grad_enabled=True) requires cpu_offload=False.")

    z_temp = float(z_temp)

    model_device = next(model.parameters()).device
    cond = cond.to(device=model_device, dtype=torch.float32, non_blocking=True)

    device = model_device
    B = cond.shape[0]
    S = int(n_samples)

    if force_fp32:
        model = model.float()
        if cond.dtype != torch.float32:
            cond = cond.float()

    N = int(getattr(model, "n_nodes", 1))
    is_single = (N == 1)

    T = int(model.T)
    if n_steps is None or int(n_steps) >= T:
        ts = list(range(T - 1, -1, -1))
    else:
        n_steps = max(1, int(n_steps))
        idx = torch.linspace(0, T - 1, n_steps, device=device).round().long().tolist()
        ts = list(sorted(set(idx), reverse=True))
        if len(ts) == 0:
            ts = [T - 1]
        if ts[0] != T - 1:
            ts = [T - 1] + ts
        if ts[-1] != 0:
            ts = ts + [0]

    if len(ts) < 2:
        raise ValueError(f"Bad DDIM schedule: ts={ts} (len<2). T={T}, n_steps={n_steps}")

    n_pairs = len(ts) - 1

    K = int(trunc_steps)
    K = max(0, min(K, n_pairs))

    outs = []
    for s0 in range(0, S, sample_chunk):
        s1 = min(S, s0 + sample_chunk)
        sc = s1 - s0

        cond_rep = cond.repeat(sc, 1, 1)
        dtype = cond_rep.dtype

        if is_single:
            y_t = torch.randn(sc * B, model.y_dim, device=device, dtype=dtype) * z_temp
        else:
            if shared_noise:
                base = torch.randn(sc * B, model.y_dim, 1, device=device, dtype=dtype) * z_temp
                y_t = base.expand(sc * B, model.y_dim, N).contiguous()
            else:
                y_t = torch.randn(sc * B, model.y_dim, N, device=device, dtype=dtype) * z_temp

        with torch.no_grad():
            for k in range(0, n_pairs - K):
                ti, ti_prev = ts[k], ts[k + 1]
                t = torch.full((sc * B,), ti, device=device, dtype=torch.long)
                t_prev = torch.full((sc * B,), ti_prev, device=device, dtype=torch.long)
                y_t = model.ddim_step(y_t, t, t_prev, cond_rep, eta=eta, grad_enabled=False, z_temp=z_temp)

        with torch.set_grad_enabled(grad_enabled):
            for k in range(n_pairs - K, n_pairs):
                ti, ti_prev = ts[k], ts[k + 1]
                t = torch.full((sc * B,), ti, device=device, dtype=torch.long)
                t_prev = torch.full((sc * B,), ti_prev, device=device, dtype=torch.long)
                y_t = model.ddim_step(y_t, t, t_prev, cond_rep, eta=eta, grad_enabled=grad_enabled, z_temp=z_temp)

        t0 = torch.full((sc * B,), 0, device=device, dtype=torch.long)
        v = model.predict_v(y_t, t0, cond_rep)
        y_t = model._predict_x0_from_v(y_t, t0, v)

        y_out = y_t.view(sc, B, model.y_dim) if is_single else y_t.view(sc, B, model.y_dim, N)

        if force_fp32 and y_out.dtype != torch.float32:
            y_out = y_out.float()

        if cpu_offload:
            y_out = y_out.cpu()

        outs.append(y_out)

        del cond_rep, y_t, y_out
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return torch.cat(outs, dim=0)


@torch.no_grad()
def select_best_ztemp_for_single_node_diffusion(
    model,
    val_set,
    device,
    ztemps,
    n_samples=50,
    batch_size=256,
    quantiles=None,
    mode="ddim",
    n_steps=50,
    eta=0.0,
    sample_chunk=50,
    verbose=False,
):
    """
    On single-node val_set, pick z_temp by avg pinball loss.

    Assumes val_set __getitem__ returns (X,y):
      X: [24,F], y: [24] or [24,1]
    """
    if quantiles is None:
        quantiles = [0.05 * i for i in range(1, 20)]  # 0.05,...,0.95

    model.eval()
    loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=False)

    best_t, best_score = None, float("inf")
    table = []

    for tval in list(ztemps):
        scores = []
        for X, y in loader:
            X = X.to(device).float()  # [B,24,F]
            y = y.to(device).float()  # [B,24] or [B,24,1]

            if mode == "ddpm":
                samp = ddpm_sample_parallel_chunked_single(
                    model,
                    X,
                    n_samples=n_samples,
                    sample_chunk=sample_chunk,
                    shared_noise=True,
                    grad_enabled=False,
                    cpu_offload=False,
                    trunc_steps=0,
                    force_fp32=True,
                    z_temp=float(tval),
                )  # [S,B,24]
            else:
                samp = ddim_sample_parallel_chunked_single(
                    model,
                    X,
                    n_samples=n_samples,
                    n_steps=n_steps,
                    eta=eta,
                    sample_chunk=sample_chunk,
                    shared_noise=True,
                    grad_enabled=False,
                    cpu_offload=False,
                    trunc_steps=0,
                    force_fp32=True,
                    z_temp=float(tval),
                )  # [S,B,24]

            pb = pinball_loss_multiq(y, samp, quantiles=quantiles)
            scores.append(pb)

        score = float(np.mean(scores)) if scores else float("inf")
        table.append((float(tval), score))

        if verbose:
            print(f"  z_temp={tval:>6} | val pinball{tuple(quantiles)}={score:.6f}")

        if score < best_score:
            best_score = score
            best_t = float(tval)

    return best_t, best_score, table


@torch.no_grad()
def sample_window_diffusion_single(
    models_s,
    handlers_s,
    pack_data_s,
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
    z_temp=None,   # None -> per-node best_z_temp
):
    node_names = list(target_nodes)
    N = len(node_names)
    split = str(split)

    first_node = node_names[0]
    if "splits" in pack_data_s[first_node]:
        if split not in pack_data_s[first_node]["splits"]:
            raise KeyError(f"split={split} not in pack_data_s[{first_node}]['splits']")
        X0 = pack_data_s[first_node]["splits"][split]["X"]
    else:
        X0 = pack_data_s[first_node]["X_test"]
        split = "test"

    D = int(X0.shape[0])
    if start_day < 0 or start_day >= D:
        raise ValueError(f"start_day={start_day} out of range, total_days({split})={D}")

    actual_days = min(int(horizon_days), D - int(start_day))
    L = actual_days * int(seq_len)

    Y_true = np.zeros((N, L), dtype=np.float32)
    Y_pred = np.zeros((int(n_samples), N, L), dtype=np.float32)

    for j, node in enumerate(node_names):
        model = models_s[node].to(dtype=torch.float32)
        device = next(model.parameters()).device
        model.eval()

        if "splits" in pack_data_s[node]:
            X_all = pack_data_s[node]["splits"][split]["X"]
            y_all = pack_data_s[node]["splits"][split]["y"]
            handler = handlers_s[node][split]
        else:
            X_all = pack_data_s[node]["X_test"]
            y_all = pack_data_s[node]["y_test"]
            handler = handlers_s[node]

        zt = z_temp
        if zt is None:
            zt = pack_data_s[node].get("best_z_temp", 1.0)
            if zt is None:
                zt = 1.0
        zt = float(zt)

        for d0 in range(0, actual_days, int(day_chunk)):
            d1 = min(actual_days, d0 + int(day_chunk))
            B_blk = d1 - d0

            g0 = int(start_day) + d0
            g1 = int(start_day) + d1

            X_blk = X_all[g0:g1].to(device=device, dtype=torch.float32, non_blocking=True)
            y_blk = y_all[g0:g1].detach().cpu().numpy()

            L0, L1 = d0 * seq_len, d1 * seq_len

            y_real = handler.inverse_transform(y_blk.reshape(-1)).reshape(B_blk, seq_len)
            Y_true[j, L0:L1] = y_real.reshape(-1).astype(np.float32)

            if mode == "ddpm":
                samples_norm = ddpm_sample_parallel_chunked_single(
                    model,
                    X_blk,
                    n_samples=int(n_samples),
                    sample_chunk=int(sample_chunk),
                    cpu_offload=True,
                    grad_enabled=False,
                    trunc_steps=0,
                    force_fp32=True,
                    z_temp=zt,
                ).numpy()
            elif mode == "ddim":
                samples_norm = ddim_sample_parallel_chunked_single(
                    model,
                    X_blk,
                    n_samples=int(n_samples),
                    n_steps=int(n_steps),
                    eta=float(eta),
                    sample_chunk=int(sample_chunk),
                    cpu_offload=True,
                    grad_enabled=False,
                    trunc_steps=0,
                    force_fp32=True,
                    z_temp=zt,
                ).numpy()
            else:
                raise ValueError(f"Unknown mode: {mode}")

            samples_real = handler.scaler_y.inverse_transform(
                samples_norm.reshape(-1, 1)
            ).reshape(int(n_samples), B_blk, seq_len)

            Y_pred[:, j, L0:L1] = samples_real.reshape(int(n_samples), -1).astype(np.float32)

            del X_blk
            if torch.cuda.is_available():
                torch.cuda.synchronize()

    return dict(
        mode=mode,
        split=split,
        target_nodes=node_names,
        start_day=int(start_day),
        horizon_days=int(actual_days),
        seq_len=int(seq_len),
        n_samples=int(n_samples),
        z_temp=z_temp,  # compatibility; actual may be per-node if None
        Y_true=Y_true,
        Y_pred=Y_pred,
    )


class Multi_diffusion_predictor(nn.Module):
    """
    多节点 diffusion 封装：每节点一个 DDPM。
    X_all: [B,N,24,F]
    output:
        scenarios: [S,B,N,24] (real scale if return_real_scale=True)
    """

    def __init__(
        self,
        node_models: dict,
        scaler_y_map: dict,
        node_order: list,
        dtype=torch.float32,
        node_device_map=None,
        output_device=None,
    ):
        super().__init__()
        self.node_order = list(node_order)
        self.dtype = dtype

        if node_device_map is None:
            first_node = self.node_order[0]
            first_dev = next(node_models[first_node].parameters()).device
            node_device_map = {k: first_dev for k in self.node_order}
        self.node_device_map = {k: torch.device(v) for k, v in node_device_map.items()}

        if output_device is None:
            output_device = self.node_device_map[self.node_order[0]]
        self.output_device = torch.device(output_device)

        # 每个节点一个 diffusion model
        self.models = nn.ModuleDict()
        for k in self.node_order:
            dev = self.node_device_map[k]
            self.models[k] = copy.deepcopy(node_models[k]).float().to(dev).eval()

        # scaler buffers：固定放到 output_device
        means, scales = [], []
        for node in self.node_order:
            sc = scaler_y_map[node]
            m = np.asarray(sc.mean_, dtype=np.float32).reshape(1)
            s = np.asarray(
                getattr(sc, "scale_", getattr(sc, "std_", None)),
                dtype=np.float32
            ).reshape(1)
            means.append(torch.tensor(m, dtype=torch.float32))
            scales.append(torch.tensor(s, dtype=torch.float32))

        mean = torch.stack(means, dim=0).reshape(len(self.node_order), 1, 1)
        scale = torch.stack(scales, dim=0).reshape(len(self.node_order), 1, 1)
        self.register_buffer("y_mean_N11", mean.to(self.output_device))
        self.register_buffer("y_scale_N11", scale.to(self.output_device))

        # 初始化后做一次强制归位，避免 deepcopy / 外部 load 后不稳
        self.relocate_models()

    def relocate_models(self):
        """
        强制每个 node model 回到 node_device_map 指定设备，
        并确保 scaler buffer 在 output_device。
        """
        for k in self.node_order:
            dev = self.node_device_map[k]
            self.models[k] = self.models[k].float().to(dev).eval()

        self._buffers["y_mean_N11"] = self._buffers["y_mean_N11"].to(
            self.output_device, dtype=torch.float32
        )
        self._buffers["y_scale_N11"] = self._buffers["y_scale_N11"].to(
            self.output_device, dtype=torch.float32
        )
        return self

    def assert_device_consistency(self):
        """
        检查每个 node 的 diffusion model 是否内部跨设备混放。
        """
        for k in self.node_order:
            m = self.models[k]
            devs = set()

            for _, p in m.named_parameters():
                devs.add(str(p.device))
            for _, b in m.named_buffers():
                devs.add(str(b.device))

            if len(devs) > 1:
                detail_lines = [f"[DeviceMismatch] predictor.models[{k}] spans devices: {devs}"]
                for n, p in m.named_parameters():
                    detail_lines.append(f"  PARAM  {n}: {p.device}")
                for n, b in m.named_buffers():
                    detail_lines.append(f"  BUFFER {n}: {b.device}")
                raise RuntimeError("\n".join(detail_lines))
        return True

    def train(self, mode: bool = True):
        for node in self.node_order:
            self.models[node].train(mode)
        return self

    def eval(self):
        for node in self.node_order:
            self.models[node].eval()
        return self

    def inverse_y(self, y_norm_BNT: torch.Tensor) -> torch.Tensor:
        mean = self.y_mean_N11.to(y_norm_BNT.device, dtype=torch.float32).view(1, -1, 1)
        scale = self.y_scale_N11.to(y_norm_BNT.device, dtype=torch.float32).view(1, -1, 1)
        return y_norm_BNT.to(torch.float32) * scale + mean

    @torch.no_grad()
    def mean_std(self, X_all, n_samples=50, **sample_kwargs):
        Y = self.sample(
            X_all,
            n_samples=n_samples,
            return_real_scale=True,
            grad_enabled=False,
            **sample_kwargs
        )
        return Y.mean(dim=0), Y.std(dim=0, unbiased=False)

    def sample(
        self,
        X_all,
        n_samples=50,
        sample_chunk=10,
        cpu_offload=False,
        return_real_scale=True,
        grad_enabled=None,
        trunc_steps=10,
        shared_noise=True,
        mode="ddim",
        n_steps=50,
        eta=0.0,
        force_relocate=True,
        check_device_consistency=True,
    ):
        if grad_enabled is None:
            grad_enabled = torch.is_grad_enabled()
        if grad_enabled and cpu_offload:
            raise ValueError("Training (grad_enabled=True) requires cpu_offload=False.")

        # 兜底：每次采样前都重新把各 node 模型放回指定 GPU
        if force_relocate:
            self.relocate_models()

        if check_device_consistency:
            self.assert_device_consistency()

        X_all = X_all.to(dtype=torch.float32)
        B, N, T, F = X_all.shape
        S = int(n_samples)

        if N != len(self.node_order):
            raise ValueError(
                f"Input node dim N={N} does not match len(node_order)={len(self.node_order)}"
            )

        node_samples = []
        for j, node in enumerate(self.node_order):
            model = self.models[node]
            node_dev = next(model.parameters()).device

            X_j = X_all[:, j, :, :].to(
                device=node_dev,
                dtype=torch.float32,
                non_blocking=True
            )

            if mode == "ddpm":
                y_s_norm = ddpm_sample_parallel_chunked_single(
                    model,
                    X_j,
                    n_samples=S,
                    sample_chunk=sample_chunk,
                    shared_noise=shared_noise,
                    grad_enabled=grad_enabled,
                    cpu_offload=cpu_offload,
                    trunc_steps=trunc_steps,
                    force_fp32=True,
                )
            elif mode == "ddim":
                y_s_norm = ddim_sample_parallel_chunked_single(
                    model,
                    X_j,
                    n_samples=S,
                    n_steps=n_steps,
                    eta=eta,
                    sample_chunk=sample_chunk,
                    shared_noise=shared_noise,
                    grad_enabled=grad_enabled,
                    cpu_offload=cpu_offload,
                    trunc_steps=trunc_steps,
                    force_fp32=True,
                )
            else:
                raise ValueError(f"Unknown mode: {mode}")

            if y_s_norm.ndim == 4:
                if y_s_norm.shape[-1] != 1:
                    raise ValueError(
                        f"Expect last dim=1, got {tuple(y_s_norm.shape)} for node={node}."
                    )
                y_s_norm = y_s_norm[..., 0]

            if y_s_norm.shape != (S, B, T):
                raise ValueError(
                    f"Expect y_s_norm shape {(S, B, T)} but got {tuple(y_s_norm.shape)} for node={node}"
                )

            y_s_norm = y_s_norm.to(
                self.output_device,
                dtype=torch.float32,
                non_blocking=True
            )
            node_samples.append(y_s_norm)

            del y_s_norm, X_j
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        YS = torch.stack(node_samples, dim=2).float()  # [S,B,N,24]

        if not return_real_scale:
            return YS

        mean = self.y_mean_N11.to(YS.device, dtype=torch.float32).view(1, 1, -1, 1)
        scale = self.y_scale_N11.to(YS.device, dtype=torch.float32).view(1, 1, -1, 1)

        return YS * scale + mean

    def forward(self, X_all, **kwargs):
        return self.sample(X_all, **kwargs)


def _get_dfl_main_device(module: nn.Module):
    for p in module.parameters():
        return p.device
    for b in module.buffers():
        return b.device
    return torch.device("cpu")




class DFL_model_diffusion_Deterministic_SingleNode(nn.Module):
    def __init__(
        self,
        mgr,
        optnet_DA,
        optnet_RT,
        predictor,
        scenario_filter=None,
        n_scen=20,
        solver="ECOS",
        clamp_min=0.0,
    ):
        super().__init__()
        self.mgr = mgr
        self.optnet_DA = optnet_DA
        self.optnet_RT = optnet_RT

        # 保留版本2：不要把 predictor 注册成子模块，避免 dfl.to(...) 破坏多卡布局
        self.__dict__["predictor"] = predictor

        self.scenario_filter = scenario_filter
        self.n_scen = int(n_scen)
        self.solver = solver
        self.clamp_min = clamp_min

        self.optnet_dtype = torch.float64

        # 补回版本1里 deterministic single 需要的成本参数
        b = torch.tensor(
            [mgr.gen_info[g]["b"] for g in mgr.gen_bus_list],
            dtype=self.optnet_dtype
        )
        self.register_buffer("b_G", b)

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

        # 恢复版本1的 RT 接口：single-node RT 吃 (R_up, R_dn, omega_true)
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
        """
        Deterministic baseline for diffusion predictor:
        1) generate scenarios
        2) optional clamp
        3) optional scenario filtering
        4) average scenarios into one deterministic forecast
        5) solve DA once
        6) evaluate realized total cost on true y
        """
        del hourly_load_min_sys, hourly_load_max_sys, eps  # compatibility only

        solver = solver or self.solver
        device = _get_dfl_main_device(self)
        dtype = self.optnet_dtype
        predictor_kwargs = dict(predictor_kwargs or {})
        filter_kwargs = dict(filter_kwargs or {})

        S = int(predictor_n_samples) if predictor_n_samples is not None else self.n_scen

        pred_out = self.predictor(
            X_all.float(),
            n_samples=S,
            return_real_scale=True,
            grad_enabled=torch.is_grad_enabled(),
            **predictor_kwargs,
        )

        if isinstance(pred_out, dict):
            Y_scen = pred_out.get("samples", pred_out.get("y_samples"))
        else:
            Y_scen = pred_out

        if Y_scen is None:
            raise ValueError("predictor must return scenario samples for deterministic baseline.")

        Y_scen = Y_scen.to(device=device, dtype=dtype)

        if self.clamp_min is not None:
            Y_scen = torch.clamp(Y_scen, min=float(self.clamp_min))

        aux_filter = None
        if self.scenario_filter is not None:
            filt_out = self.scenario_filter(
                Y_scen,
                is_train=self.training,
                **filter_kwargs,
            )
            if isinstance(filt_out, tuple):
                Y_scen = filt_out[0]
                if len(filt_out) > 1:
                    aux_filter = filt_out[1]
            else:
                Y_scen = filt_out

            Y_scen = Y_scen.to(device=device, dtype=dtype)

        if Y_scen.dim() != 4:
            raise ValueError(f"Unexpected Y_scen shape: {tuple(Y_scen.shape)}")

        # [S,B,11,T] -> [B,T]
        forecast_sys = Y_scen.sum(dim=2).mean(dim=0)

        # y_true -> real scale -> omega_true
        T_len = forecast_sys.shape[-1]
        y_true_11 = self._canon_y_true_11T(y_true, T=T_len).to(device=device)
        y_true_real11 = self.predictor.inverse_y(y_true_11.float()).to(device=device, dtype=dtype)
        omega_true = y_true_real11.sum(dim=1) - forecast_sys  # [B,T]

        B = forecast_sys.shape[0]
        loss_list = []

        aux = None
        if return_aux:
            aux = {
                "forecast_sys": [],
                "omega_true": [],
                "det_obj": [],
                "realized_total": [],
                "rt_obj_true": [],
                "Y_scen": Y_scen.detach(),
            }

        for b in range(B):
            sol_da = self.optnet_DA(
                forecast_sys[b].unsqueeze(0),
                solver=solver,
            )

            P_DA = sol_da["P_DA"][0].to(device=device, dtype=dtype)
            R_up = sol_da["R_up"][0].to(device=device, dtype=dtype)
            R_dn = sol_da["R_dn"][0].to(device=device, dtype=dtype)

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
                    aux["det_obj"].append(
                        torch.tensor(float("nan"), device=device, dtype=dtype)
                    )

                aux["realized_total"].append(realized_total.detach())
                aux["rt_obj_true"].append(sol_rt["rt_obj"][0].detach())

        loss_vec = torch.stack(loss_list, dim=0)  # [B]

        if return_aux:
            for k in ["forecast_sys", "omega_true", "det_obj", "realized_total", "rt_obj_true"]:
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
        n_scen=20,
        solver="ECOS",
        clamp_min=0.0,
    ):
        super().__init__()
        self.mgr = mgr
        self.optnet_DA = optnet_DA
        self.optnet_RT = optnet_RT

        # 保留版本2：不要注册 predictor
        self.__dict__["predictor"] = predictor

        self.scenario_filter = scenario_filter
        self.n_scen = int(n_scen)
        self.solver = solver
        self.clamp_min = clamp_min

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
        """
        Deterministic baseline for diffusion predictor:
        1) generate scenarios
        2) optional clamp
        3) optional scenario filtering
        4) average scenarios into one deterministic forecast
        5) solve DA once
        6) evaluate realized RT cost on true y
        """
        del hourly_load_min_sys, hourly_load_max_sys, eps  # compatibility only

        solver = solver or self.solver
        device = _get_dfl_main_device(self)
        dtype = self.optnet_dtype
        predictor_kwargs = dict(predictor_kwargs or {})
        filter_kwargs = dict(filter_kwargs or {})

        S = int(predictor_n_samples) if predictor_n_samples is not None else self.n_scen

        pred_out = self.predictor(
            X_all.float(),
            n_samples=S,
            return_real_scale=True,
            grad_enabled=torch.is_grad_enabled(),
            **predictor_kwargs,
        )

        if isinstance(pred_out, dict):
            Y_scen = pred_out.get("samples", pred_out.get("y_samples"))
        else:
            Y_scen = pred_out

        if Y_scen is None:
            raise ValueError("predictor must return scenario samples for deterministic baseline.")

        Y_scen = Y_scen.to(device=device, dtype=dtype)

        if self.clamp_min is not None:
            Y_scen = torch.clamp(Y_scen, min=float(self.clamp_min))

        aux_filter = None
        if self.scenario_filter is not None:
            filt_out = self.scenario_filter(
                Y_scen,
                is_train=self.training,
                **filter_kwargs,
            )
            if isinstance(filt_out, tuple):
                Y_scen = filt_out[0]
                if len(filt_out) > 1:
                    aux_filter = filt_out[1]
            else:
                Y_scen = filt_out

            Y_scen = Y_scen.to(device=device, dtype=dtype)

        if Y_scen.dim() != 4:
            raise ValueError(f"Unexpected Y_scen shape: {tuple(Y_scen.shape)}")

        # [S,B,11,T] -> [B,11,T]
        forecast11 = Y_scen.mean(dim=0)
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
        y_true_real11 = self.predictor.inverse_y(y_true_11.float()).to(device=device, dtype=dtype)
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
        total_realized_cost = cost_energy + cost_reserve + rt_obj_true  # [B]

        aux = None
        if return_aux:
            aux = {
                "forecast11": forecast11.detach(),
                "forecast14": forecast14.detach(),
                "Y_scen": Y_scen.detach(),
                "P_DA": P_DA.detach(),
                "R_up": R_up.detach(),
                "R_dn": R_dn.detach(),
                "y_true_14": y_true_14.detach(),
                "rt_obj_true": rt_obj_true.detach(),
                "cost_energy": cost_energy.detach(),
                "cost_reserve": cost_reserve.detach(),
                "total_realized_cost": total_realized_cost.detach(),
            }
            obj = sol_DA.get("obj", None)
            if isinstance(obj, torch.Tensor):
                aux["det_obj"] = obj.detach()

        if return_aux and not return_filter_aux:
            return total_realized_cost, aux
        if return_filter_aux and not return_aux:
            return total_realized_cost, aux_filter
        if return_aux and return_filter_aux:
            return total_realized_cost, aux, aux_filter
        return total_realized_cost

class DFL_model_vae_Deterministic_MultiNode(nn.Module):
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

        # 关键：不要把 predictor 注册成子模块，否则 dfl.to(...) 会破坏多卡布局
        self.__dict__["predictor"] = predictor

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
        hourly_load_min_sys=None,  # 占位：兼容 DRO 接口
        hourly_load_max_sys=None,  # 占位：兼容 DRO 接口
        eps=None,                  # 占位：兼容 DRO 接口
        solver=None,
        return_aux=False,
        return_filter_aux=False,
        predictor_n_samples=None,
        predictor_kwargs=None,
        filter_kwargs=None,
    ):
        solver = solver or self.solver
        device = _get_dfl_main_device(self)
        dtype = self.optnet_dtype
        predictor_kwargs = dict(predictor_kwargs or {})
        filter_kwargs = dict(filter_kwargs or {})

        S = int(predictor_n_samples) if predictor_n_samples is not None else self.n_scen

        Y_scen_11 = self.predictor(
            X_all.float(),
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

        # deterministic forecast = scenario mean
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
        y_true_real11 = self.predictor.inverse_y(y_true_11.float()).to(device=device, dtype=dtype)
        y_true_14 = self._map_11_to_14(y_true_real11)

        # 与 diffusion MultiNode 保持一致：RT 输入改为 (R_up, R_dn, omega14_true)
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
                "omega14_true": omega14_true.detach(),
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

class DFL_model_diffusion_SingleNode(nn.Module):
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

        # 关键：不要把 predictor 注册成子模块，否则 dfl.to(...) 会破坏多卡布局
        self.__dict__["predictor"] = predictor

        self.scenario_filter = scenario_filter
        self.n_scen = int(n_scen)
        self.clamp_min = clamp_min
        self.solver = solver

        self.optnet_dtype = torch.float64

        b = torch.tensor([mgr.gen_info[g]["b"] for g in mgr.gen_bus_list], dtype=self.optnet_dtype)
        self.register_buffer("b_G", b)

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
            solver=solver
        )
        rt_obj_true = sol_rt["rt_obj"][0]
        return stage1_energy_linear + stage1_reserve + rt_obj_true, sol_rt

    def forward(
        self,
        X_all,
        y_true,
        solver=None,
        return_aux=False,
        return_filter_aux=False,
        predictor_n_samples=None,
        predictor_kwargs=None,
        filter_kwargs=None,
    ):
        solver = solver or self.solver
        device = _get_dfl_main_device(self)
        predictor_kwargs = dict(predictor_kwargs or {})
        filter_kwargs = dict(filter_kwargs or {})

        S = int(predictor_n_samples) if predictor_n_samples is not None else self.n_scen
        Y_scen = self.predictor(
            X_all.float(),
            n_samples=S,
            return_real_scale=True,
            grad_enabled=torch.is_grad_enabled(),
            **predictor_kwargs
        ).to(device=device, dtype=self.optnet_dtype)

        if self.clamp_min is not None:
            Y_scen = torch.clamp(Y_scen, min=float(self.clamp_min))

        aux_filter = None
        if self.scenario_filter is not None:
            Y_scen, aux_filter = self.scenario_filter(Y_scen, is_train=self.training, **filter_kwargs)
            Y_scen = Y_scen.to(device=device, dtype=self.optnet_dtype)

        scen_sys = Y_scen.sum(dim=2)
        forecast_sys = scen_sys.mean(dim=0)
        omega_saa = (scen_sys - forecast_sys.unsqueeze(0)).permute(1, 0, 2).contiguous()

        y_true_11 = self._canon_y_true_11T(y_true).to(device=device)
        y_true_real = self.predictor.inverse_y(y_true_11.float()).to(device=device, dtype=self.optnet_dtype)
        omega_true = y_true_real.sum(dim=1) - forecast_sys

        B = X_all.shape[0]
        loss_list = []

        aux = None
        if return_aux:
            aux = {
                "forecast_sys": [],
                "omega_true": [],
                "saa_obj": [],
                "realized_total": [],
                "rt_obj_true": [],
            }

        for b in range(B):
            sol_da = self.optnet_DA(
                forecast_sys[b].unsqueeze(0),
                omega_saa[b].unsqueeze(0),
                solver=solver,
                return_rt=False
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
                    aux["saa_obj"].append(obj[0].detach())
                else:
                    aux["saa_obj"].append(torch.tensor(float("nan"), device=device, dtype=self.optnet_dtype))
                aux["realized_total"].append(realized_total.detach())
                aux["rt_obj_true"].append(sol_rt["rt_obj"][0].detach())

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

class DFL_model_diffusion_MultiNode(nn.Module):
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

        # 关键：不要注册 predictor
        self.__dict__["predictor"] = predictor

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
        solver=None,
        return_aux=False,
        return_filter_aux=False,
        predictor_n_samples=None,
        predictor_kwargs=None,
        filter_kwargs=None,
    ):
        solver = solver or self.solver
        device = _get_dfl_main_device(self)
        dtype = self.optnet_dtype
        predictor_kwargs = dict(predictor_kwargs or {})
        filter_kwargs = dict(filter_kwargs or {})

        S = int(predictor_n_samples) if predictor_n_samples is not None else self.n_scen
        Y_scen_11 = self.predictor(
            X_all.float(),
            n_samples=S,
            return_real_scale=True,
            grad_enabled=torch.is_grad_enabled(),
            **predictor_kwargs
        ).to(device=device, dtype=dtype)

        if self.clamp_min is not None:
            Y_scen_11 = torch.clamp(Y_scen_11, min=float(self.clamp_min))

        aux_filter = None
        if self.scenario_filter is not None:
            Y_scen_11, aux_filter = self.scenario_filter(
                Y_scen_11,
                is_train=self.training,
                **filter_kwargs
            )
            Y_scen_11 = Y_scen_11.to(device=device, dtype=dtype)

        forecast11 = Y_scen_11.mean(dim=0)                         # [B,11,T]
        scen11 = Y_scen_11.permute(1, 0, 2, 3).contiguous()       # [B,S,11,T]
        forecast14 = self._map_11_to_14(forecast11)               # [B,14,T]
        scen14 = self._map_11_to_14(scen11)                       # [B,S,14,T]
        omega14 = scen14 - forecast14.unsqueeze(1)                # [B,S,14,T]

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

        T_len = forecast14.shape[-1]
        y_true_11 = self._canon_y_true_11T(y_true, T=T_len).to(device=device)
        y_true_real11 = self.predictor.inverse_y(y_true_11.float()).to(device=device, dtype=dtype)
        y_true_14 = self._map_11_to_14(y_true_real11)

        # 关键修改：RT 输入改为 (R_up, R_dn, omega14_true)
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
                "omega14": omega14.detach(),
                "omega14_true": omega14_true.detach(),
                "P_DA": P_DA.detach(),
                "R_up": R_up.detach(),
                "R_dn": R_dn.detach(),
                "cost_energy": cost_energy.detach(),
                "cost_reserve": cost_reserve.detach(),
                "rt_obj_true": rt_obj_true.detach(),
            }

        if return_aux and not return_filter_aux:
            return total_realized_cost, aux
        if return_filter_aux and not return_aux:
            return total_realized_cost, aux_filter
        if return_aux and return_filter_aux:
            return total_realized_cost, aux, aux_filter
        return total_realized_cost


class DFL_model_diffusion_DRO_SingleNode(nn.Module):
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

        # 关键：不要注册 predictor
        self.__dict__["predictor"] = predictor

        self.scenario_filter = scenario_filter

        self.n_scen = int(n_scen)
        self.clamp_min = clamp_min
        self.solver = solver
        self.optnet_dtype = torch.float64

        b = torch.tensor([mgr.gen_info[g]["b"] for g in mgr.gen_bus_list], dtype=self.optnet_dtype)
        self.register_buffer("b_G", b)

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
        return_filter_aux=False,
        predictor_n_samples=None,
        predictor_kwargs=None,
        filter_kwargs=None,
    ):
        solver = solver or self.solver
        device = _get_dfl_main_device(self)
        dtype = self.optnet_dtype
        predictor_kwargs = dict(predictor_kwargs or {})
        filter_kwargs = dict(filter_kwargs or {})

        B = X_all.shape[0]
        S = int(predictor_n_samples) if predictor_n_samples is not None else self.n_scen

        Y_scen = self.predictor(
            X_all.float(),
            n_samples=S,
            return_real_scale=True,
            grad_enabled=torch.is_grad_enabled(),
            **predictor_kwargs
        ).to(device=device, dtype=dtype)

        if self.clamp_min is not None:
            Y_scen = torch.clamp(Y_scen, min=float(self.clamp_min))

        aux_filter = None
        if self.scenario_filter is not None:
            Y_scen, aux_filter = self.scenario_filter(Y_scen, is_train=self.training, **filter_kwargs)
            Y_scen = Y_scen.to(device=device, dtype=dtype)

        scen_sys = Y_scen.sum(dim=2)
        forecast_sys = scen_sys.mean(dim=0)
        omega_scen = (scen_sys - forecast_sys.unsqueeze(0)).permute(1, 0, 2).contiguous()

        Lmin_sys = self._to_BT(hourly_load_min_sys, B=B, device=device, dtype=dtype)
        Lmax_sys = self._to_BT(hourly_load_max_sys, B=B, device=device, dtype=dtype)

        om_min_h = Lmin_sys - forecast_sys
        om_max_h = Lmax_sys - forecast_sys
        om_min_h, om_max_h = torch.minimum(om_min_h, om_max_h), torch.maximum(om_min_h, om_max_h)

        om_min_s = omega_scen.min(dim=1).values
        om_max_s = omega_scen.max(dim=1).values

        om_min = torch.minimum(om_min_h, om_min_s)
        om_max = torch.maximum(om_max_h, om_max_s)

        y_true_11 = self._canon_y_true_11T(y_true).to(device=device)
        y_true_real = self.predictor.inverse_y(y_true_11.float()).to(device=device, dtype=dtype)
        omega_true = y_true_real.sum(dim=1) - forecast_sys

        eps_t = torch.as_tensor(eps, device=device, dtype=dtype)
        if eps_t.ndim == 0:
            eps_t = eps_t.view(1).expand(B)

        loss_list = []
        aux = None
        if return_aux:
            aux = {
                "forecast_sys": [],
                "omega_true": [],
                "dro_obj": [],
                "gamma": [],
                "realized_total": [],
                "rt_obj_true": [],
                "om_min": [],
                "om_max": [],
            }


        # print('forecast14',forecast_sys[0])
        # print('omega14',omega_scen[0])
        # print('om_min',om_min[0])
        # print('om_max',om_max[0])
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
                aux["forecast_sys"].append(forecast_sys[b].detach())
                aux["omega_true"].append(omega_true[b].detach())
                obj = sol_da.get("obj", None)
                gamma = sol_da.get("gamma", None)
                aux["dro_obj"].append(
                    obj[0].detach() if isinstance(obj, torch.Tensor)
                    else torch.tensor(float("nan"), device=device, dtype=dtype)
                )
                aux["gamma"].append(
                    gamma[0].detach() if isinstance(gamma, torch.Tensor)
                    else torch.tensor(float("nan"), device=device, dtype=dtype)
                )
                aux["realized_total"].append(realized_total.detach())
                aux["rt_obj_true"].append(sol_rt["rt_obj"][0].detach())
                aux["om_min"].append(om_min[b].detach())
                aux["om_max"].append(om_max[b].detach())

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



class DFL_model_diffusion_DRO_MultiNode(nn.Module):
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

        # 不注册 predictor，避免 dfl.to(...) 破坏多卡布局
        self.__dict__["predictor"] = predictor

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
        hourly_load_min_sys,
        hourly_load_max_sys,
        eps,
        solver=None,
        return_aux=False,
        return_filter_aux=False,
        predictor_n_samples=None,
        predictor_kwargs=None,
        filter_kwargs=None,
    ):
        solver = solver or self.solver
        device = X_all.device
        dtype = self.optnet_dtype
        predictor_kwargs = dict(predictor_kwargs or {})
        filter_kwargs = dict(filter_kwargs or {})

        B = X_all.shape[0]
        S = int(predictor_n_samples) if predictor_n_samples is not None else self.n_scen

        Y_scen_11 = self.predictor(
            X_all.float(),
            n_samples=S,
            return_real_scale=True,
            grad_enabled=torch.is_grad_enabled(),
            **predictor_kwargs
        ).to(device=device, dtype=dtype)

        if self.clamp_min is not None:
            Y_scen_11 = torch.clamp(Y_scen_11, min=float(self.clamp_min))

        aux_filter = None
        if self.scenario_filter is not None:
            Y_scen_11, aux_filter = self.scenario_filter(
                Y_scen_11,
                is_train=self.training,
                **filter_kwargs
            )
            Y_scen_11 = Y_scen_11.to(device=device, dtype=dtype)

        forecast11 = Y_scen_11.mean(dim=0)                    # [B,11,T]
        scen11 = Y_scen_11.permute(1, 0, 2, 3).contiguous()  # [B,S,11,T]

        forecast14 = self._map_11_to_14(forecast11)          # [B,14,T]
        scen14 = self._map_11_to_14(scen11)                  # [B,S,14,T]
        omega14 = scen14 - forecast14.unsqueeze(1)           # [B,S,14,T]

        # ---- 对齐 VAE DRO 版：把 load bounds 转成 omega bounds ----
        Lmin14 = torch.as_tensor(hourly_load_min_sys, device=device, dtype=dtype)
        Lmax14 = torch.as_tensor(hourly_load_max_sys, device=device, dtype=dtype)

        if Lmin14.ndim == 2:
            Lmin14 = Lmin14.unsqueeze(0).expand(B, -1, -1).contiguous()
        if Lmax14.ndim == 2:
            Lmax14 = Lmax14.unsqueeze(0).expand(B, -1, -1).contiguous()

        om_min14 = Lmin14 - forecast14
        om_max14 = Lmax14 - forecast14
        om_min14 = torch.minimum(om_min14, om_max14)
        om_max14 = torch.maximum(om_min14, om_max14)

        eps_t = torch.as_tensor(eps, device=device, dtype=dtype)
        if eps_t.ndim == 0:
            eps_t = eps_t.view(1).expand(B)

        # print('forecast14',forecast14)
        # print('omega14',omega14)
        # print('om_min14',om_min14)
        # print('om_max14',om_max14)
        # ---- 关键：按 VAE 版接口，位置参数传入 ----
        sol_DA = self.optnet_DA(
            forecast14,
            omega14,
            om_min14,
            om_max14,
            eps_t,
            solver=solver,
        )

        P_DA = sol_DA["P_DA"].to(device=device, dtype=dtype)
        R_up = sol_DA["R_up"].to(device=device, dtype=dtype)
        R_dn = sol_DA["R_dn"].to(device=device, dtype=dtype)

        T_len = forecast14.shape[-1]
        y_true_11 = self._canon_y_true_11T(y_true, T=T_len).to(device=device)
        y_true_real11 = self.predictor.inverse_y(y_true_11.float()).to(device=device, dtype=dtype)
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
                "omega14": omega14.detach(),
                "om_min14": om_min14.detach(),
                "om_max14": om_max14.detach(),
                "omega14_true": omega14_true.detach(),
                "P_DA": P_DA.detach(),
                "R_up": R_up.detach(),
                "R_dn": R_dn.detach(),
                "rt_obj_true": rt_obj_true.detach(),
                "cost_energy": cost_energy.detach(),
                "cost_reserve": cost_reserve.detach(),
            }

        if return_aux and not return_filter_aux:
            return total_realized_cost, aux
        if return_filter_aux and not return_aux:
            return total_realized_cost, aux_filter
        if return_aux and return_filter_aux:
            return total_realized_cost, aux, aux_filter
        return total_realized_cost


def _resolve_separate_diffusion_z_temp(z_temp=None, pack_data_s=None, target_nodes=None, args=None, default=1.0):
    """
    解析 separate / multi-node diffusion 使用的 z_temp。
    支持:
      - None
      - float
      - dict[node] = float

    优先级:
      1) 显式传入 z_temp
      2) pack_data_s[node]["best_z_temp"]
      3) args.z_temp
      4) default
    """
    if target_nodes is None:
        if pack_data_s is None:
            raise ValueError("target_nodes and pack_data_s cannot both be None.")
        target_nodes = list(pack_data_s.keys())

    if isinstance(z_temp, dict):
        return {n: float(z_temp[n]) for n in target_nodes}

    if z_temp is not None:
        return {n: float(z_temp) for n in target_nodes}

    args_z = getattr(args, "z_temp", None) if args is not None else None
    out = {}

    for n in target_nodes:
        best_z = None
        if pack_data_s is not None and n in pack_data_s:
            best_z = pack_data_s[n].get("best_z_temp", None)

        if best_z is not None:
            out[n] = float(best_z)
        elif isinstance(args_z, dict):
            out[n] = float(args_z.get(n, default))
        elif args_z is not None:
            out[n] = float(args_z)
        else:
            out[n] = float(default)

    return out



class Multi_diffusion_predictor(nn.Module):
    """
    多节点 diffusion 封装：每节点一个 DDPM。
    X_all: [B,N,24,F]
    output:
        scenarios: [S,B,N,24] (real scale if return_real_scale=True)
    """

    def __init__(
        self,
        node_models: dict,
        scaler_y_map: dict,
        node_order: list,
        dtype=torch.float32,
        node_device_map=None,
        output_device=None,
        default_z_temp=1.0,
    ):
        super().__init__()
        self.node_order = list(node_order)
        self.dtype = dtype
        self.default_z_temp = default_z_temp  # float or dict

        if node_device_map is None:
            first_node = self.node_order[0]
            first_dev = next(node_models[first_node].parameters()).device
            node_device_map = {k: first_dev for k in self.node_order}
        self.node_device_map = {k: torch.device(v) for k, v in node_device_map.items()}

        if output_device is None:
            output_device = self.node_device_map[self.node_order[0]]
        self.output_device = torch.device(output_device)

        # 每个节点一个 diffusion model
        self.models = nn.ModuleDict()
        for k in self.node_order:
            dev = self.node_device_map[k]
            self.models[str(k)] = copy.deepcopy(node_models[k]).float().to(dev).eval()

        # scaler buffers：固定放到 output_device
        means, scales = [], []
        for node in self.node_order:
            sc = scaler_y_map[node]
            m = np.asarray(sc.mean_, dtype=np.float32).reshape(1)
            s_raw = getattr(sc, "scale_", getattr(sc, "std_", None))
            if s_raw is None:
                raise ValueError(f"Scaler for node={node} has no scale_/std_.")
            s = np.asarray(s_raw, dtype=np.float32).reshape(1)
            means.append(torch.tensor(m, dtype=torch.float32))
            scales.append(torch.tensor(s, dtype=torch.float32))

        mean = torch.stack(means, dim=0).reshape(len(self.node_order), 1, 1)
        scale = torch.stack(scales, dim=0).reshape(len(self.node_order), 1, 1)
        self.register_buffer("y_mean_N11", mean.to(self.output_device))
        self.register_buffer("y_scale_N11", scale.to(self.output_device))

        # 初始化后做一次强制归位，避免 deepcopy / 外部 load 后不稳
        self.relocate_models()

    def relocate_models(self):
        """
        强制每个 node model 回到 node_device_map 指定设备，
        并确保 scaler buffer 在 output_device。
        """
        for k in self.node_order:
            dev = self.node_device_map[k]
            self.models[str(k)] = self.models[str(k)].float().to(dev).eval()

        self._buffers["y_mean_N11"] = self._buffers["y_mean_N11"].to(
            self.output_device, dtype=torch.float32
        )
        self._buffers["y_scale_N11"] = self._buffers["y_scale_N11"].to(
            self.output_device, dtype=torch.float32
        )
        return self

    def assert_device_consistency(self):
        """
        检查每个 node 的 diffusion model 是否内部跨设备混放。
        """
        for k in self.node_order:
            m = self.models[str(k)]
            devs = set()

            for _, p in m.named_parameters():
                devs.add(str(p.device))
            for _, b in m.named_buffers():
                devs.add(str(b.device))

            if len(devs) > 1:
                detail_lines = [f"[DeviceMismatch] predictor.models[{k}] spans devices: {devs}"]
                for n, p in m.named_parameters():
                    detail_lines.append(f"  PARAM  {n}: {p.device}")
                for n, b in m.named_buffers():
                    detail_lines.append(f"  BUFFER {n}: {b.device}")
                raise RuntimeError("\n".join(detail_lines))
        return True

    def train(self, mode: bool = True):
        for node in self.node_order:
            self.models[str(node)].train(mode)
        return self

    def eval(self):
        for node in self.node_order:
            self.models[str(node)].eval()
        return self

    def inverse_y(self, y_norm_BNT: torch.Tensor) -> torch.Tensor:
        mean = self.y_mean_N11.to(y_norm_BNT.device, dtype=torch.float32).view(1, -1, 1)
        scale = self.y_scale_N11.to(y_norm_BNT.device, dtype=torch.float32).view(1, -1, 1)
        return y_norm_BNT.to(torch.float32) * scale + mean

    @torch.no_grad()
    def mean_std(self, X_all, n_samples=50, **sample_kwargs):
        Y = self.sample(
            X_all,
            n_samples=n_samples,
            return_real_scale=True,
            grad_enabled=False,
            **sample_kwargs
        )
        return Y.mean(dim=0), Y.std(dim=0, unbiased=False)

    def sample(
        self,
        X_all,
        n_samples=50,
        sample_chunk=10,
        cpu_offload=False,
        return_real_scale=True,
        grad_enabled=None,
        trunc_steps=10,
        shared_noise=True,
        mode="ddim",
        n_steps=50,
        eta=0.0,
        z_temp=None,   # NEW: None / float / dict[node] -> float
        force_relocate=True,
        check_device_consistency=True,
    ):
        if grad_enabled is None:
            grad_enabled = torch.is_grad_enabled()
        if grad_enabled and cpu_offload:
            raise ValueError("Training (grad_enabled=True) requires cpu_offload=False.")

        # 兜底：每次采样前都重新把各 node 模型放回指定 GPU
        if force_relocate:
            self.relocate_models()

        if check_device_consistency:
            self.assert_device_consistency()

        if z_temp is None:
            z_temp = self.default_z_temp

        X_all = X_all.to(dtype=torch.float32)
        B, N, T, F = X_all.shape
        S = int(n_samples)

        if N != len(self.node_order):
            raise ValueError(
                f"Input node dim N={N} does not match len(node_order)={len(self.node_order)}"
            )

        node_samples = []
        for j, node in enumerate(self.node_order):
            model = self.models[str(node)]
            node_dev = next(model.parameters()).device

            X_j = X_all[:, j, :, :].to(
                device=node_dev,
                dtype=torch.float32,
                non_blocking=True
            )

            if isinstance(z_temp, dict):
                zt = float(z_temp.get(node, z_temp.get(str(node), 1.0)))
            else:
                zt = float(z_temp)

            if mode == "ddpm":
                y_s_norm = ddpm_sample_parallel_chunked_single(
                    model,
                    X_j,
                    n_samples=S,
                    sample_chunk=sample_chunk,
                    shared_noise=shared_noise,
                    grad_enabled=grad_enabled,
                    cpu_offload=cpu_offload,
                    trunc_steps=trunc_steps,
                    z_temp=zt,
                    force_fp32=True,
                )
            elif mode == "ddim":
                y_s_norm = ddim_sample_parallel_chunked_single(
                    model,
                    X_j,
                    n_samples=S,
                    n_steps=n_steps,
                    eta=eta,
                    sample_chunk=sample_chunk,
                    shared_noise=shared_noise,
                    grad_enabled=grad_enabled,
                    cpu_offload=cpu_offload,
                    trunc_steps=trunc_steps,
                    z_temp=zt,
                    force_fp32=True,
                )
            else:
                raise ValueError(f"Unknown mode: {mode}")

            if y_s_norm.ndim == 4:
                if y_s_norm.shape[-1] != 1:
                    raise ValueError(
                        f"Expect last dim=1, got {tuple(y_s_norm.shape)} for node={node}."
                    )
                y_s_norm = y_s_norm[..., 0]

            if y_s_norm.shape != (S, B, T):
                raise ValueError(
                    f"Expect y_s_norm shape {(S, B, T)} but got {tuple(y_s_norm.shape)} for node={node}"
                )

            y_s_norm = y_s_norm.to(
                self.output_device,
                dtype=torch.float32,
                non_blocking=True
            )
            node_samples.append(y_s_norm)

            del y_s_norm, X_j
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        YS = torch.stack(node_samples, dim=2).float()  # [S,B,N,24]

        if not return_real_scale:
            return YS

        mean = self.y_mean_N11.to(YS.device, dtype=torch.float32).view(1, 1, -1, 1)
        scale = self.y_scale_N11.to(YS.device, dtype=torch.float32).view(1, 1, -1, 1)

        return YS * scale + mean

    def forward(self, X_all, **kwargs):
        return self.sample(X_all, **kwargs)



def _build_predictor_kwargs(args, *, training: bool):
    sample_chunk = (
        int(getattr(args, "train_sample_chunk", getattr(args, "sample_chunk", 1)))
        if training else
        int(getattr(args, "test_sample_chunk", getattr(args, "sample_chunk", 10)))
    )

    return dict(
        sample_chunk=sample_chunk,
        cpu_offload=False if training else bool(getattr(args, "cpu_offload", False)),
        trunc_steps=int(getattr(args, "trunc_steps", 1 if training else 0)) if training else 0,
        shared_noise=bool(getattr(args, "shared_noise", True)),
        mode=getattr(args, "diffusion_mode", "ddpm"),
        n_steps=int(getattr(args, "n_steps", 50)),
        eta=float(getattr(args, "eta", 0.0)),
        z_temp=getattr(args, "z_temp", 1.0),   # NEW
    )

def DFL_train(
    dfl, train_dataset, args, problem_mode="saa",
    train_mode="dfl", filter_kwargs=None, lambda_div=1e5,
):
    dfl.train()
    if hasattr(dfl, "predictor") and hasattr(dfl.predictor, "train"):
        dfl.predictor.train()

    # ---- 统一 train batch size（对齐参考：train_batch_size -> dfl_batch_size -> batch_size）----
    train_bs = int(
        getattr(
            args,
            "train_batch_size",
            getattr(args, "dfl_batch_size", getattr(args, "batch_size", 8)),
        )
    )
    args.train_batch_size = train_bs

    eval_mode = str(getattr(args, "eval_mode", "soft")).lower()
    avoid_rand_duplicate = bool(getattr(args, "avoid_rand_duplicate", False))
    if getattr(dfl, "scenario_filter", None) is not None:
        if hasattr(dfl.scenario_filter, "eval_mode"):
            dfl.scenario_filter.eval_mode = eval_mode
        if hasattr(dfl.scenario_filter, "avoid_rand_duplicate"):
            dfl.scenario_filter.avoid_rand_duplicate = avoid_rand_duplicate

    device = _get_dfl_main_device(dfl)

    if filter_kwargs is None:
        filter_kwargs = {
            "tau_gumbel": float(getattr(args, "tau_gumbel", getattr(args, "tau_mix", 1.0))),
            "eps_uniform": float(getattr(args, "eps_uniform", 0.1)),
        }
    else:
        filter_kwargs = dict(filter_kwargs)
        filter_kwargs.setdefault("tau_gumbel", float(getattr(args, "tau_gumbel", getattr(args, "tau_mix", 1.0))))
        filter_kwargs.setdefault("eps_uniform", float(getattr(args, "eps_uniform", 0.1)))

    div_type = str(getattr(args, "div_type", "inner")).lower()
    div_eps = float(getattr(args, "div_eps", 1e-8))

    if str(problem_mode).lower() in ["saa", "so"]:
        Lmin = Lmax = eps_value = None
    else:
        Lmin = args.Lmin
        Lmax = args.Lmax
        eps_value = args.eps_value

    scenario_filter = getattr(dfl, "scenario_filter", None)

    has_learnable_filter = (
        scenario_filter is not None and
        any(p.requires_grad for p in scenario_filter.parameters())
    )

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
            [
                {"params": optim_params_filter, "lr": filter_lr, "name": "filter"},
            ],
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
            param_groups.append({
                "params": optim_params_filter,
                "lr": filter_lr,
                "name": "filter",
            })
        if len(optim_params_predictor) > 0:
            param_groups.append({
                "params": optim_params_predictor,
                "lr": predictor_lr,
                "name": "predictor",
            })

        if len(param_groups) == 0:
            raise ValueError("No trainable parameters found in DFL_train.")

        optim = torch.optim.Adam(param_groups)

    loader = DataLoader(
        train_dataset,
        batch_size=train_bs,
        shuffle=True,
        drop_last=False,
    )

    pred_kwargs = _build_predictor_kwargs(args, training=True)
    epochs = int(getattr(args, "epochs", 1))

    lr_decay = float(getattr(args, "lr_decay", 1.0))
    filter_lr_decay = float(getattr(args, "filter_lr_decay", lr_decay))
    dfl_lr_decay = float(getattr(args, "dfl_lr_decay", lr_decay))

    print("train_mode =", train_mode)
    print("scenario_filter type =", type(scenario_filter).__name__ if scenario_filter is not None else None)
    print("has_learnable_filter =", has_learnable_filter)
    print("train_batch_size =", train_bs)
    print("pred_kwargs[z_temp] =", pred_kwargs.get("z_temp", None))
    print("lr_decay =", lr_decay, "filter_lr_decay =", filter_lr_decay, "dfl_lr_decay =", dfl_lr_decay)
    print("num filter params =", sum(p.numel() for p in optim_params_filter) if len(optim_params_filter) > 0 else 0)
    print("num predictor params =", sum(p.numel() for p in optim_params_predictor) if len(optim_params_predictor) > 0 else 0)
    print("===== optimizer param groups =====")
    for i, pg in enumerate(optim.param_groups):
        n_params = sum(p.numel() for p in pg["params"])
        print(f"group {i}: name={pg.get('name', 'unknown')}, lr={pg['lr']}, n_params={n_params}")
    print("==================================")

    train_logs = []

    for ep in tqdm(range(epochs), desc=f"Train ({train_mode})", leave=True):
        epoch_task_loss, epoch_div_loss, samples_cnt = 0.0, 0.0, 0
        pbar = tqdm(loader, desc=f"Ep {ep+1}/{epochs}", leave=False)

        for X_all, y_true in pbar:
            X_all = X_all.to(device=device, dtype=torch.float32)
            y_true = y_true.to(device=device, dtype=torch.float32)

            optim.zero_grad(set_to_none=True)

            kwargs = dict(
                solver=getattr(args, "solver", None),
                return_aux=False,
                return_filter_aux=True,
                predictor_n_samples=int(
                    getattr(
                        args,
                        "predictor_n_samples",
                        getattr(args, "S_full", getattr(args, "N_scen", 50)),
                    )
                ),
                predictor_kwargs=pred_kwargs,
                filter_kwargs=filter_kwargs,
            )

            if str(problem_mode).lower() == "dro":
                if Lmin is None or Lmax is None or eps_value is None:
                    raise ValueError("mode='dro' requires Lmin, Lmax, eps_value")
                dtype = getattr(dfl, "optnet_dtype", torch.float64)
                kwargs.update(
                    hourly_load_min_sys=torch.as_tensor(Lmin, device=device, dtype=dtype),
                    hourly_load_max_sys=torch.as_tensor(Lmax, device=device, dtype=dtype),
                    eps=torch.as_tensor(eps_value, device=device, dtype=dtype),
                )

            out = dfl(X_all, y_true, **kwargs)
            if isinstance(out, (tuple, list)):
                loss_vec = out[0]
                aux_filter = out[-1]
            else:
                loss_vec = out
                aux_filter = None

            task_loss_val = loss_vec.mean()
            div_loss_val = torch.tensor(0.0, device=device)

            if aux_filter is not None and "p" in aux_filter and lambda_div > 0:
                p = aux_filter["p"].clamp_min(div_eps)
                p = p / p.sum(dim=-1, keepdim=True).clamp_min(div_eps)
                Bc, Kc, _ = p.shape

                if div_type == "inner" and Kc > 1:
                    M = torch.bmm(p, p.transpose(1, 2))
                    eye = torch.eye(Kc, device=device, dtype=torch.bool).unsqueeze(0).expand(Bc, -1, -1)
                    div_loss_val = (M[~eye] ** 2).mean()

                elif div_type == "kl" and Kc > 1:
                    pi, pj = p.unsqueeze(2), p.unsqueeze(1)
                    kl_ij = (pi * (pi.log() - pj.log())).sum(dim=-1)
                    kl_ji = (pj * (pj.log() - pi.log())).sum(dim=-1)
                    skl = 0.5 * (kl_ij + kl_ji)
                    eye = torch.eye(Kc, device=device, dtype=torch.bool).unsqueeze(0).expand(Bc, -1, -1)
                    div_loss_val = -skl[~eye].mean()

                elif div_type == "entropy":
                    H = -(p * p.log()).sum(dim=-1).mean()
                    div_loss_val = -H

            loss = task_loss_val + float(lambda_div) * div_loss_val
            loss.backward()

            if len(optim_params_filter) > 0:
                torch.nn.utils.clip_grad_norm_(optim_params_filter, 1.0)
            if len(optim_params_predictor) > 0:
                torch.nn.utils.clip_grad_norm_(optim_params_predictor, 1.0)

            optim.step()

            Bsz = X_all.shape[0]
            epoch_task_loss += float(task_loss_val.detach().cpu()) * Bsz
            epoch_div_loss += float(div_loss_val.detach().cpu()) * Bsz
            samples_cnt += Bsz

            pbar.set_postfix(
                Task=float(task_loss_val.detach().cpu()),
                Div=float(div_loss_val.detach().cpu()),
                Type=div_type,
                LR=[pg["lr"] for pg in optim.param_groups],
            )

            for pg in optim.param_groups:
                if pg.get("name") == "filter":
                    pg["lr"] *= filter_lr_decay
                elif pg.get("name") == "predictor":
                    pg["lr"] *= dfl_lr_decay

        print(f"[Epoch {ep+1}] lr after decay:")
        for i, pg in enumerate(optim.param_groups):
            print(f" group {i}: name={pg.get('name', 'unknown')}, lr={pg['lr']}")

        train_logs.append({
            "epoch": int(ep + 1),
            "task": epoch_task_loss / max(samples_cnt, 1),
            "div": epoch_div_loss / max(samples_cnt, 1),
            "div_type": div_type,
            "train_batch_size": train_bs,
            "lr": {pg.get("name", f"group_{i}"): pg["lr"] for i, pg in enumerate(optim.param_groups)},
            "z_temp": copy.deepcopy(pred_kwargs.get("z_temp", None)),
        })

    return dfl, train_logs

@torch.no_grad()
def DFL_test(
    dfl,
    test_dataset,
    args,
    problem_mode="saa",
    filter_kwargs=None,
    return_filter_aux=False,
):
    dfl.eval()
    if hasattr(dfl, "predictor") and hasattr(dfl.predictor, "eval"):
        dfl.predictor.eval()

    eval_mode = str(getattr(args, "eval_mode", "soft")).lower()
    avoid_rand_duplicate = bool(getattr(args, "avoid_rand_duplicate", False))
    if getattr(dfl, "scenario_filter", None) is not None:
        if hasattr(dfl.scenario_filter, "eval_mode"):
            dfl.scenario_filter.eval_mode = eval_mode
        if hasattr(dfl.scenario_filter, "avoid_rand_duplicate"):
            dfl.scenario_filter.avoid_rand_duplicate = avoid_rand_duplicate

    # ---- 统一 test batch size（对齐参考：test_batch_size -> batch_size）----
    test_bs = int(getattr(args, "test_batch_size", getattr(args, "batch_size", 8)))
    args.test_batch_size = test_bs

    print(f"[DFL_test] eval_mode={eval_mode}, avoid_rand_duplicate={avoid_rand_duplicate}, test_batch_size={test_bs}")

    device = _get_dfl_main_device(dfl)

    if str(problem_mode).lower() not in ["saa", "so"]:
        Lmin = args.Lmin
        Lmax = args.Lmax
        eps_value = args.eps_value
    else:
        Lmin = Lmax = eps_value = None

    if filter_kwargs is None:
        filter_kwargs = {
            "tau_gumbel": float(getattr(args, "tau_gumbel", getattr(args, "tau_mix", 1.0))),
            "eps_uniform": float(getattr(args, "eps_uniform", 0.0)),
        }
    else:
        filter_kwargs = dict(filter_kwargs)
        filter_kwargs.setdefault("tau_gumbel", float(getattr(args, "tau_gumbel", getattr(args, "tau_mix", 1.0))))
        filter_kwargs.setdefault("eps_uniform", float(getattr(args, "eps_uniform", 0.0)))

    loader = DataLoader(
        test_dataset,
        batch_size=test_bs,
        shuffle=False,
        drop_last=False,
    )

    pred_kwargs = _build_predictor_kwargs(args, training=False)
    print("[DFL_test] pred_kwargs[z_temp] =", pred_kwargs.get("z_temp", None))

    losses = []
    all_filter_aux = [] if return_filter_aux else None
    losses_flat = []

    pbar = tqdm(loader, desc="Test", leave=True)
    for X_all, y_true in pbar:
        X_all = X_all.to(device=device, dtype=torch.float32)
        y_true = y_true.to(device=device, dtype=torch.float32)

        kwargs = dict(
            solver=getattr(args, "solver", None),
            return_aux=False,
            return_filter_aux=bool(return_filter_aux),
            predictor_n_samples=int(
                getattr(
                    args,
                    "predictor_n_samples",
                    getattr(args, "S_full", getattr(args, "N_scen", 50)),
                )
            ),
            predictor_kwargs=pred_kwargs,
            filter_kwargs=filter_kwargs,
        )

        if str(problem_mode).lower() == "dro":
            if Lmin is None or Lmax is None or eps_value is None:
                raise ValueError("mode='dro' requires Lmin, Lmax, eps_value")
            dtype = getattr(dfl, "optnet_dtype", torch.float64)
            kwargs.update(
                hourly_load_min_sys=torch.as_tensor(Lmin, device=device, dtype=dtype),
                hourly_load_max_sys=torch.as_tensor(Lmax, device=device, dtype=dtype),
                eps=torch.as_tensor(eps_value, device=device, dtype=dtype),
            )

        set_seed(0)
        out = dfl(X_all, y_true, **kwargs)

        if return_filter_aux:
            if isinstance(out, (tuple, list)):
                loss_vec = out[0]
                aux_filter = out[-1]
            else:
                loss_vec = out
                aux_filter = None
            all_filter_aux.append(aux_filter)
        else:
            loss_vec = out[0] if isinstance(out, (tuple, list)) else out

        batch_mean = float(loss_vec.mean().detach().cpu())
        losses.append(loss_vec.detach().cpu())
        losses_flat += loss_vec.detach().cpu().tolist()
        pbar.set_postfix(loss=batch_mean, avg=float(np.mean(losses_flat)))

    loss_all = torch.cat(losses, dim=0)
    if return_filter_aux:
        return loss_all, all_filter_aux
    return loss_all

def run_DFL_diffusion_separate(
    args,
    problem_mode,
    optimization_mode,
    data_path,
    target_nodes,
    models_s,
    pack_data_s,
    device,
    seed=0,
    predictor_dtype=torch.float32,
    optnet_dtype=torch.float64,
    eval_splits=("test",),
    eval_flags=(True, True, True, True, True),
    node_device_map=None,
    optnet_device=None,
    stage2_artifact=None,
):
    import copy
    import time
    import numpy as np
    import torch

    def is_so(x):
        return str(x).lower() in {"so", "saa"}

    def is_multi(x):
        return str(x).lower() in {"multi", "multinode"}

    eval_splits = tuple(s.lower() for s in (eval_splits or ("test",)))
    if not all(s in ("train", "test") for s in eval_splits):
        raise ValueError(f"eval_splits must be subset of ('train','test'), got {eval_splits}")

    if eval_flags is None:
        eval_flags = (True, True, True, True, True)
    if len(eval_flags) != 5:
        raise ValueError(f"eval_flags must be length-5, got {eval_flags}")
    eval_det_before, eval_random_before, eval_stage2_after, eval_stage3_before, eval_stage3_after = map(bool, eval_flags)

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

    print("\n[ScenarioFilter eval config]")
    print("eval_mode =", args.eval_mode)
    print("avoid_rand_duplicate =", args.avoid_rand_duplicate)
    print("train_batch_size =", args.train_batch_size)
    print("test_batch_size =", args.test_batch_size)

    reused_stage2 = stage2_artifact is not None

    if optnet_device is None:
        optnet_device = device
    optnet_device = torch.device(optnet_device)

    if node_device_map is None:
        node_device_map = {node: str(optnet_device) for node in target_nodes}

    multi = is_multi(optimization_mode)

    if multi:
        SO_Manager, DRO_Manager = IEEE14_Reserve_SO_Manager_MultiNode, IEEE14_Reserve_DRO_Manager_MultiNode
        DET_Manager = IEEE14_Reserve_Deterministic_DA_Manager_MultiNode

        SAA_DA_OptNet, DRO_DA_OptNet = MultiNode_Reserve_SAA_DA_OptNet, MultiNode_Reserve_DRO_DA_OptNet
        DET_DA_OptNet = MultiNode_Reserve_Deterministic_DA_OptNet

        RT_OptNet = MultiNode_Reserve_RT_OptNet

        DFL_SO_Class, DFL_DRO_Class = DFL_model_diffusion_MultiNode, DFL_model_diffusion_DRO_MultiNode
        DET_Class = DFL_model_diffusion_Deterministic_MultiNode
    else:
        SO_Manager, DRO_Manager = IEEE14_Reserve_SO_Manager_SingleNode, IEEE14_Reserve_DRO_Manager_SingleNode
        DET_Manager = IEEE14_Reserve_Deterministic_DA_Manager_SingleNode

        SAA_DA_OptNet = SingleNode_Reserve_SAA_DA_OptNet
        DRO_DA_OptNet = SingleNode_Reserve_DRO_DA_OptNet
        DET_DA_OptNet = SingleNode_Reserve_Deterministic_DA_OptNet

        RT_OptNet = SingleNode_Reserve_RT_OptNet

        DFL_SO_Class, DFL_DRO_Class = DFL_model_diffusion_SingleNode, DFL_model_diffusion_DRO_SingleNode
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

    stage0_det_eval = {}
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
            return Combined_dataset_diffusion(
                data_path=data_path,
                target_nodes=target_nodes,
                flag=flag,
                train_length=8760,
                val_ratio=0.2,
                seed=42,
            )

        train_data = make_dataset("train")
        test_data = make_dataset("test")

        best_z = _resolve_separate_diffusion_z_temp(
            z_temp=None,
            pack_data_s=pack_data_s,
            target_nodes=target_nodes,
            args=args,
            default=1.0,
        )
        args.z_temp = copy.deepcopy(best_z)
        print("[run_DFL_diffusion_separate] using z_temp =", args.z_temp)

        def build_predictor(node_models, scaler_y_map):
            return Multi_diffusion_predictor(
                node_models=copy.deepcopy(node_models),
                scaler_y_map=scaler_y_map,
                node_order=target_nodes,
                dtype=predictor_dtype,
                node_device_map=node_device_map,
                output_device=optnet_device,
                default_z_temp=args.z_temp,
            )

        def make_filter():
            set_seed(seed)
            print('=========set_seed========')
            return ScenarioFilter(
                args=args,
                prob_type="multi" if multi else "single",
                T=24,
                N_nodes=11,
                K=K,
                K_rand=K_rand,
                hidden=128,
            ).to(optnet_device)

        predictor_before = build_predictor(models_s, train_data.scaler_y_map)
        predictor_train = build_predictor(models_s, train_data.scaler_y_map)

        if is_so(problem_mode):
            mgr_local_before = SO_Manager(args)
            mgr_local_train = SO_Manager(args)

            optnet_DA_before = SAA_DA_OptNet(
                mgr=mgr_local_before, N_scen=args.N_scen, T=24, dtype=optnet_dtype
            ).to(optnet_device)
            optnet_DA = SAA_DA_OptNet(
                mgr=mgr_local_train, N_scen=args.N_scen, T=24, dtype=optnet_dtype
            ).to(optnet_device)

            DFLClass = DFL_SO_Class
            mode_str = "SO"
        else:
            mgr_local_before = DRO_Manager(args)
            mgr_local_train = DRO_Manager(args)

            optnet_DA_before = DRO_DA_OptNet(
                mgr=mgr_local_before, N_scen=args.N_scen, T=24, dtype=optnet_dtype
            ).to(optnet_device)
            optnet_DA = DRO_DA_OptNet(
                mgr=mgr_local_train, N_scen=args.N_scen, T=24, dtype=optnet_dtype
            ).to(optnet_device)

            DFLClass = DFL_DRO_Class
            mode_str = "DRO"

        if (not is_so(problem_mode)) and multi:
            if not hasattr(args, "Lmin") or not hasattr(args, "Lmax"):
                raise ValueError("DRO multi-node needs args.Lmin / args.Lmax (per-node bounds).")

            Lmin = args.Lmin
            Lmax = args.Lmax

            def _ndim(x):
                return x.ndim if hasattr(x, "ndim") else np.asarray(x).ndim

            def _shape(x):
                return x.shape if hasattr(x, "shape") else np.asarray(x).shape

            if _ndim(Lmin) == 2 and _shape(Lmin)[0] == 11:
                args.Lmin = mgr_local_train.map_11load_to_14bus(Lmin)
            if _ndim(Lmax) == 2 and _shape(Lmax)[0] == 11:
                args.Lmax = mgr_local_train.map_11load_to_14bus(Lmax)

            if _ndim(args.Lmin) == 3 and _shape(args.Lmin)[1] == 11:
                args.Lmin = np.stack([mgr_local_train.map_11load_to_14bus(x) for x in args.Lmin], axis=0)
            if _ndim(args.Lmax) == 3 and _shape(args.Lmax)[1] == 11:
                args.Lmax = np.stack([mgr_local_train.map_11load_to_14bus(x) for x in args.Lmax], axis=0)

            if _ndim(args.Lmin) == 1:
                args.Lmin = np.tile(np.asarray(args.Lmin)[None, :], (14, 1))
            if _ndim(args.Lmax) == 1:
                args.Lmax = np.tile(np.asarray(args.Lmax)[None, :], (14, 1))

            if not hasattr(args, "eps_value"):
                raise ValueError("DRO needs args.eps_value (scalar).")

        mgr_det = DET_Manager(args)
        optnet_DA_det = DET_DA_OptNet(mgr=mgr_det, T=24, dtype=optnet_dtype).to(optnet_device)

        optnet_RT_before = RT_OptNet(mgr=mgr_local_before, T=24, dtype=optnet_dtype).to(optnet_device)
        optnet_RT = RT_OptNet(mgr=mgr_local_train, T=24, dtype=optnet_dtype).to(optnet_device)

        filter_module = make_filter()

        dfl_det_before = DET_Class(
            mgr=mgr_det,
            optnet_DA=optnet_DA_det,
            optnet_RT=optnet_RT_before,
            predictor=copy.deepcopy(predictor_before),
            scenario_filter=copy.deepcopy(filter_module),
            n_scen=args.N_scen,
            clamp_min=float(getattr(args, "clamp_min", 0.0)),
            solver=getattr(args, "solver", "ECOS"),
        ).to(optnet_device)

        dfl_before = DFLClass(
            mgr=mgr_local_before,
            optnet_DA=optnet_DA_before,
            optnet_RT=optnet_RT_before,
            predictor=predictor_before,
            scenario_filter=copy.deepcopy(filter_module),
            n_scen=args.N_scen,
            clamp_min=float(getattr(args, "clamp_min", 0.0)),
            solver=getattr(args, "solver", "ECOS"),
        ).to(optnet_device)

        dfl = DFLClass(
            mgr=mgr_local_train,
            optnet_DA=optnet_DA,
            optnet_RT=optnet_RT,
            predictor=predictor_train,
            scenario_filter=filter_module,
            n_scen=args.N_scen,
            clamp_min=float(getattr(args, "clamp_min", 0.0)),
            solver=getattr(args, "solver", "ECOS"),
        ).to(optnet_device)

        def rebuild_model_like(src_model, kind: str, shell: str = "train"):
            if src_model is None:
                return None

            new_pred = build_predictor(models_s, train_data.scaler_y_map)
            new_pred.load_state_dict(copy.deepcopy(src_model.predictor.state_dict()))

            new_filter = make_filter()
            if hasattr(src_model, "scenario_filter") and src_model.scenario_filter is not None:
                try:
                    new_filter.load_state_dict(copy.deepcopy(src_model.scenario_filter.state_dict()))
                except Exception:
                    pass

            if kind == "dfl":
                if shell == "train":
                    _mgr, _DA, _RT = mgr_local_train, optnet_DA, optnet_RT
                elif shell == "before":
                    _mgr, _DA, _RT = mgr_local_before, optnet_DA_before, optnet_RT_before
                else:
                    raise ValueError(f"unknown shell={shell}")

                return DFLClass(
                    mgr=_mgr,
                    optnet_DA=_DA,
                    optnet_RT=_RT,
                    predictor=new_pred,
                    scenario_filter=new_filter,
                    n_scen=args.N_scen,
                    clamp_min=float(getattr(args, "clamp_min", 0.0)),
                    solver=getattr(args, "solver", "ECOS"),
                ).to(optnet_device)

            elif kind == "det":
                return DET_Class(
                    mgr=mgr_det,
                    optnet_DA=optnet_DA_det,
                    optnet_RT=optnet_RT_before,
                    predictor=new_pred,
                    scenario_filter=new_filter,
                    n_scen=args.N_scen,
                    clamp_min=float(getattr(args, "clamp_min", 0.0)),
                    solver=getattr(args, "solver", "ECOS"),
                ).to(optnet_device)

            else:
                raise ValueError(f"unknown kind={kind}")

        def eval_on_splits(model, stage_tag):
            out = {}
            if "test" in eval_splits:
                set_seed(seed)
                out[f"test_losses_{stage_tag}"] = DFL_test(model, test_data, args, problem_mode=problem_mode)
            if "train" in eval_splits:
                set_seed(seed)
                out[f"train_losses_{stage_tag}"] = DFL_test(model, train_data, args, problem_mode=problem_mode)
            return out

        if eval_det_before:
            stage0_det_eval = eval_on_splits(dfl_det_before, "deterministic_before")
            if "test" in eval_splits:
                print_mean_loss(
                    "Deterministic baseline (before DFL training)",
                    stage0_det_eval,
                    "test_losses_deterministic_before",
                )

        if eval_random_before:
            random_filter = RandomScenarioSelector(n_scen=int(args.N_scen)).to(optnet_device)
            orig_filter = dfl_before.scenario_filter
            dfl_before.scenario_filter = random_filter
            stage1_eval = eval_on_splits(dfl_before, "stage1_after")
            dfl_before.scenario_filter = orig_filter

            if "test" in eval_splits:
                print_mean_loss(
                    "Random filter baseline (before DFL training)",
                    stage1_eval,
                    "test_losses_stage1_after",
                )

        args.epochs = int(getattr(args, "dfl_epochs", getattr(args, "epochs", 2)))
        args.dfl_lr = float(getattr(args, "dfl_lr", 1e-5))
        args.filter_lr = float(getattr(args, "filter_lr", 1e-3))

        if run_stage2:
            print(
                f"\n ---> Stage A: train DFL with RANDOM selector "
                f"(epochs={args.epochs}, dfl_lr={args.dfl_lr}, train_bs={args.train_batch_size})"
            )

            dfl.scenario_filter = RandomScenarioSelector(n_scen=int(args.N_scen)).to(optnet_device)
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

        dfl_after_stage2_snapshot = rebuild_model_like(dfl, kind="dfl", shell="train")
        dfl_det_before_snapshot = rebuild_model_like(dfl_det_before, kind="det") if dfl_det_before is not None else None
        dfl_before_snapshot = rebuild_model_like(dfl_before, kind="dfl", shell="before") if dfl_before is not None else None

    else:
        print("\n ---> Reusing passed stage2_artifact, skip Stage A and jump to Stage B")

        train_data = stage2_artifact["train_data"]
        test_data = stage2_artifact["test_data"]
        mode_str = stage2_artifact["mode_str"]

        best_z = copy.deepcopy(stage2_artifact.get("z_temp", stage2_artifact.get("best_z", None)))
        if best_z is None:
            best_z = _resolve_separate_diffusion_z_temp(
                z_temp=None,
                pack_data_s=pack_data_s,
                target_nodes=target_nodes,
                args=args,
                default=1.0,
            )
        args.z_temp = copy.deepcopy(best_z)
        print("[run_DFL_diffusion_separate] using z_temp =", args.z_temp)

        stage0_det_eval = copy.deepcopy(stage2_artifact.get("stage0_det_eval", {}))
        stage1_eval = copy.deepcopy(stage2_artifact.get("stage1_eval", {}))
        stage2_eval = copy.deepcopy(stage2_artifact.get("stage2_eval", {}))

        train_logs_stage2 = copy.deepcopy(stage2_artifact.get("train_logs_stage2", []))
        time_stage2 = float(stage2_artifact.get("time_stage2_sec", 0.0))

        if is_so(problem_mode):
            mgr_local_train = SO_Manager(args)
            optnet_DA = SAA_DA_OptNet(mgr=mgr_local_train, N_scen=args.N_scen, T=24, dtype=optnet_dtype).to(optnet_device)
            DFLClass = DFL_SO_Class
        else:
            mgr_local_train = DRO_Manager(args)
            optnet_DA = DRO_DA_OptNet(mgr=mgr_local_train, N_scen=args.N_scen, T=24, dtype=optnet_dtype).to(optnet_device)
            DFLClass = DFL_DRO_Class

        if (not is_so(problem_mode)) and multi:
            if not hasattr(args, "Lmin") or not hasattr(args, "Lmax"):
                raise ValueError("DRO multi-node needs args.Lmin / args.Lmax (per-node bounds).")

            Lmin = args.Lmin
            Lmax = args.Lmax

            def _ndim(x):
                return x.ndim if hasattr(x, "ndim") else np.asarray(x).ndim

            def _shape(x):
                return x.shape if hasattr(x, "shape") else np.asarray(x).shape

            if _ndim(Lmin) == 2 and _shape(Lmin)[0] == 11:
                args.Lmin = mgr_local_train.map_11load_to_14bus(Lmin)
            if _ndim(Lmax) == 2 and _shape(Lmax)[0] == 11:
                args.Lmax = mgr_local_train.map_11load_to_14bus(Lmax)

            if _ndim(args.Lmin) == 3 and _shape(args.Lmin)[1] == 11:
                args.Lmin = np.stack([mgr_local_train.map_11load_to_14bus(x) for x in args.Lmin], axis=0)
            if _ndim(args.Lmax) == 3 and _shape(args.Lmax)[1] == 11:
                args.Lmax = np.stack([mgr_local_train.map_11load_to_14bus(x) for x in args.Lmax], axis=0)

            if _ndim(args.Lmin) == 1:
                args.Lmin = np.tile(np.asarray(args.Lmin)[None, :], (14, 1))
            if _ndim(args.Lmax) == 1:
                args.Lmax = np.tile(np.asarray(args.Lmax)[None, :], (14, 1))

            if not hasattr(args, "eps_value"):
                raise ValueError("DRO needs args.eps_value (scalar).")

        optnet_RT = RT_OptNet(mgr=mgr_local_train, T=24, dtype=optnet_dtype).to(optnet_device)

        def build_predictor(node_models, scaler_y_map):
            return Multi_diffusion_predictor(
                node_models=copy.deepcopy(node_models),
                scaler_y_map=scaler_y_map,
                node_order=target_nodes,
                dtype=predictor_dtype,
                node_device_map=node_device_map,
                output_device=optnet_device,
                default_z_temp=args.z_temp,
            )

        def make_filter():
            set_seed(seed)
            print('=========set_seed========')
            return ScenarioFilter(
                args=args,
                prob_type="multi" if multi else "single",
                T=24,
                N_nodes=11,
                K=K,
                K_rand=K_rand,
                hidden=128,
            ).to(optnet_device)

        def rebuild_model_like(src_model, kind: str, shell: str = "train"):
            if src_model is None:
                return None
            if kind != "dfl":
                raise ValueError("reuse-branch rebuild_model_like only supports kind='dfl' (for stage3 training).")

            new_pred = build_predictor(models_s, train_data.scaler_y_map)
            new_pred.load_state_dict(copy.deepcopy(src_model.predictor.state_dict()))

            new_filter = make_filter()
            if hasattr(src_model, "scenario_filter") and src_model.scenario_filter is not None:
                try:
                    new_filter.load_state_dict(copy.deepcopy(src_model.scenario_filter.state_dict()))
                except Exception:
                    pass

            return DFLClass(
                mgr=mgr_local_train,
                optnet_DA=optnet_DA,
                optnet_RT=optnet_RT,
                predictor=new_pred,
                scenario_filter=new_filter,
                n_scen=args.N_scen,
                clamp_min=float(getattr(args, "clamp_min", 0.0)),
                solver=getattr(args, "solver", "ECOS"),
            ).to(optnet_device)

        stage2_model = stage2_artifact["dfl_after_stage2"]
        dfl = rebuild_model_like(stage2_model, kind="dfl", shell="train")

        dfl_after_stage2_snapshot = stage2_model
        dfl_det_before_snapshot = stage2_artifact.get("dfl_det_before", None)
        dfl_before_snapshot = stage2_artifact.get("dfl_before", None)

        def eval_on_splits(model, stage_tag):
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
            f"(epochs={args.epochs}, lr={args.lr}, train_bs={args.train_batch_size})"
        )

        set_seed(seed)
        dfl.scenario_filter = ScenarioFilter(
            args=args,
            prob_type="multi" if multi else "single",
            T=24,
            N_nodes=11,
            K=K,
            K_rand=K_rand,
            hidden=128,
        ).to(optnet_device)

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
            lambda_div=float(getattr(args, "lambda_div_stage_3", getattr(args, "lambda_div", 1e5))),
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
        "best_z": copy.deepcopy(args.z_temp),
        "z_temp": copy.deepcopy(args.z_temp),
        "dfl_after_stage2": dfl_after_stage2_snapshot,
        "dfl_det_before": dfl_det_before_snapshot,
        "dfl_before": dfl_before_snapshot,
        "stage0_det_eval": copy.deepcopy(stage0_det_eval),
        "stage1_eval": copy.deepcopy(stage1_eval),
        "stage2_eval": copy.deepcopy(stage2_eval),
        "train_logs_stage2": copy.deepcopy(train_logs_stage2),
        "time_stage2_sec": float(time_stage2),
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
        "z_temp": copy.deepcopy(args.z_temp),
        "train_batch_size_used": int(args.train_batch_size),
        "test_batch_size_used": int(args.test_batch_size),
        "N_scen": int(args.N_scen),
        "S_full": int(args.S_full),
        "K_rand": int(K_rand),
        "predictor_dtype": str(predictor_dtype),
        "optnet_dtype": str(optnet_dtype),
        "node_device_map": {k: str(v) for k, v in node_device_map.items()},
        "optnet_device": str(optnet_device),
        "eval_mode": str(getattr(args, "eval_mode", "soft")).lower(),
        "avoid_rand_duplicate": bool(getattr(args, "avoid_rand_duplicate", False)),
        "stage2_artifact": stage2_artifact_out,
    }

    result.update(stage0_det_eval)
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

    if "test_losses_stage1_after" in result:
        result["test_losses_before"] = result["test_losses_stage1_after"]
    if "test_losses_stage3_after" in result:
        result["test_losses_after"] = result["test_losses_stage3_after"]
    if "train_losses_stage1_after" in result:
        result["train_losses_before"] = result["train_losses_stage1_after"]
    if "train_losses_stage3_after" in result:
        result["train_losses_after"] = result["train_losses_stage3_after"]

    return result

def compare_scenario_filters_with_stage3_learned_diff(
    base_result,
    args,
    problem_mode,
    optimization_mode,
    models_s,
    target_nodes,
    device,
    seed=0,
    predictor_dtype=torch.float32,
    optnet_dtype=torch.float64,
    eval_splits=("test",),
    method_names=("learned", "random", "kmeans", "kmedoids", "hierarchical"),
    node_device_map=None,
    optnet_device=None,
    verbose=True,
):
    """
    Diffusion版（对齐 VAE compare 的“rebuild_with_specific_filter”策略）：
    - backbone predictor 只来自 stage2（stage2_artifact['dfl_after_stage2'].predictor）
    - learned filter 来自 stage3（base_result['dfl_trained'].scenario_filter）
    - 其他方法只替换 filter
    - 不再 load_state_dict(backbone_dfl.state_dict(), strict=False) 去“拼全量状态”
    - summary_mean 按 split 分别保存，如 {'test': xxx, 'train': xxx}
    """
    import copy
    import torch

    def is_so(x):
        return str(x).lower() in {"so", "saa"}

    def is_multi(x):
        return str(x).lower() in {"multi", "multinode"}

    def _mean_of(x):
        if x is None:
            return None
        if hasattr(x, "detach"):
            return float(x.detach().float().mean().item())
        return float(torch.as_tensor(x).float().mean().item())

    # ---- 统一 train/test batch size（同 run_DFL_diffusion_separate 风格）----
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

    eval_splits = tuple(s.lower() for s in (eval_splits or ("test",)))
    if not all(s in ("train", "test") for s in eval_splits):
        raise ValueError(f"eval_splits must be subset of ('train','test'), got {eval_splits}")

    if optnet_device is None:
        optnet_device = device
    optnet_device = torch.device(optnet_device)

    if node_device_map is None:
        node_device_map = {node: str(optnet_device) for node in target_nodes}

    if "stage2_artifact" not in base_result:
        raise ValueError("base_result does not contain 'stage2_artifact'.")

    stage2_artifact = base_result["stage2_artifact"]
    train_data = stage2_artifact["train_data"]
    test_data = stage2_artifact["test_data"]
    stage2_model = stage2_artifact.get("dfl_after_stage2", None)
    trained_model = base_result.get("dfl_trained", None)

    best_z = copy.deepcopy(stage2_artifact.get("z_temp", stage2_artifact.get("best_z", None)))
    if best_z is None:
        best_z = copy.deepcopy(base_result.get("z_temp", getattr(args, "z_temp", 1.0)))
    args.z_temp = copy.deepcopy(best_z)

    if stage2_model is None:
        raise ValueError("stage2_artifact['dfl_after_stage2'] is missing.")
    if trained_model is None:
        raise ValueError("base_result['dfl_trained'] is missing.")

    learned_filter = copy.deepcopy(trained_model.scenario_filter)

    multi = is_multi(optimization_mode)
    if multi:
        SO_Manager, DRO_Manager = (
            IEEE14_Reserve_SO_Manager_MultiNode,
            IEEE14_Reserve_DRO_Manager_MultiNode,
        )
        SAA_DA_OptNet, DRO_DA_OptNet = (
            MultiNode_Reserve_SAA_DA_OptNet,
            MultiNode_Reserve_DRO_DA_OptNet,
        )
        RT_OptNet = MultiNode_Reserve_RT_OptNet
        DFL_SO_Class, DFL_DRO_Class = (
            DFL_model_diffusion_MultiNode,
            DFL_model_diffusion_DRO_MultiNode,
        )
    else:
        SO_Manager, DRO_Manager = (
            IEEE14_Reserve_SO_Manager_SingleNode,
            IEEE14_Reserve_DRO_Manager_SingleNode,
        )
        SAA_DA_OptNet, DRO_DA_OptNet = (
            SingleNode_Reserve_SAA_DA_OptNet,
            SingleNode_Reserve_DRO_DA_OptNet,
        )
        RT_OptNet = SingleNode_Reserve_RT_OptNet
        DFL_SO_Class, DFL_DRO_Class = (
            DFL_model_diffusion_SingleNode,
            DFL_model_diffusion_DRO_SingleNode,
        )

    compare_result = {
        "backbone_predictor_source": "stage2_artifact['dfl_after_stage2'].predictor",
        "learned_filter_source": "base_result['dfl_trained'].scenario_filter",
        "method_names": list(method_names),
        "eval_splits": list(eval_splits),
        "z_temp": copy.deepcopy(best_z),
        "train_batch_size_used": int(train_bs),
        "test_batch_size_used": int(test_bs),
        "details": {},
        "summary_mean": {},
    }

    if verbose:
        print(f"[compare_diff] train_batch_size = {train_bs}")
        print(f"[compare_diff] test_batch_size = {test_bs}")
        print(f"[compare_diff] eval_mode = {args.eval_mode}")
        print(f"[compare_diff] avoid_rand_duplicate = {args.avoid_rand_duplicate}")
        print(f"[compare_diff] z_temp = {best_z}")
        print(f"[compare_diff] methods = {list(method_names)}")
        print(f"[compare_diff] eval_splits = {list(eval_splits)}")

    def build_predictor_fresh():
        pred = Multi_diffusion_predictor(
            node_models=copy.deepcopy(models_s),
            scaler_y_map=train_data.scaler_y_map,
            node_order=target_nodes,
            dtype=predictor_dtype,
            node_device_map=node_device_map,
            output_device=optnet_device,
            default_z_temp=args.z_temp,
        )
        pred.load_state_dict(copy.deepcopy(stage2_model.predictor.state_dict()))
        return pred

    def rebuild_with_specific_filter(filter_module):
        predictor = build_predictor_fresh()

        if is_so(problem_mode):
            mgr_local = SO_Manager(args)
            optnet_DA = SAA_DA_OptNet(
                mgr=mgr_local,
                N_scen=args.N_scen,
                T=24,
                dtype=optnet_dtype,
            ).to(optnet_device)
            DFLClass = DFL_SO_Class
        else:
            mgr_local = DRO_Manager(args)
            optnet_DA = DRO_DA_OptNet(
                mgr=mgr_local,
                N_scen=args.N_scen,
                T=24,
                dtype=optnet_dtype,
            ).to(optnet_device)
            DFLClass = DFL_DRO_Class

        optnet_RT = RT_OptNet(
            mgr=mgr_local,
            T=24,
            dtype=optnet_dtype,
        ).to(optnet_device)

        model = DFLClass(
            mgr=mgr_local,
            optnet_DA=optnet_DA,
            optnet_RT=optnet_RT,
            predictor=predictor,
            scenario_filter=filter_module.to(optnet_device),
            n_scen=args.N_scen,
            clamp_min=float(getattr(args, "clamp_min", 0.0)),
            solver=getattr(args, "solver", "ECOS"),
        ).to(optnet_device)
        return model

    def _eval_on_splits(model, stage_tag):
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
            print(f"[compare_diff] evaluating method: {name}")

        if name == "learned":
            filter_module = copy.deepcopy(learned_filter)
        elif name == "random":
            filter_module = RandomScenarioSelector(n_scen=int(args.N_scen)).to(optnet_device)
        else:
            filter_module = build_scenario_baseline_filter(name, args, optnet_device)

        if hasattr(filter_module, "eval_mode"):
            filter_module.eval_mode = args.eval_mode
        if hasattr(filter_module, "avoid_rand_duplicate"):
            filter_module.avoid_rand_duplicate = args.avoid_rand_duplicate

        model = rebuild_with_specific_filter(filter_module)
        stage_tag = f"compare_stage2backbone_{name}"
        eval_out = _eval_on_splits(model, stage_tag)

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
            print(f"[compare_diff][{name}]")
            for split in eval_splits:
                if split in mean_info and mean_info[split] is not None:
                    print(f"  {split} mean loss: {mean_info[split]:.6f}")

    return compare_result
