import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from data_loader import *
from combined_data_loader import *
from tqdm import tqdm
import time
from utils import *
from utils_draw import *
from Optimization_single_node import *
from Optimization_multi_node import *
from scenarios_reduce import *

class ANN_quantiles_non_parametric(nn.Module):
    """
    Input:  X [B,F]
    Output: q_hat [B,Q]  for quantiles like [0.05,...,0.95]
    """
    def __init__(self, input_dim, quantiles, hidden=(128,128,128), dropout=0.0):
        super().__init__()
        self.quantiles = torch.tensor(quantiles, dtype=torch.float32)
        h1, h2, h3 = hidden
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, h3),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h3, len(quantiles)),
        )

    def forward(self, x):
        return self.net(x)  # [B,Q]

def pinball_loss_non_parametric(q_hat, y, quantiles):
    """
    q_hat: [B,Q]
    y:     [B]
    """
    q = torch.tensor(quantiles, device=q_hat.device, dtype=q_hat.dtype).view(1, -1)  # [1,Q]
    y = y.view(-1, 1)  # [B,1]
    err = y - q_hat    # [B,Q]
    loss = torch.maximum(q * err, (q - 1.0) * err)
    return loss.mean()


class Runner_non_parametric:
    def __init__(self, train_set, val_set, test_set, quantiles, hidden=(256,256,128), lr=1e-3, device=None):
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        self.quantiles = list(quantiles)

        input_dim = train_set.X.shape[-1]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ANN_quantiles_non_parametric(input_dim=input_dim, quantiles=self.quantiles, hidden=hidden).to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)

    def fit(self, epochs=500, batch_size=256, patience=50, best_path="best.pt", verbose=False):
        train_loader = DataLoader(self.train_set, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(self.val_set, batch_size=batch_size, shuffle=False, drop_last=False)

        best_val = float("inf")
        bad = 0

        for ep in range(1, epochs + 1):
            self.model.train()
            tr_losses = []
            for X, y in train_loader:
                X = X.to(self.device)
                y = y.to(self.device)
                q_hat = self.model(X)
                loss = pinball_loss_non_parametric(q_hat, y, self.quantiles)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                tr_losses.append(float(loss.item()))

            self.model.eval()
            va_losses = []
            with torch.no_grad():
                for X, y in val_loader:
                    X = X.to(self.device)
                    y = y.to(self.device)
                    q_hat = self.model(X)
                    loss = pinball_loss_non_parametric(q_hat, y, self.quantiles)
                    va_losses.append(float(loss.item()))
            val_loss = float(np.mean(va_losses)) if len(va_losses) else float("inf")

            if verbose and (ep == 1 or ep % 10 == 0):
                print(f"Epoch {ep:4d} | train={np.mean(tr_losses):.6f} | val={val_loss:.6f}")

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

    @torch.no_grad()
    def predict_quantiles(self):
        loader = DataLoader(self.test_set, batch_size=4096, shuffle=False, drop_last=False)
        qs, ys = [], []
        self.model.eval()
        for X, y in loader:
            X = X.to(self.device)
            q_hat = self.model(X).cpu()  # [B,Q]
            qs.append(q_hat)
            ys.append(y.cpu())           # [B]
        q_hat = torch.cat(qs, dim=0)     # [T,Q]  hour-level
        y = torch.cat(ys, dim=0)         # [T]
        return q_hat, y

def run_non_parametric_benchmark(
    DataHandler,
    device=None,
    epochs=500,
    batch_size=256,
    target_nodes=None,
    lr=1e-3,
    hidden=(128,128,128),
    patience=50,
    ckpt_dir="../Model/Non_parametric/ckpt_nodes_nonparam",
    verbose=True,
    data_path=None,
    train_length=8760,
    val_ratio=0.2,
    seed=42,
    quantiles=None,
):
    os.makedirs(ckpt_dir, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if target_nodes is None or len(target_nodes) == 0:
        raise ValueError("target_nodes is empty.")
    quantiles = quantiles or [0.05 * i for i in range(1, 20)]

    models_s, handlers_s, pack_data_s = {}, {}, {}

    for node in target_nodes:
        if data_path is None:
            raise ValueError("For this implementation, please pass data_path (CSV).")

        train_set = DataHandler(data_path, node, flag="train", train_length=train_length, val_ratio=val_ratio, seed=seed)
        val_set   = DataHandler(data_path, node, flag="val",   train_length=train_length, val_ratio=val_ratio, seed=seed)
        test_set  = DataHandler(data_path, node, flag="test",  train_length=train_length, val_ratio=val_ratio, seed=seed)

        runner = Runner_non_parametric(
            train_set, val_set, test_set,
            quantiles=quantiles, hidden=hidden, lr=lr, device=device
        )
        best_path = os.path.join(ckpt_dir, f"best_{node}.pt")
        runner.fit(epochs=epochs, batch_size=batch_size, patience=patience, best_path=best_path, verbose=verbose)

        model = runner.model.to(device).eval()
        models_s[node] = model

        # NEW: split handlers
        handlers_s[node] = {
            "train": train_set,
            "val": val_set,
            "test": test_set,
        }

        # NEW: split data
        pack_data_s[node] = {
            "quantiles": quantiles,
            "splits": {
                "train": {"X": train_set.X, "y": train_set.y},  # [T,F], [T]
                "val":   {"X": val_set.X,   "y": val_set.y},
                "test":  {"X": test_set.X,  "y": test_set.y},
            },
        }

    return models_s, handlers_s, pack_data_s


def interp_extrap_1d(u, taus, qvals):
    u = np.asarray(u, dtype=np.float32)
    taus = np.asarray(taus, dtype=np.float32)
    qvals = np.asarray(qvals, dtype=np.float32)

    if taus.ndim != 1 or qvals.ndim != 1:
        raise ValueError("taus and qvals must be 1D arrays")
    if len(taus) != len(qvals):
        raise ValueError("taus and qvals must have same length")
    if len(taus) < 2:
        raise ValueError("Need at least 2 points for interpolation/extrapolation")

    y = np.empty_like(u, dtype=np.float32)

    # left extrapolation
    left_mask = u < taus[0]
    if np.any(left_mask):
        t0, t1 = taus[0], taus[1]
        q0, q1 = qvals[0], qvals[1]
        slope_left = (q1 - q0) / (t1 - t0)
        y[left_mask] = q0 + slope_left * (u[left_mask] - t0)

    # right extrapolation
    right_mask = u > taus[-1]
    if np.any(right_mask):
        t0, t1 = taus[-2], taus[-1]
        q0, q1 = qvals[-2], qvals[-1]
        slope_right = (q1 - q0) / (t1 - t0)
        y[right_mask] = q1 + slope_right * (u[right_mask] - t1)

    # interpolation
    mid_mask = (~left_mask) & (~right_mask)
    if np.any(mid_mask):
        y[mid_mask] = np.interp(u[mid_mask], taus, qvals).astype(np.float32)

    return y

@torch.no_grad()
def sample_window_non_parametric_benchmark(
    models_s,
    handlers_s,
    pack_data_s,
    target_nodes,
    horizon_days=7,
    start_day=0,
    n_samples=50,
    seq_len=24,
    split="test",
):
    node_names = list(target_nodes)
    N = len(node_names)
    if N == 0:
        raise ValueError("target_nodes is empty.")
    split = str(split)

    # 用第一个节点确定长度（hour-level）
    first = node_names[0]
    if "splits" not in pack_data_s[first] or split not in pack_data_s[first]["splits"]:
        raise KeyError(f"split={split} not found in pack_data_s[{first}]['splits']")
    T0 = int(pack_data_s[first]["splits"][split]["X"].shape[0])
    D0 = T0 // int(seq_len)

    if start_day < 0 or start_day >= D0:
        raise ValueError(f"start_day={start_day} out of range, total_{split}_days={D0}")

    actual_days = min(int(horizon_days), D0 - int(start_day))
    L = actual_days * int(seq_len)

    Y_true = np.zeros((N, L), dtype=np.float32)
    Y_pred = np.zeros((int(n_samples), N, L), dtype=np.float32)

    for j, node in enumerate(node_names):
        model = models_s[node]
        handler = handlers_s[node][split]

        X_all = pack_data_s[node]["splits"][split]["X"]   # [T,F]
        y_all = pack_data_s[node]["splits"][split]["y"]   # [T]
        quantiles = np.asarray(pack_data_s[node]["quantiles"], dtype=np.float32)
        Q = len(quantiles)

        if Q < 2:
            raise ValueError(f"Need at least 2 quantiles for interpolation/extrapolation, got Q={Q}")

        device = next(model.parameters()).device
        model.eval()

        t0 = int(start_day) * int(seq_len)
        t1 = t0 + L

        # truth inverse
        y_norm = y_all[t0:t1].detach().cpu().numpy()      # [L]
        y_real = handler.inverse_transform_y(y_norm)      # [L]
        Y_true[j, :] = np.asarray(y_real, dtype=np.float32)

        # predict quantiles
        X = X_all[t0:t1].to(device)                       # [L,F]
        q_hat = model(X).detach().cpu().numpy()           # [L,Q] normalized

        # inverse to real scale
        q_real = handler.scaler_y.inverse_transform(
            q_hat.reshape(-1, 1)
        ).reshape(L, Q)                                   # [L,Q]

        # 确保 quantiles 升序，同时按相同顺序重排 q_real
        order = np.argsort(quantiles)
        taus = quantiles[order]
        q_real = q_real[:, order]

        # 处理 quantile crossing：强制单调不减
        q_real = np.maximum.accumulate(q_real, axis=1)

        # 从 [0,1] 采样，然后做插值/外推
        u = np.random.uniform(0.0, 1.0, size=(int(n_samples), L)).astype(np.float32)  # [S,L]

        s = np.empty((int(n_samples), L), dtype=np.float32)
        for t in range(L):
            s[:, t] = interp_extrap_1d(u[:, t], taus, q_real[t, :])

        Y_pred[:, j, :] = s

    return dict(
        mode="single_window_non_parametric_interp_extrap",
        split=split,
        target_nodes=node_names,
        start_day=int(start_day),
        horizon_days=int(actual_days),
        seq_len=int(seq_len),
        n_samples=int(n_samples),
        Y_true=Y_true,   # [N,L]
        Y_pred=Y_pred,   # [S,N,L]
    )


class Multi_nonparametric_quantile_predictor(nn.Module):
    """
    输入:
      X_all: [B,N,24,F] float32
    输出:
      q_real: [B,N,24,Q] float64 (真实量纲，已 inverse scaler_y)
    """
    def __init__(self, node_models: dict, scaler_y_map: dict, node_order: list,
                 quantiles: list, dtype=torch.float64):
        super().__init__()
        self.node_order = list(node_order)
        self.models = nn.ModuleDict({str(k): node_models[k] for k in self.node_order})
        self.quantiles = sorted(list(quantiles))
        self.Q = len(self.quantiles)
        self.dtype = dtype

        means, scales = [], []
        for node in self.node_order:
            sc = scaler_y_map[node]
            m = np.asarray(sc.mean_, dtype=np.float64).reshape(1)
            s = np.asarray(sc.std_, dtype=np.float64).reshape(1)
            means.append(torch.tensor(m))
            scales.append(torch.tensor(s))

        mean = torch.stack(means, dim=0).reshape(len(self.node_order), 1, 1, 1)
        scale = torch.stack(scales, dim=0).reshape(len(self.node_order), 1, 1, 1)
        self.register_buffer("y_mean_N111", mean.to(dtype))
        self.register_buffer("y_scale_N111", scale.to(dtype))

    def forward(self, X_all):
        B, N, T, F = X_all.shape
        device = X_all.device
        Q = self.Q

        q_out = torch.empty((B, N, T, Q), dtype=self.dtype, device=device)

        for j, node in enumerate(self.node_order):
            X_j = X_all[:, j, :, :]                # [B,24,F]
            X_flat = X_j.reshape(B * T, F)         # [B*24,F]

            q_nor = self.models[str(node)](X_flat) # [B*24,Q] normalized
            q_nor = q_nor.reshape(B, T, Q)         # [B,24,Q]

            mean = self.y_mean_N111[j]
            scale = self.y_scale_N111[j]
            q_real = q_nor.to(self.dtype) * scale + mean  # [B,24,Q]

            # 修正 quantile crossing
            q_real = torch.cummax(q_real, dim=-1).values

            q_out[:, j, :, :] = q_real

        return q_out

def sample_from_quantiles_linear(q_curve_11TQ, taus, n_scen, clamp_min=0.0):
    """
    q_curve_11TQ: [N,T,Q]，例如 [11,24,Q]
    taus: [Q]，升序 quantile levels
    return: [S,N,T]
    
    逻辑：
    - u ~ Uniform(0,1)
    - 在已知 quantile 节点之间线性插值
    - 在左右尾做线性外推
    """
    if q_curve_11TQ.dim() != 3:
        raise ValueError(f"q_curve_11TQ must be [N,T,Q], got {tuple(q_curve_11TQ.shape)}")

    device = q_curve_11TQ.device
    dtype = q_curve_11TQ.dtype

    N, T, Q = q_curve_11TQ.shape
    if Q < 2:
        raise ValueError(f"Need at least 2 quantiles, got Q={Q}")

    taus = torch.as_tensor(taus, device=device, dtype=dtype).flatten()
    if taus.numel() != Q:
        raise ValueError(f"taus length {taus.numel()} != Q {Q}")

    # 保证 taus 升序；q_curve 同步重排
    order = torch.argsort(taus)
    taus = taus[order]
    q_curve_11TQ = q_curve_11TQ.index_select(dim=-1, index=order)

    # 修正 quantile crossing：单调不减
    q_curve_11TQ = torch.cummax(q_curve_11TQ, dim=-1).values

    # u ~ U(0,1)
    u = torch.rand((n_scen, N, T), device=device, dtype=dtype)  # [S,N,T]

    # bucketize:
    # idx_raw in [0..Q]
    #   0   => u < taus[0]
    #   Q   => u > taus[-1]
    #   k   => taus[k-1] <= u < taus[k]
    idx_raw = torch.bucketize(u, taus)

    # 左端点索引，夹到 [0, Q-2]
    idx_left = (idx_raw - 1).clamp(0, Q - 2)   # [S,N,T]
    idx_right = idx_left + 1                    # [S,N,T]

    # gather q0, q1
    q_expand = q_curve_11TQ.unsqueeze(0).expand(n_scen, -1, -1, -1)  # [S,N,T,Q]
    idx_left_4 = idx_left.unsqueeze(-1)
    idx_right_4 = idx_right.unsqueeze(-1)

    q0 = torch.gather(q_expand, dim=-1, index=idx_left_4).squeeze(-1)   # [S,N,T]
    q1 = torch.gather(q_expand, dim=-1, index=idx_right_4).squeeze(-1)  # [S,N,T]

    t0 = taus[idx_left]    # [S,N,T]
    t1 = taus[idx_right]   # [S,N,T]

    # 分段线性公式：中间是插值，左右自动成为外推
    w = (u - t0) / (t1 - t0 + 1e-12)
    scen = q0 + w * (q1 - q0)   # [S,N,T]

    if clamp_min is not None:
        scen = torch.clamp(scen, min=float(clamp_min))

    return scen


class DFL_model_nonparametric_Deterministic_SingleNode(nn.Module):
    def __init__(
        self,
        mgr,
        optnet_DA,
        optnet_RT,
        predictor,              # outputs q_real [B,11,T,Q]
        quantiles: list,
        scenario_filter=None,
        n_scen=50,
        clamp_min=0.0,
        solver="ECOS",
        dtype=torch.float64,
    ):
        super().__init__()
        self.mgr = mgr
        self.optnet_DA = optnet_DA
        self.optnet_RT = optnet_RT
        self.predictor = predictor
        self.scenario_filter = scenario_filter

        self.quantiles = list(quantiles)
        self.register_buffer("taus_Q", torch.tensor(self.quantiles, dtype=dtype))

        self.n_scen = int(n_scen)
        self.clamp_min = clamp_min
        self.solver = solver
        self.dtype = dtype
        self.optnet_dtype = dtype

        b = torch.tensor([mgr.gen_info[g]["b"] for g in mgr.gen_bus_list], dtype=dtype)
        self.register_buffer("b_G", b)

        self.res_up_ratio = float(getattr(mgr.args, "reserve_up_ratio", 0.05))
        self.res_dn_ratio = float(getattr(mgr.args, "reserve_dn_ratio", 0.02))

    def _canon_y_true_11T(self, y_true, T=24):
        if y_true.dim() != 3:
            raise ValueError(f"y_true_real must be [B,11,T] or [B,T,11], got {tuple(y_true.shape)}")
        if y_true.shape[1] == 11 and y_true.shape[2] == T:
            return y_true
        if y_true.shape[1] == T and y_true.shape[2] == 11:
            return y_true.permute(0, 2, 1).contiguous()
        raise ValueError(f"y_true_real must be [B,11,{T}] or [B,{T},11], got {tuple(y_true.shape)}")

    def _forecast_11T_from_q(self, q_curve_11TQ):
        if 0.5 in self.quantiles:
            i = self.quantiles.index(0.5)
            return q_curve_11TQ[:, :, i]
        return q_curve_11TQ.mean(dim=-1)

    def _sample_scenarios_for_filter(self, q_curve_11TQ, S):
        return sample_from_quantiles_linear(
            q_curve_11TQ, self.taus_Q, S, clamp_min=self.clamp_min
        )  # [S,11,T]

    def _realized_total_cost(self, P_DA_GT, R_up_GT, R_dn_GT, omega_true_T, solver):
        bG = self.b_G.to(device=P_DA_GT.device, dtype=self.dtype)

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
        y_true_real=None,
        hourly_load_min_sys=None,
        hourly_load_max_sys=None,
        eps=None,
        solver=None,
        return_aux=False,
        return_filter_aux=False,
        predictor_n_samples=None,
        filter_kwargs=None,
    ):
        del hourly_load_min_sys, hourly_load_max_sys, eps  # compatibility only

        solver = solver or self.solver
        device = X_all.device
        filter_kwargs = dict(filter_kwargs or {})

        if y_true_real is None:
            raise ValueError("Need y_true_real to compute realized (true) cost loss.")

        q_real = self.predictor(X_all).to(device=device, dtype=self.dtype)  # [B,11,T,Q]
        B, N, T, Q = q_real.shape

        S = int(predictor_n_samples) if predictor_n_samples is not None else self.n_scen
        y_true_11 = self._canon_y_true_11T(y_true_real, T=T).to(device=device, dtype=self.dtype)

        loss_list = []
        aux_filter_list = []

        aux = None
        if return_aux:
            aux = {
                "q_real": q_real.detach(),
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
            q_curve_11TQ = q_real[b]
            forecast_11T = self._forecast_11T_from_q(q_curve_11TQ)
            forecast_sys_T = forecast_11T.sum(dim=0)

            aux_filter = None
            if self.scenario_filter is not None:
                scenarios_11T = self._sample_scenarios_for_filter(q_curve_11TQ, S)  # [S,11,T]
                scen_11T_f, aux_filter = self.scenario_filter(
                    scenarios_11T.unsqueeze(1), is_train=self.training, **filter_kwargs
                )
                _ = scen_11T_f.squeeze(1).to(device=device, dtype=self.dtype)
            aux_filter_list.append(aux_filter)

            sol_da = self.optnet_DA(
                forecast_sys_T.unsqueeze(0),
                solver=solver,
            )
            P_DA = sol_da["P_DA"][0].to(dtype=self.dtype)
            R_up = sol_da["R_up"][0].to(dtype=self.dtype)
            R_dn = sol_da["R_dn"][0].to(dtype=self.dtype)

            omega_true_T = y_true_11[b].sum(dim=0) - forecast_sys_T
            realized_total, sol_rt = self._realized_total_cost(
                P_DA, R_up, R_dn, omega_true_T, solver=solver
            )
            loss_list.append(realized_total)

            if return_aux:
                aux["forecast_sys"].append(forecast_sys_T.detach())
                aux["omega_true"].append(omega_true_T.detach())
                obj = sol_da.get("obj", None)
                if isinstance(obj, torch.Tensor):
                    aux["det_obj"].append(obj[0].detach())
                else:
                    aux["det_obj"].append(torch.tensor(float("nan"), device=device, dtype=self.dtype))
                aux["realized_total"].append(realized_total.detach())
                aux["rt_obj_true"].append(sol_rt["rt_obj"][0].detach())
                aux["P_DA"].append(P_DA.detach())
                aux["R_up"].append(R_up.detach())
                aux["R_dn"].append(R_dn.detach())

        loss_vec = torch.stack(loss_list, dim=0)

        if return_aux:
            for k in ["forecast_sys", "omega_true", "det_obj", "realized_total",
                      "rt_obj_true", "P_DA", "R_up", "R_dn"]:
                aux[k] = torch.stack(aux[k], dim=0)

        if return_aux and not return_filter_aux:
            return loss_vec, aux
        if return_filter_aux and not return_aux:
            return loss_vec, aux_filter_list
        if return_aux and return_filter_aux:
            return loss_vec, aux, aux_filter_list
        return loss_vec

class DFL_model_nonparametric_Deterministic_MultiNode(nn.Module):
    def __init__(
        self,
        mgr,
        optnet_DA,
        optnet_RT,
        predictor,
        quantiles: list,
        scenario_filter=None,
        n_scen=20,
        clamp_min=0.0,
        solver="ECOS",
        dtype=torch.float64,
    ):
        super().__init__()
        self.mgr = mgr
        self.optnet_DA = optnet_DA
        self.optnet_RT = optnet_RT
        self.predictor = predictor
        self.scenario_filter = scenario_filter

        self.quantiles = list(quantiles)
        self.register_buffer("taus_Q", torch.tensor(self.quantiles, dtype=dtype))

        self.n_scen = int(n_scen)
        self.clamp_min = clamp_min
        self.solver = solver
        self.dtype = dtype
        self.optnet_dtype = dtype

        bus_sched_0 = [b - 1 for b in mgr.default_bus_order]
        self.register_buffer("bus_indices_11_to_14", torch.tensor(bus_sched_0, dtype=torch.long))

        b_vals = [mgr.gen_info[g]["b"] for g in mgr.gen_bus_list]
        self.register_buffer("b_G", torch.tensor(b_vals, dtype=dtype))

        self.res_up_ratio = float(getattr(mgr.args, "reserve_up_ratio", 0.05))
        self.res_dn_ratio = float(getattr(mgr.args, "reserve_dn_ratio", 0.02))

    def _canon_y_true_11T(self, y_true, T=24):
        if y_true.dim() != 3:
            raise ValueError(f"y_true_real must be [B,11,T] or [B,T,11], got {tuple(y_true.shape)}")
        if y_true.shape[1] == 11 and y_true.shape[2] == T:
            return y_true
        if y_true.shape[1] == T and y_true.shape[2] == 11:
            return y_true.permute(0, 2, 1).contiguous()
        raise ValueError(f"y_true_real must be [B,11,{T}] or [B,{T},11], got {tuple(y_true.shape)}")

    def _map_11_to_14(self, x_11):
        *prefix, n11, T = x_11.shape
        if n11 != 11:
            raise ValueError(f"expected 11 nodes, got {n11}")
        out = x_11.new_zeros((*prefix, 14, T))
        out[..., self.bus_indices_11_to_14, :] = x_11
        return out

    def _forecast_11T_from_q(self, q_11TQ):
        if 0.5 in self.quantiles:
            i = self.quantiles.index(0.5)
            return q_11TQ[..., i]
        return q_11TQ.mean(dim=-1)

    def _sample_scenarios_from_q(self, q_11TQ, S):
        B, N, T, Q = q_11TQ.shape
        scen_list = []
        for b in range(B):
            scen_b = sample_from_quantiles_linear(
                q_11TQ[b], self.taus_Q, S, clamp_min=self.clamp_min
            )
            scen_list.append(scen_b)
        return torch.stack(scen_list, dim=0)  # [B,S,11,T]

    def forward(
        self,
        X_all,
        y_true_real=None,
        hourly_load_min_sys=None,
        hourly_load_max_sys=None,
        eps=None,
        solver=None,
        return_aux=False,
        return_filter_aux=False,
        predictor_n_samples=None,
        filter_kwargs=None,
    ):
        del hourly_load_min_sys, hourly_load_max_sys, eps  # compatibility only

        solver = solver or self.solver
        device = X_all.device
        dtype = self.dtype
        filter_kwargs = dict(filter_kwargs or {})

        if y_true_real is None:
            raise ValueError("Need y_true_real with shape [B,11,T] for realized cost.")

        q_real = self.predictor(X_all).to(device=device, dtype=dtype)   # [B,11,T,Q]
        B, N, T, Q = q_real.shape
        if N != 11:
            raise ValueError(f"Expected q_real [B,11,T,Q], got {tuple(q_real.shape)}")

        forecast11 = self._forecast_11T_from_q(q_real)   # [B,11,T]
        S = int(predictor_n_samples) if predictor_n_samples is not None else self.n_scen

        aux_filter = None
        if self.scenario_filter is not None:
            scen_11 = self._sample_scenarios_from_q(q_real, S).to(device=device, dtype=dtype)  # [B,S,11,T]
            scen_for_filter = scen_11.permute(1, 0, 2, 3).contiguous()  # [S,B,11,T]
            scen_for_filter, aux_filter = self.scenario_filter(
                scen_for_filter, is_train=self.training, **filter_kwargs
            )
            _ = scen_for_filter.to(device=device, dtype=dtype)

        forecast14 = self._map_11_to_14(forecast11)

        sol_DA = self.optnet_DA(
            forecast14,
            solver=solver,
            return_cost=True,
        )
        P_DA = sol_DA["P_DA"].to(device=device, dtype=dtype)
        R_up = sol_DA["R_up"].to(device=device, dtype=dtype)
        R_dn = sol_DA["R_dn"].to(device=device, dtype=dtype)

        y_true_11 = self._canon_y_true_11T(y_true_real, T=T).to(device=device, dtype=dtype)
        y_true_14 = self._map_11_to_14(y_true_11)
        omega14_true = y_true_14 - forecast14
        sol_RT = self.optnet_RT(R_up, R_dn, omega14_true, solver=solver)
        rt_obj_true = sol_RT["rt_obj"].to(device=device, dtype=dtype)

        bG = self.b_G.to(device=device, dtype=dtype)
        cost_energy = (P_DA * bG[None, :, None]).sum(dim=(1, 2))
        cost_reserve = (
            (R_up * (self.res_up_ratio * bG)[None, :, None]).sum(dim=(1, 2)) +
            (R_dn * (self.res_dn_ratio * bG)[None, :, None]).sum(dim=(1, 2))
        )
        total_realized_cost = cost_energy + cost_reserve + rt_obj_true

        aux = None
        if return_aux:
            aux = {
                "q_real": q_real.detach(),
                "forecast11": forecast11.detach(),
                "forecast14": forecast14.detach(),
                "cost_energy": cost_energy.detach(),
                "cost_reserve": cost_reserve.detach(),
                "rt_obj_true": rt_obj_true.detach(),
                "P_DA": P_DA.detach(),
                "R_up": R_up.detach(),
                "R_dn": R_dn.detach(),
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

class DFL_model_nonparametric_SingleNode(nn.Module):
    def __init__(
        self,
        mgr,
        optnet_DA,
        optnet_RT,
        predictor,              # outputs q_real [B,11,T,Q]
        quantiles: list,
        scenario_filter=None,   # <- added
        n_scen=50,
        clamp_min=0.0,
        solver="ECOS",
        dtype=torch.float64,
    ):
        super().__init__()
        self.mgr = mgr
        self.optnet_DA = optnet_DA
        self.optnet_RT = optnet_RT
        self.predictor = predictor
        self.scenario_filter = scenario_filter

        self.quantiles = list(quantiles)
        self.register_buffer("taus_Q", torch.tensor(self.quantiles, dtype=dtype))

        self.n_scen = int(n_scen)
        self.clamp_min = clamp_min
        self.solver = solver
        self.dtype = dtype
        self.optnet_dtype = dtype

        b = torch.tensor([mgr.gen_info[g]["b"] for g in mgr.gen_bus_list], dtype=dtype)
        self.register_buffer("b_G", b)

        self.res_up_ratio = float(getattr(mgr.args, "reserve_up_ratio", 0.05))
        self.res_dn_ratio = float(getattr(mgr.args, "reserve_dn_ratio", 0.02))

    def _canon_y_true_11T(self, y_true, T=24):
        if y_true.dim() != 3:
            raise ValueError(f"y_true_real must be [B,11,T] or [B,T,11], got {tuple(y_true.shape)}")
        if y_true.shape[1] == 11 and y_true.shape[2] == T:
            return y_true
        if y_true.shape[1] == T and y_true.shape[2] == 11:
            return y_true.permute(0, 2, 1).contiguous()
        raise ValueError(f"y_true_real must be [B,11,{T}] or [B,{T},11], got {tuple(y_true.shape)}")

    def _forecast_11T_from_q(self, q_curve_11TQ):
        if 0.5 in self.quantiles:
            i = self.quantiles.index(0.5)
            return q_curve_11TQ[:, :, i]
        return q_curve_11TQ.mean(dim=-1)

    def _build_single_node_inputs(self, scenarios_11T, forecast_11T):
        forecast_sys = forecast_11T.sum(dim=0)            # [T]
        scen_sys = scenarios_11T.sum(dim=1)               # [S,T]
        omega = scen_sys - forecast_sys.unsqueeze(0)      # [S,T]
        return forecast_sys, omega

    def _realized_total_cost(self, P_DA_GT, R_up_GT, R_dn_GT, omega_true_T, solver):
        bG = self.b_G.to(device=P_DA_GT.device, dtype=self.dtype)

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
        y_true_real=None,
        solver=None,
        return_aux=False,
        return_filter_aux=False,   # <- added
        predictor_n_samples=None,
        filter_kwargs=None,        # <- added
    ):
        solver = solver or self.solver
        device = X_all.device
        filter_kwargs = dict(filter_kwargs or {})

        if y_true_real is None:
            raise ValueError("Need y_true_real to compute realized (true) cost loss.")

        q_real = self.predictor(X_all).to(device=device, dtype=self.dtype)  # [B,11,T,Q]
        B, N, T, Q = q_real.shape

        S = int(predictor_n_samples) if predictor_n_samples is not None else self.n_scen

        # sample scenarios per batch sample -> [B,S,11,T]
        scen_list = []
        forecast_list = []
        for b in range(B):
            q_curve_11TQ = q_real[b]
            forecast_11T = self._forecast_11T_from_q(q_curve_11TQ)
            scenarios_11T = sample_from_quantiles_linear(
                q_curve_11TQ, self.taus_Q, S, clamp_min=self.clamp_min
            )  # [S,11,T]
            scen_list.append(scenarios_11T)
            forecast_list.append(forecast_11T)

        scen_11 = torch.stack(scen_list, dim=0).to(device=device, dtype=self.dtype)       # [B,S,11,T]
        forecast11 = torch.stack(forecast_list, dim=0).to(device=device, dtype=self.dtype) # [B,11,T]

        # scenario filter hook expects [S,B,11,T]
        aux_filter = None
        if self.scenario_filter is not None:
            scen_for_filter = scen_11.permute(1, 0, 2, 3).contiguous()
            scen_for_filter, aux_filter = self.scenario_filter(
                scen_for_filter, is_train=self.training, **filter_kwargs
            )
            scen_for_filter = scen_for_filter.to(device=device, dtype=self.dtype)
            scen_11 = scen_for_filter.permute(1, 0, 2, 3).contiguous()  # back [B,S,11,T]

        y_true_11 = self._canon_y_true_11T(y_true_real, T=T).to(device=device, dtype=self.dtype)

        loss_list = []
        aux = None
        if return_aux:
            aux = {
                "q_real": q_real.detach(),
                "forecast_sys": [],
                "omega_true": [],
                "saa_obj": [],
                "realized_total": [],
                "rt_obj_true": [],
                "P_DA": [],
                "R_up": [],
                "R_dn": [],
            }

        for b in range(B):
            forecast_11T = forecast11[b]
            scenarios_11T = scen_11[b]
            forecast_sys_T, omega_saa_ST = self._build_single_node_inputs(scenarios_11T, forecast_11T)

            sol_da = self.optnet_DA(
                forecast_sys_T.unsqueeze(0),
                omega_saa_ST.unsqueeze(0),
                solver=solver,
                return_rt=False
            )
            P_DA = sol_da["P_DA"][0].to(dtype=self.dtype)
            R_up = sol_da["R_up"][0].to(dtype=self.dtype)
            R_dn = sol_da["R_dn"][0].to(dtype=self.dtype)

            omega_true_T = y_true_11[b].sum(dim=0) - forecast_11T.sum(dim=0)
            realized_total, sol_rt = self._realized_total_cost(P_DA, R_up, R_dn, omega_true_T, solver=solver)
            loss_list.append(realized_total)

            if return_aux:
                aux["forecast_sys"].append(forecast_sys_T.detach())
                aux["omega_true"].append(omega_true_T.detach())
                obj = sol_da.get("obj", None)
                if isinstance(obj, torch.Tensor):
                    aux["saa_obj"].append(obj[0].detach())
                else:
                    aux["saa_obj"].append(torch.tensor(float("nan"), device=device, dtype=self.dtype))
                aux["realized_total"].append(realized_total.detach())
                aux["rt_obj_true"].append(sol_rt["rt_obj"][0].detach())
                aux["P_DA"].append(P_DA.detach())
                aux["R_up"].append(R_up.detach())
                aux["R_dn"].append(R_dn.detach())

        loss_vec = torch.stack(loss_list, dim=0)

        if return_aux:
            for k in ["forecast_sys", "omega_true", "saa_obj", "realized_total", "rt_obj_true", "P_DA", "R_up", "R_dn"]:
                aux[k] = torch.stack(aux[k], dim=0)

        if return_aux and not return_filter_aux:
            return loss_vec, aux
        if return_filter_aux and not return_aux:
            return loss_vec, aux_filter
        if return_aux and return_filter_aux:
            return loss_vec, aux, aux_filter
        return loss_vec

class DFL_model_nonparametric_MultiNode(nn.Module):
    def __init__(
        self,
        mgr,
        optnet_DA,
        optnet_RT,
        predictor,
        quantiles: list,
        scenario_filter=None,   # <- added
        n_scen=20,
        clamp_min=0.0,
        solver="ECOS",
        dtype=torch.float64,
    ):
        super().__init__()
        self.mgr = mgr
        self.optnet_DA = optnet_DA
        self.optnet_RT = optnet_RT
        self.predictor = predictor
        self.scenario_filter = scenario_filter

        self.quantiles = list(quantiles)
        self.register_buffer("taus_Q", torch.tensor(self.quantiles, dtype=dtype))

        self.n_scen = int(n_scen)
        self.clamp_min = clamp_min
        self.solver = solver
        self.dtype = dtype
        self.optnet_dtype = dtype

        bus_sched_0 = [b - 1 for b in mgr.default_bus_order]
        self.register_buffer("bus_indices_11_to_14", torch.tensor(bus_sched_0, dtype=torch.long))

        b_vals = [mgr.gen_info[g]["b"] for g in mgr.gen_bus_list]
        self.register_buffer("b_G", torch.tensor(b_vals, dtype=dtype))

        self.res_up_ratio = float(getattr(mgr.args, "reserve_up_ratio", 0.05))
        self.res_dn_ratio = float(getattr(mgr.args, "reserve_dn_ratio", 0.02))

    def _canon_y_true_11T(self, y_true, T=24):
        if y_true.dim() != 3:
            raise ValueError(f"y_true_real must be [B,11,T] or [B,T,11], got {tuple(y_true.shape)}")
        if y_true.shape[1] == 11 and y_true.shape[2] == T:
            return y_true
        if y_true.shape[1] == T and y_true.shape[2] == 11:
            return y_true.permute(0, 2, 1).contiguous()
        raise ValueError(f"y_true_real must be [B,11,{T}] or [B,{T},11], got {tuple(y_true.shape)}")

    def _map_11_to_14(self, x_11):
        *prefix, n11, T = x_11.shape
        if n11 != 11:
            raise ValueError(f"expected 11 nodes, got {n11}")
        out = x_11.new_zeros((*prefix, 14, T))
        out[..., self.bus_indices_11_to_14, :] = x_11
        return out

    def _forecast_11T_from_q(self, q_11TQ):
        if 0.5 in self.quantiles:
            i = self.quantiles.index(0.5)
            return q_11TQ[..., i]
        return q_11TQ.mean(dim=-1)

    def _sample_scenarios_from_q(self, q_11TQ, S):
        B, N, T, Q = q_11TQ.shape
        scen_list = []
        for b in range(B):
            scen_b = sample_from_quantiles_linear(
                q_11TQ[b], self.taus_Q, S, clamp_min=self.clamp_min
            )
            scen_list.append(scen_b)
        return torch.stack(scen_list, dim=0)  # [B,S,11,T]

    def forward(
        self,
        X_all,
        y_true_real=None,
        solver=None,
        return_aux=False,
        return_filter_aux=False,   # <- added
        predictor_n_samples=None,
        filter_kwargs=None,        # <- added
    ):
        solver = solver or self.solver
        device = X_all.device
        dtype = self.dtype
        filter_kwargs = dict(filter_kwargs or {})

        if y_true_real is None:
            raise ValueError("Need y_true_real with shape [B,11,T] for realized cost.")

        q_real = self.predictor(X_all).to(device=device, dtype=dtype)   # [B,11,T,Q]
        B, N, T, Q = q_real.shape
        if N != 11:
            raise ValueError(f"Expected q_real [B,11,T,Q], got {tuple(q_real.shape)}")

        forecast11 = self._forecast_11T_from_q(q_real)                  # [B,11,T]
        S = int(predictor_n_samples) if predictor_n_samples is not None else self.n_scen
        scen_11 = self._sample_scenarios_from_q(q_real, S).to(device=device, dtype=dtype)  # [B,S,11,T]

        aux_filter = None
        if self.scenario_filter is not None:
            scen_for_filter = scen_11.permute(1, 0, 2, 3).contiguous()  # [S,B,11,T]
            scen_for_filter, aux_filter = self.scenario_filter(
                scen_for_filter, is_train=self.training, **filter_kwargs
            )
            scen_for_filter = scen_for_filter.to(device=device, dtype=dtype)
            scen_11 = scen_for_filter.permute(1, 0, 2, 3).contiguous()  # [B,S,11,T]

        forecast14 = self._map_11_to_14(forecast11)
        scen14 = self._map_11_to_14(scen_11)
        omega14 = scen14 - forecast14.unsqueeze(1)

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

        y_true_11 = self._canon_y_true_11T(y_true_real, T=T).to(device=device, dtype=dtype)
        y_true_14 = self._map_11_to_14(y_true_11)
        rt_obj_true = sol_RT["rt_obj"].to(device=device, dtype=dtype)

        bG = self.b_G.to(device=device, dtype=dtype)
        cost_energy = (P_DA * bG[None, :, None]).sum(dim=(1, 2))
        cost_reserve = (
            (R_up * (self.res_up_ratio * bG)[None, :, None]).sum(dim=(1, 2)) +
            (R_dn * (self.res_dn_ratio * bG)[None, :, None]).sum(dim=(1, 2))
        )
        total_realized_cost = cost_energy + cost_reserve + rt_obj_true

        aux = None
        if return_aux:
            aux = {
                "q_real": q_real.detach(),
                "forecast11": forecast11.detach(),
                "forecast14": forecast14.detach(),
                "omega14": omega14.detach(),
                "cost_energy": cost_energy.detach(),
                "cost_reserve": cost_reserve.detach(),
                "rt_obj_true": rt_obj_true.detach(),
                "P_DA": P_DA.detach(),
                "R_up": R_up.detach(),
                "R_dn": R_dn.detach(),
            }

        if return_aux and not return_filter_aux:
            return total_realized_cost, aux
        if return_filter_aux and not return_aux:
            return total_realized_cost, aux_filter
        if return_aux and return_filter_aux:
            return total_realized_cost, aux, aux_filter
        return total_realized_cost

class DFL_model_nonparametric_DRO_SingleNode(nn.Module):
    def __init__(
        self,
        mgr,
        optnet_DA,
        optnet_RT,
        predictor,
        quantiles: list,
        scenario_filter=None,   # <- added
        n_scen=50,
        clamp_min=0.0,
        solver="ECOS",
        dtype=torch.float64,
    ):
        super().__init__()
        self.mgr = mgr
        self.optnet_DA = optnet_DA
        self.optnet_RT = optnet_RT
        self.predictor = predictor
        self.scenario_filter = scenario_filter

        self.quantiles = list(quantiles)
        self.register_buffer("taus_Q", torch.tensor(self.quantiles, dtype=dtype))

        self.n_scen = int(n_scen)
        self.clamp_min = clamp_min
        self.solver = solver
        self.dtype = dtype
        self.optnet_dtype = dtype

        b = torch.tensor([mgr.gen_info[g]["b"] for g in mgr.gen_bus_list], dtype=dtype)
        self.register_buffer("b_G", b)

        self.res_up_ratio = float(getattr(mgr.args, "reserve_up_ratio", 0.05))
        self.res_dn_ratio = float(getattr(mgr.args, "reserve_dn_ratio", 0.02))

    def _canon_y_true_11T(self, y_true, T=24):
        if y_true.dim() != 3:
            raise ValueError(f"y_true_real must be [B,11,T] or [B,T,11], got {tuple(y_true.shape)}")
        if y_true.shape[1] == 11 and y_true.shape[2] == T:
            return y_true
        if y_true.shape[1] == T and y_true.shape[2] == 11:
            return y_true.permute(0, 2, 1).contiguous()
        raise ValueError(f"y_true_real must be [B,11,{T}] or [B,{T},11], got {tuple(y_true.shape)}")

    def _forecast_11T_from_q(self, q_curve_11TQ):
        if 0.5 in self.quantiles:
            i = self.quantiles.index(0.5)
            return q_curve_11TQ[:, :, i]
        return q_curve_11TQ.mean(dim=-1)

    def _realized_total_cost(self, P_DA_GT, R_up_GT, R_dn_GT, omega_true_T, solver):
        bG = self.b_G.to(device=P_DA_GT.device, dtype=self.dtype)
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
        return stage1_energy_linear + stage1_reserve + sol_rt["rt_obj"][0], sol_rt

    def _to_BT(self, x, B, device, dtype):
        x = torch.as_tensor(x, device=device, dtype=dtype)
        if x.ndim == 1:
            x = x.unsqueeze(0).expand(B, -1)
        return x

    def forward(
        self,
        X_all,
        y_true_real,
        hourly_load_min_sys,
        hourly_load_max_sys,
        eps,
        solver=None,
        return_aux=False,
        return_filter_aux=False,   # <- added
        predictor_n_samples=None,
        filter_kwargs=None,        # <- added
    ):
        solver = solver or self.solver
        device = X_all.device
        dtype = self.dtype
        filter_kwargs = dict(filter_kwargs or {})

        if y_true_real is None:
            raise ValueError("Need y_true_real to compute realized (true) cost loss.")

        q_real = self.predictor(X_all).to(device=device, dtype=dtype)  # [B,11,T,Q]
        B, N, T, Q = q_real.shape
        if N != 11:
            raise ValueError(f"Expected q_real [B,11,T,Q], got {tuple(q_real.shape)}")

        S = int(predictor_n_samples) if predictor_n_samples is not None else self.n_scen
        Lmin_sys = self._to_BT(hourly_load_min_sys, B=B, device=device, dtype=dtype)
        Lmax_sys = self._to_BT(hourly_load_max_sys, B=B, device=device, dtype=dtype)

        eps_t = torch.as_tensor(eps, device=device, dtype=dtype)
        if eps_t.ndim == 0:
            eps_t = eps_t.view(1).expand(B)

        # pre-sample all scenarios -> [B,S,11,T]
        scen_list = []
        forecast_list = []
        for b in range(B):
            q_curve_11TQ = q_real[b]
            forecast_11T = self._forecast_11T_from_q(q_curve_11TQ)
            scenarios_11T = sample_from_quantiles_linear(
                q_curve_11TQ, self.taus_Q, S, clamp_min=self.clamp_min
            )
            scen_list.append(scenarios_11T)
            forecast_list.append(forecast_11T)

        scen_11 = torch.stack(scen_list, dim=0).to(device=device, dtype=dtype)       # [B,S,11,T]
        forecast11 = torch.stack(forecast_list, dim=0).to(device=device, dtype=dtype) # [B,11,T]

        aux_filter = None
        if self.scenario_filter is not None:
            scen_for_filter = scen_11.permute(1, 0, 2, 3).contiguous()  # [S,B,11,T]
            scen_for_filter, aux_filter = self.scenario_filter(
                scen_for_filter, is_train=self.training, **filter_kwargs
            )
            scen_for_filter = scen_for_filter.to(device=device, dtype=dtype)
            scen_11 = scen_for_filter.permute(1, 0, 2, 3).contiguous()

        y_true_11 = self._canon_y_true_11T(y_true_real, T=T).to(device=device, dtype=dtype)

        loss_list = []
        aux = None
        if return_aux:
            aux = {
                "q_real": q_real.detach(),
                "forecast_sys": [],
                "omega_true": [],
                "dro_obj": [],
                "gamma": [],
                "realized_total": [],
                "rt_obj_true": [],
                "P_DA": [],
                "R_up": [],
                "R_dn": [],
                "om_min": [],
                "om_max": [],
            }

        for b in range(B):
            forecast_11T = forecast11[b]
            forecast_sys_T = forecast_11T.sum(dim=0)
            scenarios_11T = scen_11[b]
            scen_sys_ST = scenarios_11T.sum(dim=1)
            omega_scen_ST = scen_sys_ST - forecast_sys_T.unsqueeze(0)

            om_min_h = Lmin_sys[b] - forecast_sys_T
            om_max_h = Lmax_sys[b] - forecast_sys_T
            om_min_h, om_max_h = torch.minimum(om_min_h, om_max_h), torch.maximum(om_min_h, om_max_h)

            om_min_s = omega_scen_ST.min(dim=0).values
            om_max_s = omega_scen_ST.max(dim=0).values
            om_min = torch.minimum(om_min_h, om_min_s)
            om_max = torch.maximum(om_max_h, om_max_s)

            sol_da = self.optnet_DA(
                forecast_sys_T.unsqueeze(0),
                omega_scen_ST.unsqueeze(0),
                om_min.unsqueeze(0),
                om_max.unsqueeze(0),
                eps_t[b].view(1),
                solver=solver,
            )
            P_DA = sol_da["P_DA"][0].to(dtype=dtype)
            R_up = sol_da["R_up"][0].to(dtype=dtype)
            R_dn = sol_da["R_dn"][0].to(dtype=dtype)

            omega_true_T = y_true_11[b].sum(dim=0) - forecast_sys_T
            realized_total, sol_rt = self._realized_total_cost(P_DA, R_up, R_dn, omega_true_T, solver=solver)
            loss_list.append(realized_total)

            if return_aux:
                aux["forecast_sys"].append(forecast_sys_T.detach())
                aux["omega_true"].append(omega_true_T.detach())
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
                aux["P_DA"].append(P_DA.detach())
                aux["R_up"].append(R_up.detach())
                aux["R_dn"].append(R_dn.detach())
                aux["om_min"].append(om_min.detach())
                aux["om_max"].append(om_max.detach())

        loss_vec = torch.stack(loss_list, dim=0)

        if return_aux:
            for k in ["forecast_sys", "omega_true", "dro_obj", "gamma", "realized_total",
                      "rt_obj_true", "P_DA", "R_up", "R_dn", "om_min", "om_max"]:
                aux[k] = torch.stack(aux[k], dim=0)

        if return_aux and not return_filter_aux:
            return loss_vec, aux
        if return_filter_aux and not return_aux:
            return loss_vec, aux_filter
        if return_aux and return_filter_aux:
            return loss_vec, aux, aux_filter
        return loss_vec

class DFL_model_nonparametric_DRO_MultiNode(nn.Module):
    def __init__(
        self,
        mgr,
        optnet_DA,
        optnet_RT,
        predictor,
        quantiles: list,
        scenario_filter=None,   # <- added
        n_scen=20,
        clamp_min=0.0,
        solver="ECOS",
        dtype=torch.float64,
    ):
        super().__init__()
        self.mgr = mgr
        self.optnet_DA = optnet_DA
        self.optnet_RT = optnet_RT
        self.predictor = predictor
        self.scenario_filter = scenario_filter

        self.quantiles = list(quantiles)
        self.register_buffer("taus_Q", torch.tensor(self.quantiles, dtype=dtype))

        self.n_scen = int(n_scen)
        self.clamp_min = clamp_min
        self.solver = solver
        self.dtype = dtype
        self.optnet_dtype = dtype

        bus_sched_0 = [b - 1 for b in mgr.default_bus_order]
        self.register_buffer("bus_indices_11_to_14", torch.tensor(bus_sched_0, dtype=torch.long))

        b_vals = [mgr.gen_info[g]["b"] for g in mgr.gen_bus_list]
        self.register_buffer("b_G", torch.tensor(b_vals, dtype=dtype))

        self.res_up_ratio = float(getattr(mgr.args, "reserve_up_ratio", 0.05))
        self.res_dn_ratio = float(getattr(mgr.args, "reserve_dn_ratio", 0.02))

    def _canon_y_true_11T(self, y_true, T=24):
        if y_true.dim() != 3:
            raise ValueError(f"y_true_real bad shape {tuple(y_true.shape)}")
        if y_true.shape[1] == 11 and y_true.shape[2] == T:
            return y_true
        if y_true.shape[1] == T and y_true.shape[2] == 11:
            return y_true.permute(0, 2, 1).contiguous()
        raise ValueError(f"y_true_real must be [B,11,{T}] or [B,{T},11], got {tuple(y_true.shape)}")

    def _map_11_to_14(self, x_11):
        *prefix, n11, T = x_11.shape
        if n11 != 11:
            raise ValueError(f"expected 11 nodes, got {n11}")
        out = x_11.new_zeros((*prefix, 14, T))
        out[..., self.bus_indices_11_to_14, :] = x_11
        return out

    def _forecast_11T_from_q(self, q_11TQ):
        if 0.5 in self.quantiles:
            i = self.quantiles.index(0.5)
            return q_11TQ[..., i]
        return q_11TQ.mean(dim=-1)

    def _sample_scenarios_from_q(self, q_11TQ, S):
        B, N, T, Q = q_11TQ.shape
        scen_list = []
        for b in range(B):
            scen_b = sample_from_quantiles_linear(
                q_11TQ[b], self.taus_Q, S, clamp_min=self.clamp_min
            )
            scen_list.append(scen_b)
        return torch.stack(scen_list, dim=0)  # [B,S,11,T]

    def forward(
        self,
        X_all,
        y_true_real,
        hourly_load_min_sys,
        hourly_load_max_sys,
        eps,
        solver=None,
        return_aux=False,
        return_filter_aux=False,   # <- added
        predictor_n_samples=None,
        filter_kwargs=None,        # <- added
    ):
        solver = solver or self.solver
        device = X_all.device
        dtype = self.dtype
        filter_kwargs = dict(filter_kwargs or {})

        if y_true_real is None:
            raise ValueError("Need y_true_real with shape [B,11,T] for realized cost.")

        q_real = self.predictor(X_all).to(device=device, dtype=dtype)  # [B,11,T,Q]
        B, N, T, Q = q_real.shape
        if N != 11:
            raise ValueError(f"Expected q_real [B,11,T,Q], got {tuple(q_real.shape)}")

        forecast11 = self._forecast_11T_from_q(q_real)
        S = int(predictor_n_samples) if predictor_n_samples is not None else self.n_scen
        scen_11 = self._sample_scenarios_from_q(q_real, S).to(device=device, dtype=dtype)

        aux_filter = None
        if self.scenario_filter is not None:
            scen_for_filter = scen_11.permute(1, 0, 2, 3).contiguous()  # [S,B,11,T]
            scen_for_filter, aux_filter = self.scenario_filter(
                scen_for_filter, is_train=self.training, **filter_kwargs
            )
            scen_for_filter = scen_for_filter.to(device=device, dtype=dtype)
            scen_11 = scen_for_filter.permute(1, 0, 2, 3).contiguous()

        forecast14 = self._map_11_to_14(forecast11)
        scen14 = self._map_11_to_14(scen_11)
        omega14 = scen14 - forecast14.unsqueeze(1)

        Lmin14 = torch.as_tensor(hourly_load_min_sys, device=device, dtype=dtype)
        Lmax14 = torch.as_tensor(hourly_load_max_sys, device=device, dtype=dtype)

        if Lmin14.ndim == 2:
            Lmin14 = Lmin14.unsqueeze(0).expand(B, -1, -1).contiguous()
        if Lmax14.ndim == 2:
            Lmax14 = Lmax14.unsqueeze(0).expand(B, -1, -1).contiguous()

        if Lmin14.shape != forecast14.shape:
            raise ValueError(f"Lmin14 {tuple(Lmin14.shape)} must match forecast14 {tuple(forecast14.shape)}")
        if Lmax14.shape != forecast14.shape:
            raise ValueError(f"Lmax14 {tuple(Lmax14.shape)} must match forecast14 {tuple(forecast14.shape)}")

        om_min14 = Lmin14 - forecast14
        om_max14 = Lmax14 - forecast14
        om_min14, om_max14 = torch.minimum(om_min14, om_max14), torch.maximum(om_min14, om_max14)

        eps_t = torch.as_tensor(eps, device=device, dtype=dtype)
        if eps_t.ndim == 0:
            eps_t = eps_t.view(1).expand(B)

        sol_DA = self.optnet_DA(forecast14, omega14, om_min14, om_max14, eps_t, solver=solver)
        P_DA = sol_DA["P_DA"].to(device=device, dtype=dtype)
        R_up = sol_DA["R_up"].to(device=device, dtype=dtype)
        R_dn = sol_DA["R_dn"].to(device=device, dtype=dtype)

        y_true_11 = self._canon_y_true_11T(y_true_real, T=T).to(device=device, dtype=dtype)
        y_true_14 = self._map_11_to_14(y_true_11)

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
                "q_real": q_real.detach(),
                "forecast11": forecast11.detach(),
                "forecast14": forecast14.detach(),
                "omega14": omega14.detach(),
                "om_min14": om_min14.detach(),
                "om_max14": om_max14.detach(),
                "eps": eps_t.detach(),
                "P_DA": P_DA.detach(),
                "R_up": R_up.detach(),
                "R_dn": R_dn.detach(),
                "rt_obj_true": rt_obj_true.detach(),
                "cost_energy": cost_energy.detach(),
                "cost_reserve": cost_reserve.detach(),
                "da_obj": sol_DA.get("obj", None).detach()
                if isinstance(sol_DA.get("obj", None), torch.Tensor) else None,
                "gamma": sol_DA.get("gamma", None).detach()
                if isinstance(sol_DA.get("gamma", None), torch.Tensor) else None,
            }

        if return_aux and not return_filter_aux:
            return total_realized_cost, aux
        if return_filter_aux and not return_aux:
            return total_realized_cost, aux_filter
        if return_aux and return_filter_aux:
            return total_realized_cost, aux, aux_filter
        return total_realized_cost


def DFL_train(
    dfl,
    train_dataset,
    args,
    problem_mode="saa",
    train_mode="dfl",
    filter_kwargs=None,
    lambda_div=1e5,
):
    dfl.train()
    device = next(dfl.parameters()).device

    # ---- 与参考代码一致：统一 train batch size ----
    train_bs = int(
        getattr(
            args,
            "train_batch_size",
            getattr(args, "dfl_batch_size", getattr(args, "batch_size", 8))
        )
    )

    # ---- 同步 scenario_filter 的 eval 配置 ----
    eval_mode = str(getattr(args, "eval_mode", "soft")).lower()
    avoid_rand_duplicate = bool(getattr(args, "avoid_rand_duplicate", False))
    if getattr(dfl, "scenario_filter", None) is not None:
        if hasattr(dfl.scenario_filter, "eval_mode"):
            dfl.scenario_filter.eval_mode = eval_mode
        if hasattr(dfl.scenario_filter, "avoid_rand_duplicate"):
            dfl.scenario_filter.avoid_rand_duplicate = avoid_rand_duplicate

    if filter_kwargs is None:
        filter_kwargs = {
            "tau_gumbel": getattr(args, "tau_mix", 1.0),
            "eps_uniform": getattr(args, "eps_uniform", 0.1),
        }

    div_type = str(getattr(args, "div_type", "inner")).lower()
    div_eps = float(getattr(args, "div_eps", 1e-8))

    lr_decay = float(getattr(args, "lr_decay", 1.0))
    filter_lr_decay = float(getattr(args, "filter_lr_decay", lr_decay))
    dfl_lr_decay = float(getattr(args, "dfl_lr_decay", lr_decay))

    if str(problem_mode).lower() in ["saa", "so"]:
        Lmin = Lmax = eps_value = None
    else:
        Lmin = args.Lmin
        Lmax = args.Lmax
        eps_value = args.eps_value

    scenario_filter = getattr(dfl, "scenario_filter", None)

    # 当前 filter 是否真的可学习
    has_learnable_filter = (
        scenario_filter is not None and
        any(p.requires_grad for p in scenario_filter.parameters())
    )

    if train_mode == "filter_only" and scenario_filter is not None:
        # 只训练 filter
        for p in dfl.parameters():
            p.requires_grad = False
        for p in dfl.scenario_filter.parameters():
            p.requires_grad = True

        optim_params_filter = [p for p in dfl.scenario_filter.parameters() if p.requires_grad]
        optim_params_predictor = []

        if len(optim_params_filter) == 0:
            raise ValueError("train_mode='filter_only' but scenario_filter has no trainable parameters.")

        filter_lr = float(getattr(args, "lr", getattr(args, "filter_lr", 1e-3)))
        optim = torch.optim.SGD(
            [
                {"params": optim_params_filter, "lr": filter_lr, "name": "filter"},
            ],
            weight_decay=float(getattr(args, "weight_decay", 0.0)),
        )

    else:
        # 默认 joint / dfl 训练：至少训练 predictor
        for p in dfl.parameters():
            p.requires_grad = True

        optim_params_predictor = [p for p in dfl.predictor.parameters() if p.requires_grad]

        # 只有当前 filter 可学习时，才把它放进 optimizer
        if has_learnable_filter:
            optim_params_filter = [p for p in dfl.scenario_filter.parameters() if p.requires_grad]
        else:
            optim_params_filter = []

        filter_lr = float(getattr(args, "filter_lr", 1e-3))
        predictor_lr = float(getattr(args, "dfl_lr", getattr(args, "lr", 1e-5)))

        param_groups = []

        if len(optim_params_filter) > 0:
            param_groups.append(
                {"params": optim_params_filter, "lr": filter_lr, "name": "filter"}
            )

        if len(optim_params_predictor) > 0:
            param_groups.append(
                {"params": optim_params_predictor, "lr": predictor_lr, "name": "predictor"}
            )

        if len(param_groups) == 0:
            raise ValueError("No trainable parameters found in DFL_train.")

        optim = torch.optim.Adam(
            param_groups,
            weight_decay=float(getattr(args, "weight_decay", 0.0)),
        )

    # ---- debug info ----
    print("train_mode =", train_mode)
    print("scenario_filter type =", type(scenario_filter).__name__ if scenario_filter is not None else None)
    print("has_learnable_filter =", has_learnable_filter)
    print("lr_decay =", lr_decay, "filter_lr_decay =", filter_lr_decay, "dfl_lr_decay =", dfl_lr_decay)
    print("num filter params =", sum(p.numel() for p in optim_params_filter) if len(optim_params_filter) > 0 else 0)
    print("num predictor params =", sum(p.numel() for p in optim_params_predictor) if len(optim_params_predictor) > 0 else 0)
    print("train_batch_size =", train_bs)

    print("===== optimizer param groups =====")
    for i, pg in enumerate(optim.param_groups):
        n_params = sum(p.numel() for p in pg["params"])
        print(f"group {i}: name={pg.get('name', 'unknown')}, lr={pg['lr']}, n_params={n_params}")
    print("==================================")

    loader = DataLoader(
        train_dataset,
        batch_size=train_bs,
        shuffle=True,
        drop_last=False,
    )

    epochs = int(getattr(args, "epochs", 1))
    train_logs = []

    for ep in tqdm(range(epochs), desc=f"Train ({train_mode})", leave=True):
        epoch_task_loss, epoch_div_loss, samples_cnt = 0.0, 0.0, 0
        pbar = tqdm(loader, desc=f"Ep {ep+1}/{epochs}", leave=False)

        for X_all, y_true_real in pbar:
            X_all = X_all.to(device).float()
            y_true_real = y_true_real.to(device).float()
            optim.zero_grad(set_to_none=True)

            kwargs = dict(
                solver=getattr(args, "solver", None),
                return_aux=False,
                return_filter_aux=True,
                predictor_n_samples=int(getattr(args, "S_full", getattr(args, "N_scen", 50))),
                filter_kwargs=filter_kwargs,
            )

            if str(problem_mode).lower() == "dro":
                dtype = getattr(dfl, "optnet_dtype", torch.float64)
                kwargs.update(
                    hourly_load_min_sys=torch.as_tensor(Lmin, device=device, dtype=dtype),
                    hourly_load_max_sys=torch.as_tensor(Lmax, device=device, dtype=dtype),
                    eps=torch.as_tensor(eps_value, device=device, dtype=dtype),
                )

            out = dfl(X_all, y_true_real, **kwargs)
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
                BS=train_bs,
                LR=[pg["lr"] for pg in optim.param_groups],
            )

        # ---- lr decay: 按 group name 衰减 ----
        for pg in optim.param_groups:
            if pg.get("name") == "filter":
                pg["lr"] *= filter_lr_decay
            elif pg.get("name") == "predictor":
                pg["lr"] *= dfl_lr_decay

        print(f"[Epoch {ep+1}] lr after decay:")
        for i, pg in enumerate(optim.param_groups):
            print(f" group {i}: name={pg.get('name', 'unknown')}, lr={pg['lr']}")

        train_logs.append({
            "task": epoch_task_loss / max(samples_cnt, 1),
            "div": epoch_div_loss / max(samples_cnt, 1),
            "div_type": div_type,
            "train_batch_size": train_bs,
            "lr": {pg.get("name", f"group_{i}"): pg["lr"] for i, pg in enumerate(optim.param_groups)},
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
    device = next(dfl.parameters()).device

    # ---- 规范化 eval config ----
    eval_mode = str(getattr(args, "eval_mode", "soft")).lower()
    avoid_rand_duplicate = bool(getattr(args, "avoid_rand_duplicate", False))

    if getattr(dfl, "scenario_filter", None) is not None:
        if hasattr(dfl.scenario_filter, "eval_mode"):
            dfl.scenario_filter.eval_mode = eval_mode
        if hasattr(dfl.scenario_filter, "avoid_rand_duplicate"):
            dfl.scenario_filter.avoid_rand_duplicate = avoid_rand_duplicate

    print(f"[DFL_test] eval_mode={eval_mode}, avoid_rand_duplicate={avoid_rand_duplicate}")

    if str(problem_mode).lower() not in ["saa", "so"]:
        Lmin = args.Lmin
        Lmax = args.Lmax
        eps_value = args.eps_value
    else:
        Lmin = Lmax = eps_value = None

    if filter_kwargs is None:
        filter_kwargs = {
            "tau_gumbel": float(getattr(args, "tau_mix", 1.0)),
            "eps_uniform": float(getattr(args, "eps_uniform", 0.10)),
        }

    test_bs = int(getattr(args, "test_batch_size", getattr(args, "batch_size", 8)))

    loader = DataLoader(
        test_dataset,
        batch_size=test_bs,
        shuffle=False,
        drop_last=False,
    )

    losses = []
    all_filter_aux = [] if return_filter_aux else None
    losses_flat = []

    pbar = tqdm(loader, desc="Test", leave=True)
    for X_all, y_true_real in pbar:
        X_all = X_all.to(device).float()
        y_true_real = y_true_real.to(device).float()

        # 固定 test-time randomness
        set_seed(0)

        kwargs = dict(
            solver=getattr(args, "solver", None),
            return_aux=False,
            return_filter_aux=bool(return_filter_aux),
            predictor_n_samples=int(getattr(args, "predictor_n_samples", getattr(args, "S_full", getattr(args, "N_scen", 50)))),
            filter_kwargs=filter_kwargs,
        )

        if str(problem_mode).lower() == "dro":
            dtype = getattr(dfl, "optnet_dtype", torch.float64)
            kwargs.update(
                hourly_load_min_sys=torch.as_tensor(Lmin, device=device, dtype=dtype),
                hourly_load_max_sys=torch.as_tensor(Lmax, device=device, dtype=dtype),
                eps=torch.as_tensor(eps_value, device=device, dtype=dtype),
            )

        out = dfl(X_all, y_true_real, **kwargs)

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

def run_DFL_non_parametric_separate(
    args,
    problem_mode,          # "so" / "dro"
    optimization_mode,     # "single" / "multi"
    quantiles,
    data_path,
    target_nodes,
    models_s,
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

    eval_splits = tuple(s.lower() for s in (eval_splits or ("test",)))
    if not all(s in ("train", "test") for s in eval_splits):
        raise ValueError(f"eval_splits must be subset of ('train','test'), got {eval_splits}")

    if eval_flags is None:
        eval_flags = (True, True, True, True, True)
    if len(eval_flags) != 5:
        raise ValueError(
            "eval_flags must be a 5-tuple: "
            "(det_before, random_before, stage2_after, stage3_before, stage3_after)"
        )
    eval_det_before, eval_random_before, eval_stage2_after, eval_stage3_before, eval_stage3_after = map(bool, eval_flags)

    # ---- 规范化 eval config ----
    args.eval_mode = str(getattr(args, "eval_mode", "soft")).lower()
    args.avoid_rand_duplicate = bool(getattr(args, "avoid_rand_duplicate", False))

    print("\n[ScenarioFilter eval config]")
    print("eval_mode =", args.eval_mode)
    print("avoid_rand_duplicate =", args.avoid_rand_duplicate)

    # ---- 与参考代码一致：统一 train/test batch size ----
    train_bs = int(
        getattr(
            args,
            "train_batch_size",
            getattr(args, "dfl_batch_size", getattr(args, "batch_size", 8))
        )
    )
    test_bs = int(getattr(args, "test_batch_size", getattr(args, "batch_size", 8)))

    args.train_batch_size = train_bs
    args.test_batch_size = test_bs

    print("args.train_batch_size =", train_bs)
    print("args.test_batch_size =", test_bs)

    run_stage2 = bool(getattr(args, "run_stage2", True))
    run_stage3 = bool(getattr(args, "run_stage3", True))
    reused_stage2 = stage2_artifact is not None
    multi = is_multi(optimization_mode)

    print(
        "run_stage2:", run_stage2,
        "run_stage3:", run_stage3,
        "reused_stage2:", reused_stage2,
        "multi:", multi
    )

    # -------- pick manager/optnet/DFL --------
    if multi:
        SO_Manager = IEEE14_Reserve_SO_Manager_MultiNode
        DRO_Manager = IEEE14_Reserve_DRO_Manager_MultiNode
        DET_Manager = IEEE14_Reserve_Deterministic_DA_Manager_MultiNode

        SAA_DA_OptNet = MultiNode_Reserve_SAA_DA_OptNet
        DRO_DA_OptNet = MultiNode_Reserve_DRO_DA_OptNet
        DET_DA_OptNet = MultiNode_Reserve_Deterministic_DA_OptNet

        RT_OptNet = MultiNode_Reserve_RT_OptNet
        DFL_SO_Class = DFL_model_nonparametric_MultiNode
        DFL_DRO_Class = DFL_model_nonparametric_DRO_MultiNode
        DET_Class = DFL_model_nonparametric_Deterministic_MultiNode
    else:
        SO_Manager = IEEE14_Reserve_SO_Manager_SingleNode
        DRO_Manager = IEEE14_Reserve_DRO_Manager_SingleNode
        DET_Manager = IEEE14_Reserve_Deterministic_DA_Manager_SingleNode

        SAA_DA_OptNet = SingleNode_Reserve_SAA_DA_OptNet
        DRO_DA_OptNet = SingleNode_Reserve_DRO_DA_OptNet
        DET_DA_OptNet = SingleNode_Reserve_Deterministic_DA_OptNet

        RT_OptNet = SingleNode_Reserve_RT_OptNet
        DFL_SO_Class = DFL_model_nonparametric_SingleNode
        DFL_DRO_Class = DFL_model_nonparametric_DRO_SingleNode
        DET_Class = DFL_model_nonparametric_Deterministic_SingleNode

    # -------- filter config --------
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

    train_logs_stage2 = []
    train_logs_stage3 = []
    time_stage2 = 0.0
    time_stage3 = 0.0

    set_seed(seed)
    t0_total = time.time()

    # ---------- dataset ----------
    if reused_stage2:
        train_data = stage2_artifact["train_data"]
        test_data = stage2_artifact["test_data"]
    else:
        def make_dataset(flag):
            return Combined_dataset_non_parametric(
                data_path=data_path,
                target_nodes=target_nodes,
                flag=flag,
                train_length=8760,
                val_ratio=0.2,
                seed=42,
                y_real=True,
            )

        train_data = make_dataset("train")
        test_data = make_dataset("test")

    # -------- predictor builder --------
    def build_predictor():
        return Multi_nonparametric_quantile_predictor(
            node_models=copy.deepcopy(models_s),
            scaler_y_map=train_data.scaler_y_map,
            node_order=target_nodes,
            quantiles=quantiles,
        ).to(device)

    # -------- build core --------
    if is_so(problem_mode):
        mgr_local = SO_Manager(args)
        optnet_DA = SAA_DA_OptNet(
            mgr=mgr_local,
            N_scen=args.N_scen,
            T=24,
        ).to(device)
        DFLClass = DFL_SO_Class
        mode_str = "SO"
    else:
        mgr_local = DRO_Manager(args)
        optnet_DA = DRO_DA_OptNet(
            mgr=mgr_local,
            N_scen=args.N_scen,
            T=24,
        ).to(device)
        DFLClass = DFL_DRO_Class
        mode_str = "DRO"

    mgr_det = DET_Manager(args)
    optnet_DA_det = DET_DA_OptNet(mgr=mgr_det, T=24).to(device)
    optnet_RT = RT_OptNet(mgr=mgr_local, T=24).to(device)

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

    def make_det_model(predictor, scenario_filter):
        return DET_Class(
            mgr=mgr_det,
            optnet_DA=optnet_DA_det,
            optnet_RT=optnet_RT,
            predictor=predictor,
            quantiles=quantiles,
            scenario_filter=scenario_filter,
            n_scen=args.N_scen,
            clamp_min=float(getattr(args, "clamp_min", 0.0)),
            solver=getattr(args, "solver", "ECOS"),
        ).to(device)

    def make_dfl_model(predictor, scenario_filter):
        return DFLClass(
            mgr=mgr_local,
            optnet_DA=optnet_DA,
            optnet_RT=optnet_RT,
            predictor=predictor,
            quantiles=quantiles,
            scenario_filter=scenario_filter,
            n_scen=args.N_scen,
            clamp_min=float(getattr(args, "clamp_min", 0.0)),
            solver=getattr(args, "solver", "ECOS"),
        ).to(device)

    def rebuild_model_like(src_model, predictor_builder, filter_builder, cls_builder):
        """
        Rebuild a fresh model using:
        - predictor weights from src_model
        - scenario_filter weights from src_model if compatible
        """
        if src_model is None:
            return None

        new_predictor = predictor_builder()
        new_predictor.load_state_dict(copy.deepcopy(src_model.predictor.state_dict()))

        new_filter = filter_builder()
        if hasattr(src_model.scenario_filter, "state_dict") and hasattr(new_filter, "load_state_dict"):
            try:
                new_filter.load_state_dict(copy.deepcopy(src_model.scenario_filter.state_dict()))
            except Exception:
                pass

        return cls_builder(new_predictor, new_filter)

    predictor_before = build_predictor()
    predictor_train = build_predictor()

    dfl_det_before = make_det_model(
        predictor=copy.deepcopy(predictor_before),
        scenario_filter=copy.deepcopy(make_filter()),
    )

    dfl_before = make_dfl_model(
        predictor=predictor_before,
        scenario_filter=copy.deepcopy(make_filter()),
    )

    dfl = make_dfl_model(
        predictor=predictor_train,
        scenario_filter=make_filter(),
    )

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

    # ---------- pre-stage eval ----------
    if not reused_stage2:
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

    # =========================
    # Stage A source selection
    # =========================
    if reused_stage2:
        print("\n ---> Reusing passed stage2_artifact, skip Stage A and jump to Stage B")

        mode_str = stage2_artifact["mode_str"]

        det_before_eval = copy.deepcopy(stage2_artifact.get("det_before_eval", det_before_eval))
        stage1_eval = copy.deepcopy(stage2_artifact.get("stage1_eval", stage1_eval))
        stage2_eval = copy.deepcopy(stage2_artifact.get("stage2_eval", {}))

        train_logs_stage2 = copy.deepcopy(stage2_artifact.get("train_logs_stage2", []))
        time_stage2 = float(stage2_artifact.get("time_stage2_sec", 0.0))

        dfl_det_before = stage2_artifact.get("dfl_det_before", dfl_det_before)
        dfl_before = stage2_artifact.get("dfl_before", dfl_before)
        dfl = stage2_artifact["dfl_after_stage2"]

        dfl_after_stage2_snapshot = dfl
        dfl_det_before_snapshot = dfl_det_before
        dfl_before_snapshot = dfl_before

    elif run_stage2:
        args.epochs = int(getattr(args, "dfl_epochs", getattr(args, "epochs", 2)))
        args.dfl_lr = float(getattr(args, "dfl_lr", 1e-5))
        args.filter_lr = float(getattr(args, "filter_lr", 1e-3))

        print(
            f"\n ---> Stage A: train DFL with RANDOM selector "
            f"(epochs={args.epochs}, dfl_lr={args.dfl_lr}, train_bs={train_bs})"
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
            lambda_div=float(getattr(args, "lambda_div", 1e8)),
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

        dfl_after_stage2_snapshot = rebuild_model_like(
            src_model=dfl,
            predictor_builder=build_predictor,
            filter_builder=make_filter,
            cls_builder=make_dfl_model,
        )

        dfl_det_before_snapshot = rebuild_model_like(
            src_model=dfl_det_before,
            predictor_builder=build_predictor,
            filter_builder=make_filter,
            cls_builder=make_det_model,
        )

        dfl_before_snapshot = rebuild_model_like(
            src_model=dfl_before,
            predictor_builder=build_predictor,
            filter_builder=make_filter,
            cls_builder=make_dfl_model,
        )

    else:
        print("\n ---> [skip] Stage A disabled (run_stage2=False), use pre-Stage-A model and jump to Stage B")

        dfl = rebuild_model_like(
            src_model=dfl_before,
            predictor_builder=build_predictor,
            filter_builder=make_filter,
            cls_builder=make_dfl_model,
        )

        dfl_after_stage2_snapshot = dfl
        dfl_det_before_snapshot = rebuild_model_like(
            src_model=dfl_det_before,
            predictor_builder=build_predictor,
            filter_builder=make_filter,
            cls_builder=make_det_model,
        )
        dfl_before_snapshot = rebuild_model_like(
            src_model=dfl_before,
            predictor_builder=build_predictor,
            filter_builder=make_filter,
            cls_builder=make_dfl_model,
        )

        stage2_eval = {}

    # =========================
    # Stage B
    # =========================
    if run_stage3:
        args.epochs = int(getattr(args, "filter_epochs", 10))
        args.lr = float(getattr(args, "filter_lr", 1e-3))

        print(
            f"\n ---> Stage B: switch to learnable filter, train Filter only "
            f"(epochs={args.epochs}, lr={args.lr}, train_bs={train_bs})"
        )

        dfl.scenario_filter = make_filter()

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
            lambda_div=float(getattr(args, "lambda_div_stage3", 1e5)),
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
        "dfl_det_before": dfl_det_before_snapshot if dfl_det_before_snapshot is not None else None,
        "dfl_before": dfl_before_snapshot if dfl_before_snapshot is not None else None,
        "det_before_eval": copy.deepcopy(det_before_eval),
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
        "dfl_det_before": dfl_det_before,
        "dfl_before": dfl_before,
        "dfl_trained": dfl_trained,
        "train_logs_stage2": train_logs_stage2,
        "train_logs_stage3": train_logs_stage3,
        "time_stage2_sec": float(time_stage2),
        "time_stage3_sec": float(time_stage3),
        "train_time_sec_total": float(train_time_sec_total),
        "train_batch_size_used": int(train_bs),
        "test_batch_size_used": int(test_bs),
        "N_scen": int(args.N_scen),
        "S_full": int(args.S_full),
        "K_rand": int(K_rand),
        "eval_mode": str(getattr(args, "eval_mode", "soft")).lower(),
        "avoid_rand_duplicate": bool(getattr(args, "avoid_rand_duplicate", False)),
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

    # backward-compatible aliases
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



import copy
import torch
from utils import _parse_compare_method_names

def compare_scenario_filters_with_stage3_learned_non_parametric(
    base_result,
    args,
    problem_mode,
    optimization_mode,
    quantiles,
    models_s,
    target_nodes,
    device,
    eval_splits=("test",),
    method_names=None,
    seed=0,
    verbose=True,
):
    """
    Fair comparison under non-parametric protocol:

    - Stage 2 trains predictor
    - Stage 3 trains only the learned ScenarioFilter

    Comparison rule:
    - same Stage-2-trained predictor backbone for all methods
    - learned: use Stage-3-trained scenario_filter
    - random/kmeans/kmedoids/hierarchical: replace scenario_filter with baseline filter

    IMPORTANT:
    We do NOT deepcopy the whole model because optnet may be inside.
    We rebuild models safely using reconstruction.
    """
    method_names = _parse_compare_method_names(method_names)
    eval_splits = tuple(s.lower() for s in (eval_splits or ("test",)))
    if not all(s in ("train", "test") for s in eval_splits):
        raise ValueError(f"eval_splits must be subset of ('train','test'), got {eval_splits}")

    if "stage2_artifact" not in base_result:
        raise ValueError("base_result does not contain 'stage2_artifact'.")

    stage2_artifact = base_result["stage2_artifact"]
    train_data = stage2_artifact["train_data"]
    test_data = stage2_artifact["test_data"]

    stage2_model = stage2_artifact.get("dfl_after_stage2", None)
    trained_model = base_result.get("dfl_trained", None)

    if stage2_model is None:
        raise ValueError("stage2_artifact['dfl_after_stage2'] is missing.")
    if trained_model is None:
        raise ValueError("base_result['dfl_trained'] is missing.")

    def is_so(x):
        return str(x).lower() in {"so", "saa"}

    def is_multi(x):
        return str(x).lower() in {"multi", "multinode"}

    multi = is_multi(optimization_mode)

    # -------- managers / classes --------
    if multi:
        SO_Manager = IEEE14_Reserve_SO_Manager_MultiNode
        DRO_Manager = IEEE14_Reserve_DRO_Manager_MultiNode

        SAA_DA_OptNet = MultiNode_Reserve_SAA_DA_OptNet
        DRO_DA_OptNet = MultiNode_Reserve_DRO_DA_OptNet
        RT_OptNet = MultiNode_Reserve_RT_OptNet

        DFL_SO_Class = DFL_model_nonparametric_MultiNode
        DFL_DRO_Class = DFL_model_nonparametric_DRO_MultiNode
    else:
        SO_Manager = IEEE14_Reserve_SO_Manager_SingleNode
        DRO_Manager = IEEE14_Reserve_DRO_Manager_SingleNode

        SAA_DA_OptNet = SingleNode_Reserve_SAA_DA_OptNet
        DRO_DA_OptNet = SingleNode_Reserve_DRO_DA_OptNet
        RT_OptNet = SingleNode_Reserve_RT_OptNet

        DFL_SO_Class = DFL_model_nonparametric_SingleNode
        DFL_DRO_Class = DFL_model_nonparametric_DRO_SingleNode

    if is_so(problem_mode):
        mgr_local = SO_Manager(args)
        optnet_DA = SAA_DA_OptNet(mgr=mgr_local, N_scen=args.N_scen, T=24).to(device)
        DFLClass = DFL_SO_Class
    else:
        mgr_local = DRO_Manager(args)
        optnet_DA = DRO_DA_OptNet(mgr=mgr_local, N_scen=args.N_scen, T=24).to(device)
        DFLClass = DFL_DRO_Class

    optnet_RT = RT_OptNet(mgr=mgr_local, T=24).to(device)

    def build_predictor():
        return Multi_nonparametric_quantile_predictor(
            node_models=copy.deepcopy(models_s),
            scaler_y_map=train_data.scaler_y_map,
            node_order=target_nodes,
            quantiles=quantiles,
        ).to(device)

    def make_dfl_model(predictor, scenario_filter):
        return DFLClass(
            mgr=mgr_local,
            optnet_DA=optnet_DA,
            optnet_RT=optnet_RT,
            predictor=predictor,
            quantiles=quantiles,
            scenario_filter=scenario_filter,
            n_scen=args.N_scen,
            clamp_min=float(getattr(args, "clamp_min", 0.0)),
            solver=getattr(args, "solver", "ECOS"),
        ).to(device)

    def rebuild_with_specific_filter(src_model, scenario_filter):
        """
        Rebuild a fresh model using:
        - predictor weights from src_model
        - a provided scenario_filter
        """
        if src_model is None:
            return None

        new_predictor = build_predictor()
        new_predictor.load_state_dict(copy.deepcopy(src_model.predictor.state_dict()))

        return make_dfl_model(new_predictor, scenario_filter)

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

    def _mean_of(x):
        if x is None:
            return None
        if hasattr(x, "detach"):
            return float(x.detach().float().mean().item())
        return float(torch.as_tensor(x).float().mean().item())

    compare_result = {
        "backbone_predictor_source": "stage2_artifact['dfl_after_stage2']",
        "learned_filter_source": "base_result['dfl_trained'].scenario_filter",
        "method_names": list(method_names),
        "eval_splits": eval_splits,
        "details": {},
        "summary_mean": {},
    }

    if verbose:
        print("\n" + "=" * 90)
        print("[Compare Scenario Filters - NonParametric]")
        print("Predictor backbone : Stage-2 trained model")
        print("Learned filter     : Stage-3 trained scenario_filter")
        print("Methods            :", method_names)
        print("Eval splits        :", eval_splits)
        print("=" * 90)

    for name in method_names:
        if name == "learned":
            learned_filter = copy.deepcopy(trained_model.scenario_filter)
            model_i = rebuild_with_specific_filter(stage2_model, learned_filter)
        else:
            baseline_filter = build_scenario_baseline_filter(name, args, device)
            model_i = rebuild_with_specific_filter(stage2_model, baseline_filter)

        stage_tag = f"compare_stage2backbone_{name}"
        eval_out = _eval_on_splits(model_i, stage_tag)

        compare_result["details"][name] = eval_out

        mean_info = {}
        for split in eval_splits:
            key = f"{split}_losses_{stage_tag}"
            if key in eval_out:
                mean_info[split] = _mean_of(eval_out[key])

        compare_result["summary_mean"][name] = mean_info

        if verbose:
            print(f"\n[Compare][{name}]")
            for split in eval_splits:
                if split in mean_info:
                    print(f" {split} mean loss: {mean_info[split]:.6f}")

    return compare_result