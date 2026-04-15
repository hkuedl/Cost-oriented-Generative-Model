import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from data_loader import *
from combined_data_loader import *
from tqdm import tqdm
from Optimization_single_node import *
from Optimization_multi_node import *
from utils import *
from scenarios_reduce import *


class ANN_parametric(nn.Module):
    """
    Input:  X [B,24,F]
    Output: mu [B,24], sigma [B,24]  (sigma positive)
    """
    def __init__(self, input_dim, hidden=(256, 256, 128), dropout=0.0):
        super().__init__()
        h1, h2, h3 = hidden
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(24 * input_dim, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, h3),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h3, 24 * 2),
        )
        self.softplus = nn.Softplus()

    def forward(self, x):
        out = self.net(x)             # [B, 48]
        out = out.view(-1, 24, 2)     # [B,24,2]
        mu = out[:, :, 0]
        log_sigma = out[:, :, 1]
        sigma = self.softplus(log_sigma) + 1e-6
        return mu, sigma

def gaussian_nll(mu, sigma, y):
    return torch.mean(
        0.5 * torch.log(2 * torch.pi * sigma**2) + 0.5 * ((y - mu) ** 2) / (sigma**2)
    )


class Runner_parametric_24:
    def __init__(self, train_set, val_set, test_set, hidden=(256, 256, 128), lr=1e-3, device=None):
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set

        input_dim = train_set.X.shape[-1]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ANN_parametric(input_dim=input_dim, hidden=hidden).to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)

    def fit(self, epochs=500, batch_size=128, patience=50, best_path="best.pt", verbose=False):
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
                mu, sigma = self.model(X)
                #loss = gaussian_nll(mu, sigma, y)
                loss = gaussian_nll(mu, sigma, y) + 2.0 * F.mse_loss(mu, y)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                tr_losses.append(loss.item())

            self.model.eval()
            va_losses = []
            with torch.no_grad():
                for X, y in val_loader:
                    X = X.to(self.device)
                    y = y.to(self.device)
                    mu, sigma = self.model(X)
                    loss = gaussian_nll(mu, sigma, y)
                    va_losses.append(loss.item())

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


def run_parametric_benchmark(
    DataHandler,
    device=None,
    epochs=500,
    batch_size=128,
    target_nodes=None,
    lr=1e-3,
    hidden=(256, 256, 128),
    patience=50,
    ckpt_dir="../Model/Parametric/ckpt_nodes",
    verbose=True,
    data_path=None,
    train_length=4296,
    val_ratio=0.2,
    seed=42,   # NEW
):
    os.makedirs(ckpt_dir, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    if target_nodes is None or len(target_nodes) == 0:
        raise ValueError("target_nodes is empty.")

    models_s, handlers_s, pack_data_s = {}, {}, {}

    for node in target_nodes:
        if data_path is not None:
            train_set = DataHandler(data_path, node, flag="train", train_length=train_length, val_ratio=val_ratio, seed=seed)
            val_set   = DataHandler(data_path, node, flag="val",   train_length=train_length, val_ratio=val_ratio, seed=seed)
            test_set  = DataHandler(data_path, node, flag="test",  train_length=train_length, val_ratio=val_ratio, seed=seed)


        runner = Runner_parametric_24(train_set, val_set, test_set, hidden=hidden, lr=lr, device=device)
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
            "splits": {
                "train": {"X": train_set.X, "y": train_set.y},  # [D,24,F], [D,24]
                "val":   {"X": val_set.X,   "y": val_set.y},
                "test":  {"X": test_set.X,  "y": test_set.y},
            }
        }

    return models_s, handlers_s, pack_data_s





@torch.no_grad()
def sample_window_parametric_benchmark(
    models_s,
    handlers_s,
    pack_data_s,
    target_nodes,
    horizon_days=7,
    start_day=0,
    n_samples=50,
    seq_len=24,
    split="test",  # NEW
):
    node_names = list(target_nodes)
    N = len(node_names)
    if N == 0:
        raise ValueError("target_nodes is empty.")
    split = str(split)

    first = node_names[0]
    if "splits" not in pack_data_s[first] or split not in pack_data_s[first]["splits"]:
        raise KeyError(f"split={split} not found in pack_data_s[{first}]['splits']")

    X0 = pack_data_s[first]["splits"][split]["X"]
    D = int(X0.shape[0])

    if start_day < 0 or start_day >= D:
        raise ValueError(f"start_day={start_day} out of range, total_days({split})={D}")

    actual_days = min(int(horizon_days), D - int(start_day))
    L = actual_days * int(seq_len)

    Y_true = np.zeros((N, L), dtype=np.float32)
    Y_pred = np.zeros((int(n_samples), N, L), dtype=np.float32)

    for j, node in enumerate(node_names):
        model = models_s[node]
        handler = handlers_s[node][split]
        X_all = pack_data_s[node]["splits"][split]["X"]   # [D,24,F]
        y_all = pack_data_s[node]["splits"][split]["y"]   # [D,24]

        device = next(model.parameters()).device
        model.eval()

        truth_concat, preds_concat = [], []

        for d in range(start_day, start_day + actual_days):
            y_norm = y_all[d].detach().cpu().numpy()      # [24]
            y_real = handler.inverse_transform(y_norm)    # [24]
            truth_concat.append(y_real)

            X = X_all[d].unsqueeze(0).to(device)          # [1,24,F]
            mu, sigma = model(X)                           # [1,24], [1,24]
            mu = mu.squeeze(0).detach().cpu().numpy()
            sigma = sigma.squeeze(0).detach().cpu().numpy()

            eps = np.random.randn(int(n_samples), int(seq_len)).astype(np.float32)
            s_norm = mu[None, :] + sigma[None, :] * eps   # [S,24]
            s_real = handler.scaler_y.inverse_transform(
                s_norm.reshape(-1, 1)
            ).reshape(int(n_samples), int(seq_len))
            preds_concat.append(s_real)

        y_true = np.concatenate(truth_concat, axis=0)     # [L]
        y_pred = np.concatenate(preds_concat, axis=1)     # [S,L]

        Y_true[j, :L] = y_true[:L]
        Y_pred[:, j, :L] = y_pred[:, :L]

    return dict(
        mode="single_window_parametric",
        split=split,   # NEW
        target_nodes=node_names,
        start_day=int(start_day),
        horizon_days=int(actual_days),
        seq_len=int(seq_len),
        n_samples=int(n_samples),
        Y_true=Y_true,  # [N,L]
        Y_pred=Y_pred,  # [S,N,L]
    )


class Multi_parametric_predictor(nn.Module):
    def __init__(self, node_models: dict, scaler_y_map: dict, node_order: list,
                 dtype=torch.float64):
        super().__init__()
        self.node_order = list(node_order)
        self.models = nn.ModuleDict({k: node_models[k] for k in self.node_order})
        self.dtype = dtype

        means, scales = [], []
        for node in self.node_order:
            sc = scaler_y_map[node]  # sklearn StandardScaler, 已fit
            m = np.asarray(sc.mean_, dtype=np.float64).reshape(1)   # (1,)
            s = np.asarray(sc.std_, dtype=np.float64).reshape(1)    # (1,)
            means.append(torch.tensor(m))
            scales.append(torch.tensor(s))

        mean = torch.stack(means, dim=0).reshape(len(self.node_order), 1, 1)
        scale = torch.stack(scales, dim=0).reshape(len(self.node_order), 1, 1)

        self.register_buffer("y_mean_N11", mean.to(dtype))
        self.register_buffer("y_scale_N11", scale.to(dtype))

    def forward(self, X_all):
        """
        X_all: [B,11,24,F] float32
        returns: mu_real, sigma_real float64 [B,11,24]
        """
        B, N, T, F = X_all.shape
        device = X_all.device

        mu_out = torch.empty((B, N, T), dtype=self.dtype, device=device)
        sg_out = torch.empty((B, N, T), dtype=self.dtype, device=device)

        for j, node in enumerate(self.node_order):
            X_j = X_all[:, j, :, :]                 # [B,24,F]
            mu_nor, sig_nor = self.models[node](X_j)  # [B,24],[B,24] (normalized)

            mean = self.y_mean_N11[j]   # [1,1]
            scale = self.y_scale_N11[j] # [1,1]

            mu_out[:, j, :] = mu_nor.to(self.dtype) * scale + mean
            sg_out[:, j, :] = sig_nor.to(self.dtype) * scale

        return mu_out, sg_out

class DFL_model_parametric_Deterministic_SingleNode(nn.Module):
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
        dtype=torch.float64,
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
        self.dtype = dtype
        self.optnet_dtype = dtype

        b = torch.tensor([mgr.gen_info[g]["b"] for g in mgr.gen_bus_list], dtype=dtype)
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

    def _sample_scenarios_11T(self, mu_11T, sigma_11T, predictor_n_samples=None):
        T = mu_11T.shape[1]
        S = int(predictor_n_samples) if predictor_n_samples is not None else self.n_scen
        eps_samp = torch.randn((S, 11, T), dtype=self.dtype, device=mu_11T.device)
        scen = mu_11T.unsqueeze(0) + sigma_11T.unsqueeze(0) * eps_samp
        if self.clamp_min is not None:
            scen = torch.clamp(scen, min=float(self.clamp_min))
        return scen  # [S,11,T]

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
        rt_obj_true = sol_rt["rt_obj"][0]
        return stage1_energy_linear + stage1_reserve + rt_obj_true, sol_rt

    def _stack_filter_aux(self, aux_filter_list):
        if len(aux_filter_list) == 0:
            return None
        valid = [afi for afi in aux_filter_list if isinstance(afi, dict)]
        if len(valid) == 0:
            return None

        aux_filter = {}
        common_keys = set(valid[0].keys())
        for afi in valid[1:]:
            common_keys &= set(afi.keys())

        for k in common_keys:
            vals = [afi[k] for afi in valid]
            if not all(torch.is_tensor(v) for v in vals):
                continue

            v0 = vals[0]
            if k == "p":
                p_list = []
                for v in vals:
                    if v.dim() == 2:
                        v = v.unsqueeze(0)   # [1,K,M]
                    elif v.dim() == 3 and v.shape[0] == 1:
                        pass
                    else:
                        raise ValueError(f"single-node aux_filter['p'] must be [K,M] or [1,K,M], got {tuple(v.shape)}")
                    p_list.append(v)
                aux_filter[k] = torch.cat(p_list, dim=0)  # [B,K,M]
            else:
                same_shape = all(tuple(v.shape) == tuple(v0.shape) for v in vals)
                if same_shape:
                    aux_filter[k] = torch.stack(vals, dim=0)

        return aux_filter if len(aux_filter) > 0 else None

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
        filter_kwargs = dict(filter_kwargs or {})
        device = X_all.device

        if y_true_real is None:
            raise ValueError("Need y_true_real for realized cost.")

        mu_real, sigma_real = self.predictor(X_all)  # [B,11,T]
        mu_real = mu_real.to(device=device, dtype=self.dtype)
        sigma_real = sigma_real.to(device=device, dtype=self.dtype)

        T = mu_real.shape[-1]
        y_true_real = self._canon_y_true_11T(y_true_real, T=T).to(device=device, dtype=self.dtype)

        B = mu_real.shape[0]
        loss_list = []
        aux_filter_list = []

        aux = None
        if return_aux:
            aux = {
                "forecast_sys": [],
                "omega_true": [],
                "det_obj": [],
                "realized_total": [],
                "rt_obj_true": [],
            }

        for b in range(B):
            mu_11T = mu_real[b]
            sig_11T = sigma_real[b]

            aux_filter = None
            if self.scenario_filter is not None:
                scen_11T = self._sample_scenarios_11T(mu_11T, sig_11T, predictor_n_samples)
                scen_11T_f, aux_filter = self.scenario_filter(
                    scen_11T.unsqueeze(1), is_train=self.training, **filter_kwargs
                )
                _ = scen_11T_f.squeeze(1).to(device=device, dtype=self.dtype)
            aux_filter_list.append(aux_filter)

            forecast_sys_T = mu_11T.sum(dim=0)  # [T]

            sol_da = self.optnet_DA(
                forecast_sys_T.unsqueeze(0),
                solver=solver,
            )
            P_DA = sol_da["P_DA"][0]
            R_up = sol_da["R_up"][0]
            R_dn = sol_da["R_dn"][0]

            omega_true_T = y_true_real[b].sum(dim=0) - forecast_sys_T
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

        loss_vec = torch.stack(loss_list, dim=0)

        if return_aux:
            for k in aux:
                aux[k] = torch.stack(aux[k], dim=0)

        aux_filter = self._stack_filter_aux(aux_filter_list) if return_filter_aux else None

        if return_aux and not return_filter_aux:
            return loss_vec, aux
        if return_filter_aux and not return_aux:
            return loss_vec, aux_filter
        if return_aux and return_filter_aux:
            return loss_vec, aux, aux_filter
        return loss_vec


class DFL_model_parametric_Deterministic_MultiNode(nn.Module):
    def __init__(
        self,
        mgr,
        optnet_DA,
        optnet_RT,
        predictor,
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

        self.n_scen = int(n_scen)
        self.clamp_min = clamp_min
        self.solver = solver
        self.dtype = dtype
        self.optnet_dtype = dtype

        bus_sched_0 = [b - 1 for b in mgr.default_bus_order]
        self.register_buffer("bus_indices_11_to_14", torch.tensor(bus_sched_0, dtype=torch.long))
        self.register_buffer("b_G", torch.tensor([mgr.gen_info[g]["b"] for g in mgr.gen_bus_list], dtype=dtype))

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

    def _sample_scenarios(self, mu_11, sigma_11, predictor_n_samples=None):
        B, N, T = mu_11.shape
        if N != 11:
            raise ValueError(f"expected 11 nodes, got {N}")
        S = int(predictor_n_samples) if predictor_n_samples is not None else self.n_scen
        eps_samp = torch.randn((B, S, 11, T), device=mu_11.device, dtype=self.dtype)
        scen = mu_11.unsqueeze(1) + sigma_11.unsqueeze(1) * eps_samp
        if self.clamp_min is not None:
            scen = torch.clamp(scen, min=float(self.clamp_min))
        return scen

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
        filter_kwargs = dict(filter_kwargs or {})
        if y_true_real is None:
            raise ValueError("Need y_true_real [B,11,T] for realized cost.")

        device, dtype = X_all.device, self.dtype
        mu_11, sigma_11 = self.predictor(X_all)
        mu_11 = mu_11.to(device=device, dtype=dtype)
        sigma_11 = sigma_11.to(device=device, dtype=dtype)

        T = mu_11.shape[-1]
        y_true_real = self._canon_y_true_11T(y_true_real, T=T).to(device=device, dtype=dtype)

        aux_filter = None
        if self.scenario_filter is not None:
            scen_11 = self._sample_scenarios(mu_11, sigma_11, predictor_n_samples)  # [B,S,11,T]
            scen_11_SB = scen_11.permute(1, 0, 2, 3).contiguous()
            scen_11_SB, aux_filter = self.scenario_filter(
                scen_11_SB, is_train=self.training, **filter_kwargs
            )
            _ = scen_11_SB.permute(1, 0, 2, 3).contiguous().to(device=device, dtype=dtype)

        forecast11 = mu_11
        forecast14 = self._map_11_to_14(forecast11)

        sol_DA = self.optnet_DA(
            forecast14,
            solver=solver,
            return_cost=True,
        )
        P_DA = sol_DA["P_DA"].to(device=device, dtype=dtype)
        R_up = sol_DA["R_up"].to(device=device, dtype=dtype)
        R_dn = sol_DA["R_dn"].to(device=device, dtype=dtype)

        y_true_14 = self._map_11_to_14(y_true_real)

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
                "cost_energy": cost_energy.detach(),
                "cost_reserve": cost_reserve.detach(),
                "rt_obj_true": rt_obj_true.detach(),
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


class DFL_model_parametric_SingleNode(nn.Module):
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
        dtype=torch.float64,
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
        self.dtype = dtype
        self.optnet_dtype = dtype

        b = torch.tensor([mgr.gen_info[g]["b"] for g in mgr.gen_bus_list], dtype=dtype)
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

    def _sample_scenarios_11T(self, mu_11T, sigma_11T, predictor_n_samples=None):
        T = mu_11T.shape[1]
        S = int(predictor_n_samples) if predictor_n_samples is not None else self.n_scen
        eps = torch.randn((S, 11, T), dtype=self.dtype, device=mu_11T.device)
        scen = mu_11T.unsqueeze(0) + sigma_11T.unsqueeze(0) * eps
        if self.clamp_min is not None:
            scen = torch.clamp(scen, min=float(self.clamp_min))
        return scen  # [S,11,T]

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
        rt_obj_true = sol_rt["rt_obj"][0]
        return stage1_energy_linear + stage1_reserve + rt_obj_true, sol_rt

    def _stack_filter_aux(self, aux_filter_list):
        if len(aux_filter_list) == 0:
            return None
        valid = [afi for afi in aux_filter_list if isinstance(afi, dict)]
        if len(valid) == 0:
            return None

        aux_filter = {}
        common_keys = set(valid[0].keys())
        for afi in valid[1:]:
            common_keys &= set(afi.keys())

        for k in common_keys:
            vals = [afi[k] for afi in valid]
            if not all(torch.is_tensor(v) for v in vals):
                continue

            v0 = vals[0]
            if k == "p":
                p_list = []
                for v in vals:
                    if v.dim() == 2:
                        v = v.unsqueeze(0)
                    elif v.dim() == 3 and v.shape[0] == 1:
                        pass
                    else:
                        raise ValueError(f"single-node aux_filter['p'] must be [K,M] or [1,K,M], got {tuple(v.shape)}")
                    p_list.append(v)
                aux_filter[k] = torch.cat(p_list, dim=0)  # [B,K,M]
            else:
                same_shape = all(tuple(v.shape) == tuple(v0.shape) for v in vals)
                if same_shape:
                    aux_filter[k] = torch.stack(vals, dim=0)

        return aux_filter if len(aux_filter) > 0 else None

    def forward(
        self,
        X_all,
        y_true_real=None,
        solver=None,
        return_aux=False,
        return_filter_aux=False,
        predictor_n_samples=None,
        filter_kwargs=None,
    ):
        solver = solver or self.solver
        filter_kwargs = dict(filter_kwargs or {})
        device = X_all.device

        if y_true_real is None:
            raise ValueError("Need y_true_real for realized cost.")

        mu_real, sigma_real = self.predictor(X_all)  # [B,11,T]
        mu_real = mu_real.to(device=device, dtype=self.dtype)
        sigma_real = sigma_real.to(device=device, dtype=self.dtype)

        T = mu_real.shape[-1]
        y_true_real = self._canon_y_true_11T(y_true_real, T=T).to(device=device, dtype=self.dtype)

        B = mu_real.shape[0]
        loss_list = []
        aux_filter_list = []

        aux = None
        if return_aux:
            aux = {
                "forecast_sys": [],
                "omega_true": [],
                "realized_total": [],
            }

        for b in range(B):
            mu_11T = mu_real[b]
            sig_11T = sigma_real[b]

            scen_11T = self._sample_scenarios_11T(mu_11T, sig_11T, predictor_n_samples)  # [S,11,T]

            aux_filter = None
            if self.scenario_filter is not None:
                scen_11T_f, aux_filter = self.scenario_filter(
                    scen_11T.unsqueeze(1), is_train=self.training, **filter_kwargs
                )
                scen_11T = scen_11T_f.squeeze(1).to(device=device, dtype=self.dtype)
            aux_filter_list.append(aux_filter)

            forecast_sys_T = mu_11T.sum(dim=0)                             # [T]
            omega_saa_ST = scen_11T.sum(dim=1) - forecast_sys_T[None, :]   # [S,T]

            sol_da = self.optnet_DA(
                forecast_sys_T.unsqueeze(0),
                omega_saa_ST.unsqueeze(0),
                solver=solver,
                return_rt=False,
            )
            P_DA = sol_da["P_DA"][0]
            R_up = sol_da["R_up"][0]
            R_dn = sol_da["R_dn"][0]

            omega_true_T = y_true_real[b].sum(dim=0) - forecast_sys_T
            realized_total, _ = self._realized_total_cost(P_DA, R_up, R_dn, omega_true_T, solver=solver)
            loss_list.append(realized_total)

            if return_aux:
                aux["forecast_sys"].append(forecast_sys_T.detach())
                aux["omega_true"].append(omega_true_T.detach())
                aux["realized_total"].append(realized_total.detach())

        loss_vec = torch.stack(loss_list, dim=0)

        if return_aux:
            for k in aux:
                aux[k] = torch.stack(aux[k], dim=0)

        aux_filter = self._stack_filter_aux(aux_filter_list) if return_filter_aux else None

        if return_aux and not return_filter_aux:
            return loss_vec, aux
        if return_filter_aux and not return_aux:
            return loss_vec, aux_filter
        if return_aux and return_filter_aux:
            return loss_vec, aux, aux_filter
        return loss_vec


class DFL_model_parametric_DRO_SingleNode(nn.Module):
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
        dtype=torch.float64,
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
        self.dtype = dtype
        self.optnet_dtype = dtype

        b = torch.tensor([mgr.gen_info[g]["b"] for g in mgr.gen_bus_list], dtype=dtype)
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

    def _sample_scenarios_11T(self, mu_11T, sigma_11T, predictor_n_samples=None):
        T = mu_11T.shape[1]
        S = int(predictor_n_samples) if predictor_n_samples is not None else self.n_scen
        eps = torch.randn((S, 11, T), dtype=self.dtype, device=mu_11T.device)
        scen = mu_11T.unsqueeze(0) + sigma_11T.unsqueeze(0) * eps
        if self.clamp_min is not None:
            scen = torch.clamp(scen, min=float(self.clamp_min))
        return scen

    def _realized_total_cost(self, P_DA_GT, R_up_GT, R_dn_GT, omega_true_T, solver):
        bG = self.b_G.to(device=P_DA_GT.device, dtype=self.dtype)
        stage1_energy = (P_DA_GT * bG[:, None]).sum()
        stage1_res = (
            (R_up_GT * (self.res_up_ratio * bG)[:, None]).sum()
            + (R_dn_GT * (self.res_dn_ratio * bG)[:, None]).sum()
        )
        sol_rt = self.optnet_RT(
            R_up_GT.unsqueeze(0),
            R_dn_GT.unsqueeze(0),
            omega_true_T.unsqueeze(0),
            solver=solver,
        )
        return stage1_energy + stage1_res + sol_rt["rt_obj"][0], sol_rt

    def _stack_filter_aux(self, aux_filter_list):
        if len(aux_filter_list) == 0:
            return None
        valid = [afi for afi in aux_filter_list if isinstance(afi, dict)]
        if len(valid) == 0:
            return None

        aux_filter = {}
        common_keys = set(valid[0].keys())
        for afi in valid[1:]:
            common_keys &= set(afi.keys())

        for k in common_keys:
            vals = [afi[k] for afi in valid]
            if not all(torch.is_tensor(v) for v in vals):
                continue

            v0 = vals[0]
            if k == "p":
                p_list = []
                for v in vals:
                    if v.dim() == 2:
                        v = v.unsqueeze(0)
                    elif v.dim() == 3 and v.shape[0] == 1:
                        pass
                    else:
                        raise ValueError(f"single-node aux_filter['p'] must be [K,M] or [1,K,M], got {tuple(v.shape)}")
                    p_list.append(v)
                aux_filter[k] = torch.cat(p_list, dim=0)  # [B,K,M]
            else:
                same_shape = all(tuple(v.shape) == tuple(v0.shape) for v in vals)
                if same_shape:
                    aux_filter[k] = torch.stack(vals, dim=0)

        return aux_filter if len(aux_filter) > 0 else None

    def forward(
        self,
        X_all,
        y_true_real,
        hourly_load_min_sys,
        hourly_load_max_sys,
        eps,
        solver=None,
        return_aux=False,
        return_filter_aux=False,
        predictor_n_samples=None,
        filter_kwargs=None,
    ):
        solver = solver or self.solver
        filter_kwargs = dict(filter_kwargs or {})
        device, dtype = X_all.device, self.dtype

        mu_real, sigma_real = self.predictor(X_all)
        mu_real = mu_real.to(device=device, dtype=dtype)
        sigma_real = sigma_real.to(device=device, dtype=dtype)

        B, _, T = mu_real.shape
        y_true_real = self._canon_y_true_11T(y_true_real, T=T).to(device=device, dtype=dtype)

        Lmin_sys = self._to_BT(hourly_load_min_sys, B, device, dtype)
        Lmax_sys = self._to_BT(hourly_load_max_sys, B, device, dtype)
        eps_t = torch.as_tensor(eps, device=device, dtype=dtype)
        if eps_t.ndim == 0:
            eps_t = eps_t.view(1).expand(B)

        loss_list = []
        aux_filter_list = []
        aux = {"realized_total": []} if return_aux else None

        for b in range(B):
            mu_11T = mu_real[b]
            sig_11T = sigma_real[b]

            scen_11T = self._sample_scenarios_11T(mu_11T, sig_11T, predictor_n_samples)

            aux_filter = None
            if self.scenario_filter is not None:
                scen_11T_f, aux_filter = self.scenario_filter(
                    scen_11T.unsqueeze(1), is_train=self.training, **filter_kwargs
                )
                scen_11T = scen_11T_f.squeeze(1).to(device=device, dtype=dtype)
            aux_filter_list.append(aux_filter)

            scen_sys_ST = scen_11T.sum(dim=1)
            forecast_sys_T = mu_11T.sum(dim=0)
            omega_scen_ST = scen_sys_ST - forecast_sys_T[None, :]

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
            P_DA = sol_da["P_DA"][0]
            R_up = sol_da["R_up"][0]
            R_dn = sol_da["R_dn"][0]

            omega_true_T = y_true_real[b].sum(dim=0) - forecast_sys_T
            realized_total, _ = self._realized_total_cost(P_DA, R_up, R_dn, omega_true_T, solver=solver)
            loss_list.append(realized_total)

            if return_aux:
                aux["realized_total"].append(realized_total.detach())

        loss_vec = torch.stack(loss_list, dim=0)
        if return_aux:
            aux["realized_total"] = torch.stack(aux["realized_total"], dim=0)

        aux_filter = self._stack_filter_aux(aux_filter_list) if return_filter_aux else None

        if return_aux and not return_filter_aux:
            return loss_vec, aux
        if return_filter_aux and not return_aux:
            return loss_vec, aux_filter
        if return_aux and return_filter_aux:
            return loss_vec, aux, aux_filter
        return loss_vec

class DFL_model_parametric_MultiNode(nn.Module):
    def __init__(
        self,
        mgr,
        optnet_DA,
        optnet_RT,
        predictor,
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

        self.n_scen = int(n_scen)
        self.clamp_min = clamp_min
        self.solver = solver
        self.dtype = dtype
        self.optnet_dtype = dtype

        bus_sched_0 = [b - 1 for b in mgr.default_bus_order]
        self.register_buffer("bus_indices_11_to_14", torch.tensor(bus_sched_0, dtype=torch.long))
        self.register_buffer("b_G", torch.tensor([mgr.gen_info[g]["b"] for g in mgr.gen_bus_list], dtype=dtype))

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

    def _sample_scenarios(self, mu_11, sigma_11, predictor_n_samples=None):
        B, N, T = mu_11.shape
        if N != 11:
            raise ValueError(f"expected 11 nodes, got {N}")
        S = int(predictor_n_samples) if predictor_n_samples is not None else self.n_scen
        eps = torch.randn((B, S, 11, T), device=mu_11.device, dtype=self.dtype)
        scen = mu_11.unsqueeze(1) + sigma_11.unsqueeze(1) * eps
        if self.clamp_min is not None:
            scen = torch.clamp(scen, min=float(self.clamp_min))
        return scen

    def forward(
        self,
        X_all,
        y_true_real=None,
        solver=None,
        return_aux=False,
        return_filter_aux=False,
        predictor_n_samples=None,
        filter_kwargs=None,
    ):
        solver = solver or self.solver
        filter_kwargs = dict(filter_kwargs or {})
        if y_true_real is None:
            raise ValueError("Need y_true_real [B,11,T] for realized cost.")

        device, dtype = X_all.device, self.dtype
        mu_11, sigma_11 = self.predictor(X_all)
        mu_11 = mu_11.to(device=device, dtype=dtype)
        sigma_11 = sigma_11.to(device=device, dtype=dtype)

        T = mu_11.shape[-1]
        y_true_real = self._canon_y_true_11T(y_true_real, T=T).to(device=device, dtype=dtype)

        scen_11 = self._sample_scenarios(mu_11, sigma_11, predictor_n_samples)  # [B,S,11,T]

        aux_filter = None
        if self.scenario_filter is not None:
            scen_11_SB = scen_11.permute(1, 0, 2, 3).contiguous()
            scen_11_SB, aux_filter = self.scenario_filter(scen_11_SB, is_train=self.training, **filter_kwargs)
            scen_11 = scen_11_SB.permute(1, 0, 2, 3).contiguous().to(device=device, dtype=dtype)

        forecast14 = self._map_11_to_14(mu_11)
        scen14 = self._map_11_to_14(scen_11)
        omega14 = scen14 - forecast14.unsqueeze(1)

        sol_DA = self.optnet_DA(
            forecast14, omega14, solver=solver, return_rt=False, return_cost=True
        )
        P_DA = sol_DA["P_DA"].to(device=device, dtype=dtype)
        R_up = sol_DA["R_up"].to(device=device, dtype=dtype)
        R_dn = sol_DA["R_dn"].to(device=device, dtype=dtype)

        y_true_14 = self._map_11_to_14(y_true_real)
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

class DFL_model_parametric_DRO_MultiNode(nn.Module):
    def __init__(
        self,
        mgr,
        optnet_DA,
        optnet_RT,
        predictor,
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

        self.n_scen = int(n_scen)
        self.clamp_min = clamp_min
        self.solver = solver
        self.dtype = dtype
        self.optnet_dtype = dtype

        bus_sched_0 = [b - 1 for b in mgr.default_bus_order]
        self.register_buffer("bus_indices_11_to_14", torch.tensor(bus_sched_0, dtype=torch.long))
        self.register_buffer("b_G", torch.tensor([mgr.gen_info[g]["b"] for g in mgr.gen_bus_list], dtype=dtype))
        self.res_up_ratio = float(getattr(mgr.args, "reserve_up_ratio", 0.05))
        self.res_dn_ratio = float(getattr(mgr.args, "reserve_dn_ratio", 0.02))

    def _canon_y_true_11T(self, y_true, T=24):
        if y_true.dim() != 3:
            raise ValueError(f"y_true bad shape {tuple(y_true.shape)}")
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

    def _sample_scenarios(self, mu_11, sigma_11, predictor_n_samples=None):
        B, N, T = mu_11.shape
        if N != 11:
            raise ValueError(f"expected 11 nodes, got {N}")
        S = int(predictor_n_samples) if predictor_n_samples is not None else self.n_scen
        eps_noise = torch.randn((B, S, 11, T), device=mu_11.device, dtype=mu_11.dtype)
        scen = mu_11.unsqueeze(1) + sigma_11.unsqueeze(1) * eps_noise
        if self.clamp_min is not None:
            scen = torch.clamp(scen, min=float(self.clamp_min))
        return scen

    def forward(
        self,
        X_all,
        y_true_real,
        hourly_load_min_sys,
        hourly_load_max_sys,
        eps,
        solver=None,
        return_aux=False,
        return_filter_aux=False,
        predictor_n_samples=None,
        filter_kwargs=None,
    ):
        solver = solver or self.solver
        filter_kwargs = dict(filter_kwargs or {})
        device, dtype = X_all.device, self.dtype

        mu_11, sigma_11 = self.predictor(X_all)
        mu_11 = mu_11.to(device=device, dtype=dtype)
        sigma_11 = sigma_11.to(device=device, dtype=dtype)

        B, _, T = mu_11.shape
        y_true_real = self._canon_y_true_11T(y_true_real, T=T).to(device=device, dtype=dtype)

        scen_11 = self._sample_scenarios(mu_11, sigma_11, predictor_n_samples=predictor_n_samples)  # [B,S,11,T]

        aux_filter = None
        if self.scenario_filter is not None:
            scen_11_SB = scen_11.permute(1, 0, 2, 3).contiguous()
            scen_11_SB, aux_filter = self.scenario_filter(scen_11_SB, is_train=self.training, **filter_kwargs)
            scen_11 = scen_11_SB.permute(1, 0, 2, 3).contiguous().to(device=device, dtype=dtype)

        forecast14 = self._map_11_to_14(mu_11)
        scen14 = self._map_11_to_14(scen_11)
        omega14 = scen14 - forecast14.unsqueeze(1)

        Lmin14 = torch.as_tensor(hourly_load_min_sys, device=device, dtype=dtype)
        Lmax14 = torch.as_tensor(hourly_load_max_sys, device=device, dtype=dtype)
        if Lmin14.ndim == 2:
            Lmin14 = Lmin14.unsqueeze(0).expand(B, -1, -1).contiguous()
        if Lmax14.ndim == 2:
            Lmax14 = Lmax14.unsqueeze(0).expand(B, -1, -1).contiguous()

        if Lmin14.shape != forecast14.shape or Lmax14.shape != forecast14.shape:
            raise ValueError("Lmin/Lmax shape must match [B,14,T]")

        om_min14 = Lmin14 - forecast14
        om_max14 = Lmax14 - forecast14
        om_min14, om_max14 = torch.minimum(om_min14, om_max14), torch.maximum(om_min14, om_max14)

        eps_t = torch.as_tensor(eps, device=device, dtype=dtype)
        if eps_t.ndim == 0:
            eps_t = eps_t.view(1).expand(B)

        sol_DA = self.optnet_DA(
            forecast14, omega14, om_min14, om_max14, eps_t, solver=solver
        )
        P_DA = sol_DA["P_DA"].to(device=device, dtype=dtype)
        R_up = sol_DA["R_up"].to(device=device, dtype=dtype)
        R_dn = sol_DA["R_dn"].to(device=device, dtype=dtype)

        y_true_14 = self._map_11_to_14(y_true_real)
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


def DFL_train(
    dfl,
    train_dataset,
    args,
    problem_mode="saa",
    train_mode="dfl",
    filter_kwargs=None,
    lambda_div=1e5,
):
    print("new version")
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

    print(lr_decay, filter_lr_decay, dfl_lr_decay)

    if str(problem_mode).lower() in ["saa", "so"]:
        Lmin = Lmax = eps_value = None
    else:
        Lmin = args.Lmin
        Lmax = args.Lmax
        eps_value = args.eps_value

    scenario_filter = getattr(dfl, "scenario_filter", None)
    has_filter = scenario_filter is not None

    # 当前 filter 是否可学习
    has_learnable_filter = (
        has_filter and any(p.requires_grad for p in scenario_filter.parameters())
    )

    if train_mode == "filter_only":
        # 只训练 filter
        for p in dfl.parameters():
            p.requires_grad = False

        if has_filter:
            for p in dfl.scenario_filter.parameters():
                p.requires_grad = True

        optim_params_filter = (
            [p for p in dfl.scenario_filter.parameters() if p.requires_grad]
            if has_filter else []
        )
        optim_params_predictor = []

        filter_lr = float(getattr(args, "lr", getattr(args, "filter_lr", 1e-3)))

        if len(optim_params_filter) == 0:
            raise ValueError(
                "train_mode='filter_only' but scenario_filter has no trainable parameters."
            )

        optim = torch.optim.Adam(
            [
                {"params": optim_params_filter, "lr": filter_lr, "name": "filter"},
            ],
            # weight_decay=float(getattr(args, "weight_decay", 0.0)),
        )

    else:
        # 默认 dfl 训练：至少训练 predictor
        for p in dfl.parameters():
            p.requires_grad = True

        optim_params_predictor = [
            p for p in dfl.predictor.parameters() if p.requires_grad
        ]
        predictor_lr = float(getattr(args, "dfl_lr", getattr(args, "lr", 1e-5)))

        # 只有当前 filter 真正可学习时才加进去
        optim_params_filter = (
            [p for p in dfl.scenario_filter.parameters() if p.requires_grad]
            if has_learnable_filter else []
        )
        filter_lr = float(getattr(args, "filter_lr", 1e-3))

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
            raise ValueError("No trainable parameters found for DFL training.")

        optim = torch.optim.Adam(
            param_groups,
            # weight_decay=float(getattr(args, "weight_decay", 0.0)),
        )

    # ---- debug prints ----
    print("train_mode =", train_mode)
    print(
        "scenario_filter type =",
        type(scenario_filter).__name__ if scenario_filter is not None else None
    )
    print("has_learnable_filter =", has_learnable_filter)
    print(
        "num filter params =",
        sum(p.numel() for p in optim_params_filter) if len(optim_params_filter) > 0 else 0
    )
    print(
        "num predictor params =",
        sum(p.numel() for p in optim_params_predictor) if len(optim_params_predictor) > 0 else 0
    )
    print("train_batch_size =", train_bs)

    if train_mode == "filter_only":
        print("filter_lr =", filter_lr)
    else:
        print("filter_lr =", filter_lr if len(optim_params_filter) > 0 else None)
        print("predictor_lr =", predictor_lr)

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
                predictor_n_samples=int(
                    getattr(args, "S_full", getattr(args, "N_scen", 50))
                ),
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

            current_lrs = [pg["lr"] for pg in optim.param_groups]
            pbar.set_postfix(
                Task=float(task_loss_val.detach().cpu()),
                Div=float(div_loss_val.detach().cpu()),
                Type=div_type,
                BS=train_bs,
                LR=current_lrs,
            )

        # ---- epoch end: lr decay by group name ----
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
            "lr": [pg["lr"] for pg in optim.param_groups],
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

    if filter_kwargs is None:
        filter_kwargs = {
            "tau_gumbel": float(getattr(args, "tau_mix", 1.0)),
            "eps_uniform": float(getattr(args, "eps_uniform", 0.1)),
        }

    test_bs = int(getattr(args, "test_batch_size", getattr(args, "batch_size", 8)))

    loader = DataLoader(
        test_dataset,
        batch_size=test_bs,
        shuffle=False,
        drop_last=False,
    )

    losses = []
    filter_aux_all = [] if return_filter_aux else None

    pbar = tqdm(loader, desc="Test", leave=False)
    for X_all, y_true_real in pbar:
        X_all = X_all.to(device).float()
        y_true_real = y_true_real.to(device).float()

        kwargs = dict(
            solver=getattr(args, "solver", None),
            return_aux=False,
            return_filter_aux=bool(return_filter_aux),
            predictor_n_samples=int(
                getattr(args, "predictor_n_samples", getattr(args, "S_full", getattr(args, "N_scen", 50)))
            ),
            filter_kwargs=filter_kwargs,
        )

        if str(problem_mode).lower() == "dro":
            dtype = getattr(dfl, "optnet_dtype", torch.float64)
            kwargs.update(
                hourly_load_min_sys=torch.as_tensor(args.Lmin, device=device, dtype=dtype),
                hourly_load_max_sys=torch.as_tensor(args.Lmax, device=device, dtype=dtype),
                eps=torch.as_tensor(args.eps_value, device=device, dtype=dtype),
            )

        set_seed(0)
        out = dfl(X_all, y_true_real, **kwargs)

        if return_filter_aux:
            loss_vec = out[0] if isinstance(out, (tuple, list)) else out
            aux_filter = out[-1] if isinstance(out, (tuple, list)) else None
            filter_aux_all.append(aux_filter)
        else:
            loss_vec = out[0] if isinstance(out, (tuple, list)) else out

        losses.append(loss_vec.detach().cpu())
        pbar.set_postfix(loss=float(loss_vec.mean().detach().cpu()))

    loss_all = torch.cat(losses, dim=0)
    if return_filter_aux:
        return loss_all, filter_aux_all
    return loss_all


def run_DFL_parametric_separate(
    args,
    problem_mode,
    optimization_mode,
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

    # ---- eval config ----
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

    # -------- managers / classes --------
    if multi:
        SO_Manager = IEEE14_Reserve_SO_Manager_MultiNode
        DRO_Manager = IEEE14_Reserve_DRO_Manager_MultiNode
        DET_Manager = IEEE14_Reserve_Deterministic_DA_Manager_MultiNode

        SAA_DA_OptNet = MultiNode_Reserve_SAA_DA_OptNet
        DRO_DA_OptNet = MultiNode_Reserve_DRO_DA_OptNet
        DET_DA_OptNet = MultiNode_Reserve_Deterministic_DA_OptNet

        RT_OptNet = MultiNode_Reserve_RT_OptNet
        DFL_SO_Class = DFL_model_parametric_MultiNode
        DFL_DRO_Class = DFL_model_parametric_DRO_MultiNode
        DET_Class = DFL_model_parametric_Deterministic_MultiNode
    else:
        SO_Manager = IEEE14_Reserve_SO_Manager_SingleNode
        DRO_Manager = IEEE14_Reserve_DRO_Manager_SingleNode
        DET_Manager = IEEE14_Reserve_Deterministic_DA_Manager_SingleNode

        SAA_DA_OptNet = SingleNode_Reserve_SAA_DA_OptNet
        DRO_DA_OptNet = SingleNode_Reserve_DRO_DA_OptNet
        DET_DA_OptNet = SingleNode_Reserve_Deterministic_DA_OptNet

        RT_OptNet = SingleNode_Reserve_RT_OptNet
        DFL_SO_Class = DFL_model_parametric_SingleNode
        DFL_DRO_Class = DFL_model_parametric_DRO_SingleNode
        DET_Class = DFL_model_parametric_Deterministic_SingleNode

    args.S_full = int(getattr(args, "S_full", max(50, int(getattr(args, "N_scen", 20)))))
    K = int(getattr(args, "N_scen", 20))
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
        train_data = Combined_dataset_parametric(
            data_path=data_path,
            target_nodes=target_nodes,
            flag="train",
            train_length=8760,
            val_ratio=0.2,
            seed=42,
            y_real=True,
        )
        test_data = Combined_dataset_parametric(
            data_path=data_path,
            target_nodes=target_nodes,
            flag="test",
            train_length=8760,
            val_ratio=0.2,
            seed=42,
            y_real=True,
        )

    # -------- builders --------
    def build_predictor():
        return Multi_parametric_predictor(
            node_models=copy.deepcopy(models_s),
            scaler_y_map=train_data.scaler_y_map,
            node_order=target_nodes,
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
        mode_call = "saa"
    else:
        mgr_local = DRO_Manager(args)
        optnet_DA = DRO_DA_OptNet(
            mgr=mgr_local,
            N_scen=args.N_scen,
            T=24,
        ).to(device)
        DFLClass = DFL_DRO_Class
        mode_str = "DRO"
        mode_call = "dro"

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
            scenario_filter=scenario_filter,
            n_scen=args.N_scen,
            clamp_min=getattr(args, "clamp_min", 0.0),
            solver="ECOS",
        ).to(device)

    def make_dfl_model(predictor, scenario_filter):
        return DFLClass(
            mgr=mgr_local,
            optnet_DA=optnet_DA,
            optnet_RT=optnet_RT,
            predictor=predictor,
            scenario_filter=scenario_filter,
            n_scen=args.N_scen,
            clamp_min=getattr(args, "clamp_min", 0.0),
            solver="ECOS",
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

    def _eval_on_splits(model, stage_tag):
        out = {}
        if "test" in eval_splits:
            set_seed(seed)
            out[f"test_losses_{stage_tag}"] = DFL_test(
                model,
                test_data,
                args,
                problem_mode=mode_call,
            )
        if "train" in eval_splits:
            set_seed(seed)
            out[f"train_losses_{stage_tag}"] = DFL_test(
                model,
                train_data,
                args,
                problem_mode=mode_call,
            )
        return out

    # ---------- pre-stage eval ----------
    if not reused_stage2:
        if eval_det_before:
            det_before_eval = _eval_on_splits(dfl_det_before, "deterministic_before")
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
            stage1_eval = _eval_on_splits(dfl_before, "stage1_after")
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
        mode_call = stage2_artifact["mode_call"]

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
            problem_mode=mode_call,
            train_mode="dfl",
            lambda_div=float(getattr(args, "lambda_div", 0.0)),
        )
        time_stage2 = time.time() - t0_s2
        print(f" ---> [Stage A done] time: {time_stage2:.2f} sec")

        if eval_stage2_after:
            stage2_eval = _eval_on_splits(dfl, "stage2_after")
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
            stage3_before_eval = _eval_on_splits(dfl, "stage3_before")
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
            problem_mode=mode_call,
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
        stage3_eval = _eval_on_splits(dfl_trained, "stage3_after")
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
        "mode_call": mode_call,
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





from utils import _parse_compare_method_names

def compare_scenario_filters_with_stage3_learned_parametric(
    base_result,
    args,
    problem_mode,
    optimization_mode,
    models_s,
    target_nodes,
    device,
    eval_splits=("test",),
    method_names=None,
    seed=0,
    verbose=True,
):
    """
    Fair comparison under parametric protocol:

    - Stage 2 trains predictor
    - Stage 3 trains only the learned ScenarioFilter

    Comparison rule:
    - same Stage-2-trained predictor backbone for all methods
    - learned: use Stage-3-trained scenario_filter
    - random/kmeans/kmedoids/hierarchical: replace scenario_filter with baseline filter

    IMPORTANT:
    We do NOT deepcopy the whole model because optnet may be inside.
    We rebuild models safely using reconstruction.

    Parameters
    ----------
    base_result : dict
        Output from run_DFL_parametric_separate(...)
    args : object
    problem_mode : str
    optimization_mode : str
    models_s : dict-like
        node models used to rebuild predictor
    target_nodes : list
    device : torch.device
    eval_splits : tuple
    method_names : list/tuple/str/None
    seed : int
    verbose : bool

    Returns
    -------
    compare_result : dict
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

        DFL_SO_Class = DFL_model_parametric_MultiNode
        DFL_DRO_Class = DFL_model_parametric_DRO_MultiNode
    else:
        SO_Manager = IEEE14_Reserve_SO_Manager_SingleNode
        DRO_Manager = IEEE14_Reserve_DRO_Manager_SingleNode

        SAA_DA_OptNet = SingleNode_Reserve_SAA_DA_OptNet
        DRO_DA_OptNet = SingleNode_Reserve_DRO_DA_OptNet
        RT_OptNet = SingleNode_Reserve_RT_OptNet

        DFL_SO_Class = DFL_model_parametric_SingleNode
        DFL_DRO_Class = DFL_model_parametric_DRO_SingleNode

    if is_so(problem_mode):
        mgr_local = SO_Manager(args)
        optnet_DA = SAA_DA_OptNet(mgr=mgr_local, N_scen=args.N_scen, T=24).to(device)
        DFLClass = DFL_SO_Class
        mode_call = "saa"
    else:
        mgr_local = DRO_Manager(args)
        optnet_DA = DRO_DA_OptNet(mgr=mgr_local, N_scen=args.N_scen, T=24).to(device)
        DFLClass = DFL_DRO_Class
        mode_call = "dro"

    optnet_RT = RT_OptNet(mgr=mgr_local, T=24).to(device)

    # -------- builders --------
    K = int(getattr(args, "N_scen", 50))
    K_rand = int(getattr(args, "K_rand", 0))

    def build_predictor():
        return Multi_parametric_predictor(
            node_models=copy.deepcopy(models_s),
            scaler_y_map=train_data.scaler_y_map,
            node_order=target_nodes,
        ).to(device)

    def make_dfl_model(predictor, scenario_filter):
        return DFLClass(
            mgr=mgr_local,
            optnet_DA=optnet_DA,
            optnet_RT=optnet_RT,
            predictor=predictor,
            scenario_filter=scenario_filter,
            n_scen=args.N_scen,
            clamp_min=getattr(args, "clamp_min", 0.0),
            solver="ECOS",
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
                problem_mode=mode_call,
            )
        if "train" in eval_splits:
            set_seed(seed)
            out[f"train_losses_{stage_tag}"] = DFL_test(
                model,
                train_data,
                args,
                problem_mode=mode_call,
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
        print("[Compare Scenario Filters - Parametric]")
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
