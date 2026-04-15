import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from VAE import *
from tqdm import tqdm
from Optimization_multi_node import *
from Optimization_single_node import *
from scenarios_reduce import *

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


class Runner_vae_joint:
    def __init__(
        self,
        train_set,
        val_set,
        test_set,
        n_nodes,
        z_global=8,
        z_node=4,
        hidden=256,
        dropout=0.1,
        lr=1e-4,
        beta_g=1e-4,
        beta_n=1e-3,
        beta_anneal=True,
        anneal_warmup=500,
        device=None,
    ):
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        self.n_nodes = int(n_nodes)

        self.z_global = int(z_global)
        self.z_node = int(z_node)
        self.hidden = int(hidden)
        self.dropout = float(dropout)
        self.lr = float(lr)

        self.beta_g = float(beta_g)
        self.beta_n = float(beta_n)
        self.beta_anneal = bool(beta_anneal)
        self.anneal_warmup = int(anneal_warmup)

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        input_dim = int(train_set.X.shape[-1])

        self.model = CVAE_Multi(
            input_dim=input_dim,
            n_nodes=self.n_nodes,
            z_global=self.z_global,
            z_node=self.z_node,
            hidden=self.hidden,
            dropout=self.dropout,
        ).to(self.device)

        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.global_step = 0
        self.best_z_temp = None

    def _anneal_factor(self):
        if (not self.beta_anneal) or self.anneal_warmup <= 0:
            return 1.0
        return float(min(1.0, self.global_step / float(self.anneal_warmup)))

    def _make_loader(self, ds, batch_size, shuffle):
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)

    def _step_batch(self, batch, train=True):
        if isinstance(batch, (tuple, list)):
            X, Y = batch[0], batch[1]
        elif isinstance(batch, dict):
            X, Y = batch["X"], batch.get("Y", batch.get("y"))
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")

        X = X.to(self.device).float()
        Y = Y.to(self.device).float()

        (
            mu_Y,
            mu_g_q, logvar_g_q, mu_g_p, logvar_g_p,
            mu_n_q, logvar_n_q, mu_n_p, logvar_n_p
        ) = self.model(X, Y)

        recon = torch.mean((mu_Y - Y) ** 2)

        kl_g = kl_diag_gaussians_per_sample(mu_g_q, logvar_g_q, mu_g_p, logvar_g_p, sum_dim=1)

        var_qn = torch.exp(logvar_n_q)
        var_pn = torch.exp(logvar_n_p)
        kl_n_elem = 0.5 * (
            logvar_n_p - logvar_n_q
            + (var_qn + (mu_n_q - mu_n_p) ** 2) / (var_pn + 1e-8)
            - 1.0
        )  # [B,N,zn]
        kl_n = kl_n_elem.sum(dim=(1, 2)).mean()

        af = self._anneal_factor()
        loss = recon + (af * self.beta_g) * kl_g + (af * self.beta_n) * kl_n

        if train:
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()
            self.global_step += 1

        return {
            "loss": float(loss.detach().cpu()),
            "recon": float(recon.detach().cpu()),
            "kl_g": float(kl_g.detach().cpu()),
            "kl_n": float(kl_n.detach().cpu()),
            "beta_g": float(self.beta_g * af),
            "beta_n": float(self.beta_n * af),
        }

    @torch.no_grad()
    def evaluate(self, batch_size=256):
        self.model.eval()
        loader = self._make_loader(self.val_set, batch_size=batch_size, shuffle=False)
        logs = {"loss": [], "recon": [], "kl_g": [], "kl_n": []}
        for batch in loader:
            out = self._step_batch(batch, train=False)
            for k in logs:
                logs[k].append(out[k])
        mean_logs = {k: float(np.mean(v)) for k, v in logs.items()}
        mean_logs["beta_g"] = float(self.beta_g * self._anneal_factor())
        mean_logs["beta_n"] = float(self.beta_n * self._anneal_factor())
        return mean_logs

    def fit(self, epochs=1000, batch_size=32, patience=50, best_path=None, verbose=True):
        best_path = best_path or "best_joint_hier_splitkl.pt"
        best_val = float("inf")
        best_state = None
        bad = 0

        train_loader = self._make_loader(self.train_set, batch_size=batch_size, shuffle=True)

        for ep in range(1, int(epochs) + 1):
            self.model.train()
            tr_logs = {"loss": [], "recon": [], "kl_g": [], "kl_n": []}
            for batch in train_loader:
                out = self._step_batch(batch, train=True)
                for k in tr_logs:
                    tr_logs[k].append(out[k])

            val_out = self.evaluate(batch_size=max(256, batch_size))
            tr_mean = {k: float(np.mean(v)) for k, v in tr_logs.items()}

            if verbose and (ep == 1 or ep % 10 == 0):
                print(
                    f"[ep {ep:04d}] "
                    f"train loss={tr_mean['loss']:.6f} recon={tr_mean['recon']:.6f} "
                    f"kl_g={tr_mean['kl_g']:.4f} kl_n={tr_mean['kl_n']:.4f} "
                    f"| val loss={val_out['loss']:.6f} recon={val_out['recon']:.6f} "
                    f"kl_g={val_out['kl_g']:.4f} kl_n={val_out['kl_n']:.4f} "
                    f"beta_g={val_out['beta_g']:.6g} beta_n={val_out['beta_n']:.6g}"
                )

            if val_out["loss"] < best_val - 1e-9:
                best_val = val_out["loss"]
                best_state = copy.deepcopy(self.model.state_dict())
                bad = 0
                if best_path is not None:
                    torch.save(best_state, best_path)
            else:
                bad += 1
                if bad >= int(patience):
                    if verbose:
                        print(f"Early stop at epoch {ep}, best_val={best_val:.6f}")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        return self

    def load_best(self, best_path):
        state = torch.load(best_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()
        return self

    @torch.no_grad()
    def select_best_ztemp_by_pinball(
        self,
        ztemps,
        quantiles=(0.1, 0.5, 0.9),
        n_samples=50,
        batch_size=64,
        verbose=True,
    ):
        self.model.eval()
        loader = self._make_loader(self.val_set, batch_size=batch_size, shuffle=False)

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

                if y_np.ndim == 3:  # [B,T,N]
                    out_np = ds.inverse_transform_y(y_np)
                    return torch.as_tensor(out_np, device=self.device, dtype=torch.float32)

                if y_np.ndim == 4:  # [S,B,T,N]
                    outs = [ds.inverse_transform_y(y_np[s]) for s in range(y_np.shape[0])]
                    out_np = np.stack(outs, axis=0)
                    return torch.as_tensor(out_np, device=self.device, dtype=torch.float32)

                raise ValueError(f"Unsupported y ndim={y_np.ndim}, shape={y_np.shape}")

        best_t, best_score = None, float("inf")
        table = []

        for t in list(ztemps):
            scores = []
            for batch in loader:
                if isinstance(batch, (tuple, list)):
                    X, Y = batch[0], batch[1]
                elif isinstance(batch, dict):
                    X, Y = batch["X"], batch.get("Y", batch.get("y"))
                else:
                    raise TypeError(f"Unsupported batch type: {type(batch)}")

                X = X.to(self.device).float()
                Y = Y.to(self.device).float()  # normalized

                samp = self.model.sample(X, n_samples=n_samples, z_temp=float(t))  # [S,B,24,N] normalized

                Y_real = inv_y(self.val_set, Y)
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
        if verbose:
            print(f"best z_temp = {best_t} | best val(real-scale) pinball = {best_score:.6f}")
        return best_t, best_score, table


def run_vae_joint(
    data_path,
    node_cols,
    device=None,
    epochs=1000,
    batch_size=32,
    lr=1e-4,
    patience=50,
    ckpt_dir="../Model/VAE/ckpt_joint_cvae",
    verbose=True,
    train_length=8760,
    val_ratio=0.2,
    seed=42,
    add_lag1=True,
    z_global=16,
    z_node=6,
    hidden=256,
    dropout=0.0,
    beta_g=1e-4,
    beta_n=1e-3,
    beta_anneal=True,
    anneal_warmup=50,
    # -------- z_temp tuning --------
    find_best_ztemp=True,
    z_temp_grid=(0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0),
    ztemp_search_quantiles=(0.1, 0.5, 0.9),
    ztemp_search_n_samples=50,
    ztemp_search_batch_size=64,
):
    os.makedirs(ckpt_dir, exist_ok=True)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(seed)

    train_set = Dataset_load_multi_node_vae(
        data_path=data_path,
        node_cols=node_cols,
        flag="train",
        train_length=train_length,
        val_ratio=val_ratio,
        seed=seed,
        add_lag1=add_lag1,
    )
    val_set = Dataset_load_multi_node_vae(
        data_path=data_path,
        node_cols=node_cols,
        flag="val",
        train_length=train_length,
        val_ratio=val_ratio,
        seed=seed,
        add_lag1=add_lag1,
    )
    test_set = Dataset_load_multi_node_vae(
        data_path=data_path,
        node_cols=node_cols,
        flag="test",
        train_length=train_length,
        val_ratio=val_ratio,
        seed=seed,
        add_lag1=add_lag1,
    )

    runner = Runner_vae_joint(
        train_set=train_set,
        val_set=val_set,
        test_set=test_set,
        n_nodes=len(node_cols),
        z_global=z_global,
        z_node=z_node,
        hidden=hidden,
        dropout=dropout,
        lr=lr,
        beta_g=beta_g,
        beta_n=beta_n,
        beta_anneal=beta_anneal,
        anneal_warmup=anneal_warmup,
        device=device,
    )

    best_path = os.path.join(
        ckpt_dir, f"best_joint_{len(node_cols)}nodes_zg{z_global}_zn{z_node}_splitkl.pt"
    )
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
            verbose=verbose,
        )
        try:
            runner.best_val_pinball_real = best_val_pinball_real
            runner.ztemp_table = ztemp_table
        except Exception:
            pass

    joint_model = runner.model.to(device).eval()

    # ---- NEW: store all splits for sampling ----
    def _mk_splits(train_set, val_set, test_set):
        return {
            "train": {"X": train_set.X, "Y": train_set.y},
            "val":   {"X": val_set.X,   "Y": val_set.y},
            "test":  {"X": test_set.X,  "Y": test_set.y},
        }

    models_m = {node: joint_model for node in node_cols}
    handlers_m = {node: {"train": train_set, "val": val_set, "test": test_set} for node in node_cols}
    pack_data_m = {node: {"splits": _mk_splits(train_set, val_set, test_set)} for node in node_cols}

    models_m["_JOINT_"] = joint_model
    handlers_m["_JOINT_"] = {"train": train_set, "val": val_set, "test": test_set}
    pack_data_m["_JOINT_"] = {
        "splits": _mk_splits(train_set, val_set, test_set),
        "node_cols": list(node_cols),
        "runner": runner,
        "best_path": best_path,
        "best_z_temp": (None if best_z_temp is None else float(best_z_temp)),
        "best_val_pinball_real": (None if best_val_pinball_real is None else float(best_val_pinball_real)),
        "ztemp_table": ztemp_table,
        "ztemp_quantiles": tuple(ztemp_search_quantiles),
    }

    return models_m, handlers_m, pack_data_m

@torch.no_grad()
def sample_window_vae_joint(
    models_m,
    handlers_m,
    pack_data_m,
    target_nodes,
    horizon_days=7,
    start_day=0,
    n_samples=50,
    seq_len=24,
    z_temp=3.0,   # None -> use tuned best_z_temp in pack
    split="test", # NEW: "train" | "val" | "test"
):
    model = models_m["_JOINT_"]
    pack = pack_data_m["_JOINT_"]

    if z_temp is None:
        z_temp = pack.get("best_z_temp", None)
        z_temp = 1.0 if z_temp is None else float(z_temp)

    split = str(split)
    if "splits" not in pack or split not in pack["splits"]:
        raise KeyError(f"pack_data_m['_JOINT_']['splits'] missing split={split}. "
                       f"Available: {list(pack.get('splits', {}).keys())}")

    X_all = pack["splits"][split]["X"]
    Y_all = pack["splits"][split]["Y"]

    # use the matching dataset object for inverse_transform_y
    handler = handlers_m["_JOINT_"][split]

    node_names = list(target_nodes)
    N = len(node_names)
    D = int(X_all.shape[0])

    if start_day < 0 or start_day >= D:
        raise ValueError(f"start_day={start_day} out of range, total_days({split})={D}")

    actual_days = min(int(horizon_days), D - int(start_day))
    L = actual_days * int(seq_len)

    Y_true = np.zeros((N, L), dtype=np.float32)
    Y_pred = np.zeros((int(n_samples), N, L), dtype=np.float32)

    device = next(model.parameters()).device
    model.eval()

    truth_concat, preds_concat = [], []
    for d in range(start_day, start_day + actual_days):
        y_day_norm = Y_all[d]                               # [T,N]
        y_day_real = handler.inverse_transform_y(y_day_norm) # [T,N]
        truth_concat.append(y_day_real)

        X = X_all[d].unsqueeze(0).to(device)  # [1,T,F]
        samp_norm = model.sample(X, n_samples=n_samples, z_temp=float(z_temp)).squeeze(1)  # [S,T,N]
        samp_real = handler.inverse_transform_y(samp_norm)   # [S,T,N]
        preds_concat.append(samp_real)

    y_true = np.concatenate(truth_concat, axis=0)  # [L,N]
    y_pred = np.concatenate(preds_concat, axis=1)  # [S,L,N]

    Y_true[:, :L] = y_true[:L].T
    Y_pred[:, :, :L] = np.transpose(y_pred[:, :L, :], (0, 2, 1))

    return dict(
        mode="single_window_cvae_joint_hier_splitkl_bestztemp",
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


class Joint_vae_predictor(nn.Module):
    def __init__(self, joint_model, dataset):
        super().__init__()

        self.model = joint_model["_JOINT_"]
        self.N = int(getattr(self.model, "n_nodes", None) or getattr(self.model, "N"))

        # --- build per-node mean/scale buffers, without changing dataloader ---
        if hasattr(dataset, "scaler_y") and hasattr(dataset.scaler_y, "mean_") and hasattr(dataset.scaler_y, "scale_"):
            mean = np.asarray(dataset.scaler_y.mean_, dtype=np.float32).reshape(self.N)
            scale = np.asarray(dataset.scaler_y.scale_, dtype=np.float32).reshape(self.N)
        elif hasattr(dataset, "scalers_y"):
            if len(dataset.scalers_y) != self.N:
                raise ValueError(f"len(dataset.scalers_y)={len(dataset.scalers_y)} != N={self.N}")
            mean = np.array([sc.mean_.item() for sc in dataset.scalers_y], dtype=np.float32)   # [N]
            scale = np.array([sc.scale_.item() for sc in dataset.scalers_y], dtype=np.float32) # [N]
        else:
            raise ValueError("Dataset must provide scaler_y (global) or scalers_y (per-node).")

        self.register_buffer("y_mean", torch.from_numpy(mean).view(self.N, 1, 1))     # [N,1,1]
        self.register_buffer("y_scale", torch.from_numpy(scale).view(self.N, 1, 1))   # [N,1,1]

    def inverse_y(self, y_norm):
        # y_norm: [B,N,24] or [B,24,N] -> [B,N,24]
        if y_norm.shape[1] == 24:
            y_norm = y_norm.permute(0, 2, 1).contiguous()
        mean = self.y_mean.to(y_norm.device).view(1, self.N, 1).to(y_norm.dtype)
        scale = self.y_scale.to(y_norm.device).view(1, self.N, 1).to(y_norm.dtype)
        return y_norm * scale + mean

    def sample(
        self,
        X_all,                   # [B,24,F]
        n_samples=50,
        z_temp=1.0,
        cpu_offload=False,
        grad_enabled=None,
        return_real_scale=True,
        **kwargs,                # 兜底：避免 unexpected kw
    ):
        if grad_enabled is None:
            grad_enabled = torch.is_grad_enabled()

        if grad_enabled and cpu_offload:
            raise ValueError("Training (grad_enabled=True) requires cpu_offload=False.")

        # 期望 CVAE_Multi.sample 返回 [S,B,24,N]
        Y = self.model.sample(
            X_all,
            n_samples=int(n_samples),
            z_temp=float(z_temp),
            grad_enabled=grad_enabled,
        )  # [S,B,24,N] (expected)

        # -> [S,B,N,24] 与 diffusion predictor 保持一致
        if Y.shape[-2] == 24 and Y.shape[-1] == self.N:
            Y = Y.permute(0, 1, 3, 2).contiguous()
        else:
            raise ValueError(f"Unexpected VAE sample shape {tuple(Y.shape)}; expected [S,B,24,N].")

        if cpu_offload:
            Y = Y.detach().cpu()

        if not return_real_scale:
            return Y

        mean = self.y_mean.to(Y.device).view(1, 1, self.N, 1).to(Y.dtype)
        scale = self.y_scale.to(Y.device).view(1, 1, self.N, 1).to(Y.dtype)
        return Y * scale + mean

    def forward(self, X_all, **kwargs):
        return self.sample(X_all, **kwargs)


class DFL_model_vae_Deterministic_SingleNode(nn.Module):
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
        solver = solver or self.solver
        device = X_all.device
        dtype = self.optnet_dtype
        predictor_kwargs = dict(predictor_kwargs or {})
        filter_kwargs = dict(filter_kwargs or {})

        S = int(predictor_n_samples) if predictor_n_samples is not None else self.n_scen

        # 用 predictor 采样均值作为 deterministic forecast
        Y_scen = self.predictor(
            X_all,
            n_samples=S,
            return_real_scale=True,
            grad_enabled=torch.is_grad_enabled(),
            **predictor_kwargs,
        ).to(device=device, dtype=dtype)

        if self.clamp_min is not None:
            Y_scen = torch.clamp(Y_scen, min=float(self.clamp_min))

        aux_filter = None
        if self.scenario_filter is not None:
            Y_scen, aux_filter = self.scenario_filter(
                Y_scen, is_train=self.training, **filter_kwargs
            )
            Y_scen = Y_scen.to(device=device, dtype=dtype)

        # deterministic forecast
        forecast_sys = Y_scen.sum(dim=2).mean(dim=0)   # [B,T]

        y_true_11 = self._canon_y_true_11T(y_true).to(device=device)
        y_true_real11 = self.predictor.inverse_y(y_true_11).to(device=device, dtype=dtype)
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
            }

        for b in range(B):
            sol_da = self.optnet_DA(
                forecast_sys[b].unsqueeze(0),
                solver=solver,
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
                obj = sol_da.get("obj", None)
                if isinstance(obj, torch.Tensor):
                    aux["det_obj"].append(obj[0].detach())
                else:
                    aux["det_obj"].append(torch.tensor(float("nan"), device=device, dtype=dtype))
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
            **predictor_kwargs,
        ).to(device=device, dtype=dtype)

        if self.clamp_min is not None:
            Y_scen = torch.clamp(Y_scen, min=float(self.clamp_min))

        aux_filter = None
        if self.scenario_filter is not None:
            Y_scen, aux_filter = self.scenario_filter(
                Y_scen, is_train=self.training, **filter_kwargs
            )
            Y_scen = Y_scen.to(device=device, dtype=dtype)

        forecast11 = Y_scen.mean(dim=0)   # [B,11,T]
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
                aux["det_obj"] = obj.detach()
        else:
            aux = None

        if return_aux and not return_filter_aux:
            return total_realized_cost, aux
        if return_filter_aux and not return_aux:
            return total_realized_cost, aux_filter
        if return_aux and return_filter_aux:
            return total_realized_cost, aux, aux_filter
        return total_realized_cost


class DFL_model_vae_SingleNode(nn.Module):
    def __init__(
        self,
        mgr,
        optnet_DA,
        optnet_RT,
        predictor,
        scenario_filter=None,   # <- added
        n_scen=50,
        clamp_min=0.0,          # <- added
        solver="ECOS",
    ):
        super().__init__()
        self.mgr = mgr
        self.optnet_DA = optnet_DA
        self.optnet_RT = optnet_RT
        self.predictor = predictor
        self.scenario_filter = scenario_filter   # <- added
        self.n_scen = int(n_scen)
        self.clamp_min = clamp_min               # <- added
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
        solver=None,
        return_aux=False,
        return_filter_aux=False,   # <- added
        predictor_n_samples=None,
        predictor_kwargs=None,
        filter_kwargs=None,        # <- added
    ):
        solver = solver or self.solver
        device = X_all.device
        predictor_kwargs = dict(predictor_kwargs or {})
        filter_kwargs = dict(filter_kwargs or {})

        S = int(predictor_n_samples) if predictor_n_samples is not None else self.n_scen

        # [S,B,11,T] real scale
        Y_scen = self.predictor(
            X_all,
            n_samples=S,
            return_real_scale=True,
            grad_enabled=torch.is_grad_enabled(),
            **predictor_kwargs,
        ).to(device=device, dtype=self.optnet_dtype)

        if self.clamp_min is not None:
            Y_scen = torch.clamp(Y_scen, min=float(self.clamp_min))

        # scenario filter hook
        aux_filter = None
        if self.scenario_filter is not None:
            Y_scen, aux_filter = self.scenario_filter(
                Y_scen, is_train=self.training, **filter_kwargs
            )
            Y_scen = Y_scen.to(device=device, dtype=self.optnet_dtype)

        scen_sys = Y_scen.sum(dim=2)                       # [S,B,T]
        forecast_sys = scen_sys.mean(dim=0)               # [B,T]
        omega_saa = (scen_sys - forecast_sys.unsqueeze(0)).permute(1, 0, 2).contiguous()  # [B,S,T]

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
                "saa_obj": [],
                "realized_total": [],
                "rt_obj_true": [],
            }

        for b in range(B):
            sol_da = self.optnet_DA(
                forecast_sys[b].unsqueeze(0),
                omega_saa[b].unsqueeze(0),
                solver=solver,
                return_rt=False,
            )

            P_DA = sol_da["P_DA"][0].to(dtype=self.optnet_dtype)
            R_up = sol_da["R_up"][0].to(dtype=self.optnet_dtype)
            R_dn = sol_da["R_dn"][0].to(dtype=self.optnet_dtype)

            realized_total, sol_rt = self._realized_total_cost(P_DA, R_up, R_dn, omega_true[b], solver=solver)
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


class DFL_model_vae_MultiNode(nn.Module):
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
        self.optnet_dtype = torch.float64  # 统一数值类型

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
        device = X_all.device
        dtype = self.optnet_dtype
        predictor_kwargs = dict(predictor_kwargs or {})
        filter_kwargs = dict(filter_kwargs or {})

        S = int(predictor_n_samples) if predictor_n_samples is not None else self.n_scen

        # 对齐点1：predictor后立即到目标device/dtype
        Y_scen = self.predictor(
            X_all,
            n_samples=S,
            return_real_scale=True,
            grad_enabled=torch.is_grad_enabled(),
            **predictor_kwargs,
        ).to(device=device, dtype=dtype)

        if self.clamp_min is not None:
            Y_scen = torch.clamp(Y_scen, min=float(self.clamp_min))

        aux_filter = None
        if self.scenario_filter is not None:
            Y_scen, aux_filter = self.scenario_filter(
                Y_scen, is_train=self.training, **filter_kwargs
            )
            # 对齐点2：filter后再次对齐，防止内部device/dtype漂移
            Y_scen = Y_scen.to(device=device, dtype=dtype)

        forecast11 = Y_scen.mean(dim=0)                              # [B,11,T]
        scen11 = Y_scen.permute(1, 0, 2, 3).contiguous()            # [B,S,11,T]

        forecast14 = self._map_11_to_14(forecast11)
        scen14 = self._map_11_to_14(scen11)
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

        if return_aux and not return_filter_aux:
            return total_realized_cost, aux
        if return_filter_aux and not return_aux:
            return total_realized_cost, aux_filter
        if return_aux and return_filter_aux:
            return total_realized_cost, aux, aux_filter
        return total_realized_cost


class DFL_model_vae_DRO_SingleNode(nn.Module):
    def __init__(
        self,
        mgr,
        optnet_DA,
        optnet_RT,
        predictor,
        scenario_filter=None,   # <- added
        n_scen=50,
        clamp_min=0.0,          # <- added
        solver="ECOS",
    ):
        super().__init__()
        self.mgr, self.optnet_DA, self.optnet_RT = mgr, optnet_DA, optnet_RT
        self.predictor = predictor
        self.scenario_filter = scenario_filter   # <- added
        self.n_scen = int(n_scen)
        self.clamp_min = clamp_min               # <- added
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

    def _to_BT(self, x, B, device, dtype):
        x = torch.as_tensor(x, device=device, dtype=dtype)
        if x.ndim == 1:
            x = x.unsqueeze(0).expand(B, -1)
        return x

    def _realized_total_cost(self, P_DA_GT, R_up_GT, R_dn_GT, omega_true_T, solver):
        bG = self.b_G.to(device=P_DA_GT.device, dtype=self.optnet_dtype)
        stage1_energy = (P_DA_GT * bG[:, None]).sum()
        stage1_res = (
            (R_up_GT * (self.res_up_ratio * bG)[:, None]).sum()
            + (R_dn_GT * (self.res_dn_ratio * bG)[:, None]).sum()
        )
        sol_rt = self.optnet_RT(
            R_up_GT.unsqueeze(0),
            R_dn_GT.unsqueeze(0),
            omega_true_T.unsqueeze(0),
            solver=solver
        )
        return stage1_energy + stage1_res + sol_rt["rt_obj"][0], sol_rt

    def forward(
        self,
        X_all,
        y_true,
        hourly_load_min_sys,
        hourly_load_max_sys,
        eps,
        solver=None,
        return_aux=False,
        return_filter_aux=False,  # <- added
        predictor_n_samples=None,
        predictor_kwargs=None,
        filter_kwargs=None,       # <- added
    ):
        solver = solver or self.solver
        device, dtype = X_all.device, self.optnet_dtype
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

        # scenario filter hook
        aux_filter = None
        if self.scenario_filter is not None:
            Y_scen, aux_filter = self.scenario_filter(
                Y_scen, is_train=self.training, **filter_kwargs
            )
            Y_scen = Y_scen.to(device=device, dtype=dtype)

        scen_sys = Y_scen.sum(dim=2)                                     # [S,B,T]
        forecast_sys = scen_sys.mean(dim=0)                              # [B,T]
        omega_scen = (scen_sys - forecast_sys.unsqueeze(0)).permute(1, 0, 2).contiguous()  # [B,S,T]

        Lmin_sys = self._to_BT(hourly_load_min_sys, B, device, dtype)
        Lmax_sys = self._to_BT(hourly_load_max_sys, B, device, dtype)
        om_min_h = Lmin_sys - forecast_sys
        om_max_h = Lmax_sys - forecast_sys
        om_min_h, om_max_h = torch.minimum(om_min_h, om_max_h), torch.maximum(om_min_h, om_max_h)

        om_min_s = omega_scen.min(dim=1).values
        om_max_s = omega_scen.max(dim=1).values
        om_min = torch.minimum(om_min_h, om_min_s)
        om_max = torch.maximum(om_max_h, om_max_s)

        y_true_11 = self._canon_y_true_11T(y_true).to(device=device)
        y_true_real11 = self.predictor.inverse_y(y_true_11).to(device=device, dtype=dtype)
        omega_true = y_true_real11.sum(dim=1) - forecast_sys

        eps_t = torch.as_tensor(eps, device=device, dtype=dtype)
        if eps_t.ndim == 0:
            eps_t = eps_t.view(1).expand(B)

        loss_list = []
        aux = {"realized_total": []} if return_aux else None

        for b in range(B):
            sol_da = self.optnet_DA(
                forecast_sys[b].unsqueeze(0),
                omega_scen[b].unsqueeze(0),
                om_min[b].unsqueeze(0),
                om_max[b].unsqueeze(0),
                eps_t[b].view(1),
                solver=solver
            )
            P_DA = sol_da["P_DA"][0].to(dtype=dtype)
            R_up = sol_da["R_up"][0].to(dtype=dtype)
            R_dn = sol_da["R_dn"][0].to(dtype=dtype)
            realized_total, sol_rt = self._realized_total_cost(P_DA, R_up, R_dn, omega_true[b], solver=solver)
            loss_list.append(realized_total)
            if return_aux:
                aux["realized_total"].append(realized_total.detach())

        loss_vec = torch.stack(loss_list, dim=0)

        if return_aux and not return_filter_aux:
            return loss_vec, aux
        if return_filter_aux and not return_aux:
            return loss_vec, aux_filter
        if return_aux and return_filter_aux:
            return loss_vec, aux, aux_filter
        return loss_vec


class DFL_model_vae_DRO_MultiNode(nn.Module):
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
        self.mgr, self.optnet_DA, self.optnet_RT = mgr, optnet_DA, optnet_RT
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
        device, dtype = X_all.device, self.optnet_dtype
        predictor_kwargs = dict(predictor_kwargs or {})
        filter_kwargs = dict(filter_kwargs or {})

        B = X_all.shape[0]
        S = int(predictor_n_samples) if predictor_n_samples is not None else self.n_scen

        # 关键对齐1：predictor输出后先对齐 device/dtype 再进 filter
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
            # 关键对齐2：filter后再次对齐，避免内部产生device/dtype漂移
            Y_scen_11 = Y_scen_11.to(device=device, dtype=dtype)

        # [S,B,11,T] -> forecast/scenarios
        forecast11 = Y_scen_11.mean(dim=0)                           # [B,11,T]
        scen11 = Y_scen_11.permute(1, 0, 2, 3).contiguous()         # [B,S,11,T]

        forecast14 = self._map_11_to_14(forecast11)                 # [B,14,T]
        scen14 = self._map_11_to_14(scen11)                         # [B,S,14,T]
        omega14 = scen14 - forecast14.unsqueeze(1)                  # [B,S,14,T]

        # DRO bounds
        Lmin14 = torch.as_tensor(hourly_load_min_sys, device=device, dtype=dtype)
        Lmax14 = torch.as_tensor(hourly_load_max_sys, device=device, dtype=dtype)

        # 支持 [14,T] 或 [B,14,T]
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

        # DA DRO
        sol_DA = self.optnet_DA(
            forecast14, omega14, om_min14, om_max14, eps_t, solver=solver
        )
        P_DA = sol_DA["P_DA"].to(device=device, dtype=dtype)
        R_up = sol_DA["R_up"].to(device=device, dtype=dtype)
        R_dn = sol_DA["R_dn"].to(device=device, dtype=dtype)

        # true y
        T_len = forecast14.shape[-1]
        y_true_11 = self._canon_y_true_11T(y_true, T=T_len).to(device=device)
        y_true_real11 = self.predictor.inverse_y(y_true_11).to(device=device, dtype=dtype)
        y_true_14 = self._map_11_to_14(y_true_real11)

        omega14_true = y_true_14 - forecast14
        sol_RT = self.optnet_RT(R_up, R_dn, omega14_true, solver=solver)
        rt_obj_true = sol_RT["rt_obj"].to(device=device, dtype=dtype)   # [B]

        # linear stage1 costs
        bG = self.b_G.to(device=device, dtype=dtype)
        cost_energy = (P_DA * bG[None, :, None]).sum(dim=(1, 2))
        res_up_price = self.res_up_ratio * bG
        res_dn_price = self.res_dn_ratio * bG
        cost_reserve = (
            (R_up * res_up_price[None, :, None]).sum(dim=(1, 2))
            + (R_dn * res_dn_price[None, :, None]).sum(dim=(1, 2))
        )

        total_realized_cost = cost_energy + cost_reserve + rt_obj_true  # [B]

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
    z_temp=1.0,
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

    # ---- 训练/评估时 ScenarioFilter 的 eval 配置同步 ----
    eval_mode = str(getattr(args, "eval_mode", "soft")).lower()
    avoid_rand_duplicate = bool(getattr(args, "avoid_rand_duplicate", False))
    if getattr(dfl, "scenario_filter", None) is not None:
        if hasattr(dfl.scenario_filter, "eval_mode"):
            dfl.scenario_filter.eval_mode = eval_mode
        if hasattr(dfl.scenario_filter, "avoid_rand_duplicate"):
            dfl.scenario_filter.avoid_rand_duplicate = avoid_rand_duplicate

    if filter_kwargs is None:
        filter_kwargs = {
            "tau_gumbel": float(getattr(args, "tau_mix", 1.0)),
            "eps_uniform": float(getattr(args, "eps_uniform", 0.1)),
        }

    # 正则类型与数值稳定项
    div_type = str(getattr(args, "div_type", "inner")).lower()   # inner | kl | entropy
    div_eps = float(getattr(args, "div_eps", 1e-8))

    if train_mode == "filter_only" and getattr(dfl, "scenario_filter", None) is not None:
        for p in dfl.parameters():
            p.requires_grad = False
        for p in dfl.scenario_filter.parameters():
            p.requires_grad = True

        optim_params_filter = [p for p in dfl.scenario_filter.parameters() if p.requires_grad]
        optim_params_predictor = []

        optim = torch.optim.Adam(
            optim_params_filter,
            lr=float(getattr(args, "lr", getattr(args, "filter_lr", 1e-3))),
            # weight_decay=float(getattr(args, "weight_decay", 0.0)),
        )
    else:
        for p in dfl.parameters():
            p.requires_grad = True

        if getattr(dfl, "scenario_filter", None) is not None:
            optim_params_filter = [p for p in dfl.scenario_filter.parameters() if p.requires_grad]
        else:
            optim_params_filter = []

        optim_params_predictor = [p for p in dfl.predictor.parameters() if p.requires_grad]

        optim = torch.optim.Adam(
            [
                {"params": optim_params_filter, "lr": float(getattr(args, "filter_lr", 1e-3))},
                {"params": optim_params_predictor, "lr": float(getattr(args, "dfl_lr", 1e-5))},
            ],
            # weight_decay=float(getattr(args, "weight_decay", 0.0)),
        )

    loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_bs,
        shuffle=True,
    )
    epochs = int(getattr(args, "epochs", 1))
    train_losses = []

    for ep in tqdm(range(epochs), desc=f"Train ({train_mode})", leave=True):
        pbar = tqdm(loader, desc=f"Ep {ep+1}/{epochs}", leave=False)
        epoch_task_loss, epoch_div_loss, samples_cnt = 0.0, 0.0, 0

        for X_all, y_true in pbar:
            X_all = X_all.to(device).float()
            y_true = y_true.to(device).float()
            optim.zero_grad(set_to_none=True)

            kwargs = dict(
                solver=getattr(args, "solver", None),
                return_aux=False,
                return_filter_aux=True,
                predictor_n_samples=int(getattr(args, "S_full", getattr(args, "N_scen", 50))),
                predictor_kwargs={
                    "z_temp": z_temp,
                    "cpu_offload": False,
                },
                filter_kwargs=filter_kwargs,
            )

            if str(problem_mode).lower() == "dro":
                dtype = getattr(dfl, "optnet_dtype", torch.float64)
                kwargs.update(
                    hourly_load_min_sys=torch.as_tensor(args.Lmin, device=device, dtype=dtype),
                    hourly_load_max_sys=torch.as_tensor(args.Lmax, device=device, dtype=dtype),
                    eps=torch.as_tensor(args.eps_value, device=device, dtype=dtype),
                )

            loss_vec, aux_filter = dfl(X_all, y_true, **kwargs)
            task_loss_val = loss_vec.mean()

            # -------- diversity regularizer --------
            div_loss_val = torch.tensor(0.0, device=device)

            if aux_filter is not None and ("p" in aux_filter) and (lambda_div > 0):
                # p: [B,K,S]
                p = aux_filter["p"].clamp_min(div_eps)
                p = p / p.sum(dim=-1, keepdim=True).clamp_min(div_eps)

                B_curr, K_curr, _ = p.shape

                if div_type == "inner":
                    if K_curr > 1:
                        inner_product = torch.bmm(p, p.transpose(1, 2))  # [B,K,K]
                        eye = torch.eye(K_curr, device=device, dtype=torch.bool).unsqueeze(0).expand(B_curr, -1, -1)
                        off_diag = inner_product[~eye]
                        div_loss_val = (off_diag ** 2).mean()

                elif div_type == "kl":
                    if K_curr > 1:
                        p_i = p.unsqueeze(2)  # [B,K,1,S]
                        p_j = p.unsqueeze(1)  # [B,1,K,S]
                        kl_ij = (p_i * (p_i.log() - p_j.log())).sum(dim=-1)
                        kl_ji = (p_j * (p_j.log() - p_i.log())).sum(dim=-1)
                        skl = 0.5 * (kl_ij + kl_ji)
                        eye = torch.eye(K_curr, device=device, dtype=torch.bool).unsqueeze(0).expand(B_curr, -1, -1)
                        div_loss_val = -skl[~eye].mean()

                elif div_type == "entropy":
                    H = -(p * p.log()).sum(dim=-1).mean()
                    div_loss_val = -H

                else:
                    raise ValueError(f"Unknown div_type={div_type}, choose from inner|kl|entropy")
            # ---------------------------------------

            loss = task_loss_val + lambda_div * div_loss_val
            loss.backward()

            if len(optim_params_filter) > 0:
                torch.nn.utils.clip_grad_norm_(optim_params_filter, 1.0)
            if len(optim_params_predictor) > 0:
                torch.nn.utils.clip_grad_norm_(optim_params_predictor, 1.0)

            optim.step()

            B_size = X_all.shape[0]
            epoch_task_loss += float(task_loss_val.detach().cpu()) * B_size
            epoch_div_loss += float(div_loss_val.detach().cpu()) * B_size
            samples_cnt += B_size

            pbar.set_postfix(
                Task=float(task_loss_val.detach().cpu()),
                Div=float(div_loss_val.detach().cpu()),
                Type=div_type,
                BS=train_bs,
            )

        train_losses.append({
            "task": epoch_task_loss / max(samples_cnt, 1),
            "div": epoch_div_loss / max(samples_cnt, 1),
            "div_type": div_type,
            "train_batch_size": train_bs,
        })

    return dfl, train_losses

@torch.no_grad()
def DFL_test(
    dfl,
    test_dataset,
    args,
    z_temp=1.0,
    problem_mode="saa",
    filter_kwargs=None,
    return_filter_aux=False,
):
    """
    兼容 scenario_filter 的测试函数：
    - 默认在 eval() 下运行
    - 支持 soft / discrete eval
    - 可选返回 filter 的辅助信息用于核对 idx_model / p
    """
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
            "eps_uniform": float(getattr(args, "eps_uniform", 0.0)),
        }

    test_bs = int(getattr(args, "test_batch_size", getattr(args, "batch_size", 8)))

    loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=test_bs,
        shuffle=False,
        drop_last=False,
    )

    losses = []
    all_filter_aux = [] if return_filter_aux else None

    pbar = tqdm(loader, desc="Test", leave=True)
    for X_all, y_true in pbar:
        X_all = X_all.to(device).float()
        y_true = y_true.to(device).float()

        # ---- 固定 test-time randomness ----
        set_seed(0)

        kwargs = dict(
            solver=getattr(args, "solver", None),
            return_aux=False,
            return_filter_aux=bool(return_filter_aux),
            predictor_n_samples=int(
                getattr(args, "predictor_n_samples", getattr(args, "S_full", getattr(args, "N_scen", 50)))
            ),
            predictor_kwargs={
                "z_temp": z_temp,
                "cpu_offload": bool(getattr(args, "predictor_cpu_offload", False)),
            },
            filter_kwargs=filter_kwargs,
        )

        if str(problem_mode).lower() == "dro":
            dtype = getattr(dfl, "optnet_dtype", torch.float64)
            kwargs.update(
                hourly_load_min_sys=torch.as_tensor(args.Lmin, device=device, dtype=dtype),
                hourly_load_max_sys=torch.as_tensor(args.Lmax, device=device, dtype=dtype),
                eps=torch.as_tensor(args.eps_value, device=device, dtype=dtype),
            )

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

        losses.append(loss_vec.detach().cpu())
        pbar.set_postfix(loss=float(loss_vec.mean().detach().cpu()))

    loss_all = torch.cat(losses, dim=0)

    if return_filter_aux:
        return loss_all, all_filter_aux
    return loss_all


def run_DFL_vae_joint(
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
    stage2_artifact=None,  # 若传入，则直接从 Stage B 开始
):
    """
    VAE-joint 版本（同理修改：用 rebuild，而不是 deepcopy 整个模型）
    - Stage A 结束后：固定 3 次 rebuild snapshot（dfl_after_stage2 / det_before / before）
    - 复用 stage2_artifact：只 rebuild 1 次得到用于 Stage B 的 dfl；快照直接沿用 artifact
    """
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

    if eval_flags is None or len(eval_flags) != 5:
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

    # ---- 统一 train/test batch size ----
    train_bs = int(getattr(args, "train_batch_size", getattr(args, "dfl_batch_size", getattr(args, "batch_size", 8))))
    test_bs = int(getattr(args, "test_batch_size", getattr(args, "batch_size", 8)))
    args.train_batch_size = train_bs
    args.test_batch_size = test_bs

    print("args.train_batch_size =", train_bs)
    print("args.test_batch_size =", test_bs)

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

        DFL_SO_Class = DFL_model_vae_MultiNode
        DFL_DRO_Class = DFL_model_vae_DRO_MultiNode
        DET_Class = DFL_model_vae_Deterministic_MultiNode
    else:
        SO_Manager = IEEE14_Reserve_SO_Manager_SingleNode
        DRO_Manager = IEEE14_Reserve_DRO_Manager_SingleNode
        DET_Manager = IEEE14_Reserve_Deterministic_DA_Manager_SingleNode

        SAA_DA_OptNet = SingleNode_Reserve_SAA_DA_OptNet
        DRO_DA_OptNet = SingleNode_Reserve_DRO_DA_OptNet
        DET_DA_OptNet = SingleNode_Reserve_Deterministic_DA_OptNet

        RT_OptNet = SingleNode_Reserve_RT_OptNet

        DFL_SO_Class = DFL_model_vae_SingleNode
        DFL_DRO_Class = DFL_model_vae_DRO_SingleNode
        DET_Class = DFL_model_vae_Deterministic_SingleNode

    # -------------------------
    # scenario filter config
    # -------------------------
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

    train_logs_stage2, train_logs_stage3 = [], []
    time_stage2, time_stage3 = 0.0, 0.0

    det_before_eval, stage1_eval, stage2_eval, stage3_before_eval, stage3_eval = {}, {}, {}, {}, {}

    set_seed(seed)
    t0_total = time.time()

    run_stage2 = bool(getattr(args, "run_stage2", True))
    run_stage3 = bool(getattr(args, "run_stage3", True))

    # =========================================================
    # Case 1: 不传 stage2_artifact -> Stage A -> Stage B
    # Case 2: 传入 stage2_artifact -> 直接 Stage B
    # =========================================================
    if not reused_stage2:
        train_data = Dataset_load_multi_node_vae(
            data_path=data_path,
            node_cols=target_nodes,
            flag="train",
            train_length=8760,
            val_ratio=0.2,
            seed=42,
            scale=True,
        )
        test_data = Dataset_load_multi_node_vae(
            data_path=data_path,
            node_cols=target_nodes,
            flag="test",
            train_length=8760,
            val_ratio=0.2,
            seed=42,
            scale=True,
        )

        def build_predictor(joint_model, dataset):
            return Joint_vae_predictor(
                joint_model=copy.deepcopy(joint_model),
                dataset=dataset,
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
        best_z = pack_data_m["_JOINT_"]["best_z_temp"]

        # -------------------------
        # core components
        # -------------------------
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
            solver="ECOS",
        ).to(device)

        dfl_before = DFLClass(
            mgr=mgr_local,
            optnet_DA=optnet_DA,
            optnet_RT=optnet_RT,
            predictor=predictor_before,
            scenario_filter=copy.deepcopy(filter_module),
            n_scen=args.N_scen,
            solver="ECOS",
        ).to(device)

        dfl = DFLClass(
            mgr=mgr_local,
            optnet_DA=optnet_DA,
            optnet_RT=optnet_RT,
            predictor=predictor_train,
            scenario_filter=filter_module,
            n_scen=args.N_scen,
            solver="ECOS",
        ).to(device)

        def eval_on_splits(model, stage_tag: str):
            out = {}
            if "test" in eval_splits:
                set_seed(seed)
                out[f"test_losses_{stage_tag}"] = DFL_test(
                    model, test_data, args, z_temp=best_z, problem_mode=problem_mode
                )
            if "train" in eval_splits:
                set_seed(seed)
                out[f"train_losses_{stage_tag}"] = DFL_test(
                    model, train_data, args, z_temp=best_z, problem_mode=problem_mode
                )
            return out

        # -------- rebuild helper (NAME aligned) --------
        def rebuild_model_like(src_model, kind: str):
            """
            kind:
              - "dfl": rebuild DFLClass model with fresh predictor/filter, same mgr/optnet shells
              - "det": rebuild DET_Class model with fresh predictor/filter, same mgr_det/optnet_DA_det/optnet_RT shells
            """
            if src_model is None:
                return None

            new_pred = build_predictor(models_m, train_data)
            if hasattr(src_model, "predictor"):
                new_pred.load_state_dict(copy.deepcopy(src_model.predictor.state_dict()))

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
                    predictor=new_pred,
                    scenario_filter=new_filter,
                    n_scen=args.N_scen,
                    solver="ECOS",
                ).to(device)
            elif kind == "det":
                return DET_Class(
                    mgr=mgr_det,
                    optnet_DA=optnet_DA_det,
                    optnet_RT=optnet_RT,
                    predictor=new_pred,
                    scenario_filter=new_filter,
                    n_scen=args.N_scen,
                    solver="ECOS",
                ).to(device)
            else:
                raise ValueError(f"unknown kind={kind}")

        # ---------- baselines ----------
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

        # ==================== Stage A ====================
        args.epochs = int(getattr(args, "dfl_epochs", 2))
        args.dfl_lr = float(getattr(args, "dfl_lr", 1e-7))
        args.filter_lr = float(getattr(args, "filter_lr", 1e-3))

        if run_stage2:
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
                z_temp=best_z,
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
        else:
            print("\n ---> [skip] Stage A disabled (run_stage2=False)")

        # ===== snapshots: EXACTLY 3 rebuild calls (aligned) =====
        dfl_after_stage2_snapshot = rebuild_model_like(dfl, kind="dfl")
        dfl_det_before_snapshot = rebuild_model_like(dfl_det_before, kind="det") if dfl_det_before is not None else None
        dfl_before_snapshot = rebuild_model_like(dfl_before, kind="dfl") if dfl_before is not None else None

    else:
        print("\n ---> Reusing passed stage2_artifact, skip Stage A and jump to Stage B")

        train_data = stage2_artifact["train_data"]
        test_data = stage2_artifact["test_data"]
        best_z = stage2_artifact["best_z"]
        mode_str = stage2_artifact["mode_str"]

        det_before_eval = copy.deepcopy(stage2_artifact.get("det_before_eval", {}))
        stage1_eval = copy.deepcopy(stage2_artifact.get("stage1_eval", {}))
        stage2_eval = copy.deepcopy(stage2_artifact.get("stage2_eval", {}))

        train_logs_stage2 = copy.deepcopy(stage2_artifact.get("train_logs_stage2", []))
        time_stage2 = float(stage2_artifact.get("time_stage2_sec", 0.0))

        # rebuild fresh shells for Stage B (mgr/optnet)
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
            return Joint_vae_predictor(
                joint_model=copy.deepcopy(joint_model),
                dataset=dataset,
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

            new_pred = build_predictor(models_m, train_data)
            if hasattr(src_model, "predictor"):
                new_pred.load_state_dict(copy.deepcopy(src_model.predictor.state_dict()))

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
                predictor=new_pred,
                scenario_filter=new_filter,
                n_scen=args.N_scen,
                solver="ECOS",
            ).to(device)

        # rebuild ONCE for Stage B training
        stage2_model = stage2_artifact["dfl_after_stage2"]
        dfl = rebuild_model_like(stage2_model, kind="dfl")

        # snapshots reused directly (NO extra rebuild)
        dfl_after_stage2_snapshot = stage2_model
        dfl_det_before_snapshot = stage2_artifact.get("dfl_det_before", None)
        dfl_before_snapshot = stage2_artifact.get("dfl_before", None)

        # pass-through for result keys
        dfl_det_before = stage2_artifact.get("dfl_det_before", None)
        dfl_before = stage2_artifact.get("dfl_before", None)

        def eval_on_splits(model, stage_tag: str):
            out = {}
            if "test" in eval_splits:
                set_seed(seed)
                out[f"test_losses_{stage_tag}"] = DFL_test(
                    model, test_data, args, z_temp=best_z, problem_mode=problem_mode
                )
            if "train" in eval_splits:
                set_seed(seed)
                out[f"train_losses_{stage_tag}"] = DFL_test(
                    model, train_data, args, z_temp=best_z, problem_mode=problem_mode
                )
            return out

    # ==================== Stage B ====================
    if run_stage3:
        args.epochs = int(getattr(args, "filter_epochs", 10))
        args.lr = float(getattr(args, "filter_lr", 1e-3))

        print(
            f"\n ---> Stage B: switch to learnable filter, train Filter only "
            f"(epochs={args.epochs}, lr={args.lr}, train_bs={train_bs})"
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
            z_temp=best_z,
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
        "best_z": best_z,
        "mode_str": mode_str,
        "dfl_after_stage2": dfl_after_stage2_snapshot,
        "dfl_det_before": dfl_det_before_snapshot,
        "dfl_before": dfl_before_snapshot,
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
        "dfl_det_before": locals().get("dfl_det_before", None),
        "dfl_before": locals().get("dfl_before", None),
        "dfl_trained": dfl_trained,
        "train_logs_stage2": train_logs_stage2,
        "train_logs_stage3": train_logs_stage3,
        "time_stage2_sec": float(time_stage2),
        "time_stage3_sec": float(time_stage3),
        "train_time_sec_total": float(train_time_sec_total),
        "z_temp": best_z,
        "train_batch_size_used": int(train_bs),
        "test_batch_size_used": int(test_bs),
        "N_scen": int(args.N_scen),
        "S_full": int(args.S_full),
        "K_rand": int(K_rand),
        "eval_flags": tuple(map(bool, eval_flags)),
        "eval_mode": str(getattr(args, "eval_mode", "soft")).lower(),
        "avoid_rand_duplicate": bool(getattr(args, "avoid_rand_duplicate", False)),
        "stage2_artifact": stage2_artifact_out,
    }

    result.update(det_before_eval)
    result.update(stage1_eval)
    result.update(stage2_eval)
    result.update(stage3_before_eval)
    result.update(stage3_eval)

    # ---------- aliases ----------
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




import copy
import torch
from utils import _parse_compare_method_names

def compare_scenario_filters_with_stage3_learned_vae(
    base_result,
    args,
    problem_mode,
    optimization_mode,
    models_m,
    target_nodes,
    device,
    eval_splits=("test",),
    method_names=None,
    seed=0,
    verbose=True,
):
    """
    Fair comparison under VAE-joint protocol:

    - Stage 2 trains predictor backbone
    - Stage 3 trains only the learned ScenarioFilter

    Comparison rule:
    - same Stage-2-trained predictor backbone for all methods
    - learned: use Stage-3-trained scenario_filter
    - random/kmeans/kmedoids/hierarchical/...: replace scenario_filter with baseline filter

    IMPORTANT:
    We do NOT blindly deepcopy the whole model if reconstruction is safer.
    We rebuild a fresh model using:
      - predictor weights from stage2 backbone
      - a provided scenario filter
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
    best_z = stage2_artifact["best_z"]

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

    # ------------------------------------------------------------
    # unified eval config
    # ------------------------------------------------------------
    args.eval_mode = str(getattr(args, "eval_mode", "soft")).lower()
    args.avoid_rand_duplicate = bool(getattr(args, "avoid_rand_duplicate", False))

    print("\n[ScenarioFilter eval config]")
    print("eval_mode =", args.eval_mode)
    print("avoid_rand_duplicate =", args.avoid_rand_duplicate)

    # ------------------------------------------------------------
    # unified train/test batch size (same style as run_DFL_vae_joint)
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # scenario filter config
    # ------------------------------------------------------------
    multi = is_multi(optimization_mode)
    args.S_full = int(getattr(args, "S_full", 50))
    K = int(getattr(args, "N_scen", 50))
    K_rand = int(getattr(args, "K_rand", 0))
    if K_rand > K:
        raise ValueError(f"K_rand({K_rand}) must be <= N_scen({K}).")

    # -------- managers / classes --------
    if multi:
        SO_Manager = IEEE14_Reserve_SO_Manager_MultiNode
        DRO_Manager = IEEE14_Reserve_DRO_Manager_MultiNode

        SAA_DA_OptNet = MultiNode_Reserve_SAA_DA_OptNet
        DRO_DA_OptNet = MultiNode_Reserve_DRO_DA_OptNet

        RT_OptNet = MultiNode_Reserve_RT_OptNet

        DFL_SO_Class = DFL_model_vae_MultiNode
        DFL_DRO_Class = DFL_model_vae_DRO_MultiNode
    else:
        SO_Manager = IEEE14_Reserve_SO_Manager_SingleNode
        DRO_Manager = IEEE14_Reserve_DRO_Manager_SingleNode

        SAA_DA_OptNet = SingleNode_Reserve_SAA_DA_OptNet
        DRO_DA_OptNet = SingleNode_Reserve_DRO_DA_OptNet

        RT_OptNet = SingleNode_Reserve_RT_OptNet

        DFL_SO_Class = DFL_model_vae_SingleNode
        DFL_DRO_Class = DFL_model_vae_DRO_SingleNode

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

    optnet_RT = RT_OptNet(mgr=mgr_local, T=24).to(device)

    def build_predictor():
        return Joint_vae_predictor(
            joint_model=copy.deepcopy(models_m),
            dataset=train_data,
        ).to(device)

    def make_dfl_model(predictor, scenario_filter):
        return DFLClass(
            mgr=mgr_local,
            optnet_DA=optnet_DA,
            optnet_RT=optnet_RT,
            predictor=predictor,
            scenario_filter=scenario_filter,
            n_scen=args.N_scen,
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
                z_temp=best_z,
                problem_mode=problem_mode,
            )
        if "train" in eval_splits:
            set_seed(seed)
            out[f"train_losses_{stage_tag}"] = DFL_test(
                model,
                train_data,
                args,
                z_temp=best_z,
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
        "optimization_mode": "multi" if multi else "single",
        "problem_mode": mode_str,
        "backbone_predictor_source": "stage2_artifact['dfl_after_stage2']",
        "learned_filter_source": "base_result['dfl_trained'].scenario_filter",
        "method_names": list(method_names),
        "eval_splits": eval_splits,
        "train_batch_size_used": int(train_bs),
        "test_batch_size_used": int(test_bs),
        "N_scen": int(args.N_scen),
        "S_full": int(args.S_full),
        "K_rand": int(K_rand),
        "eval_mode": str(getattr(args, "eval_mode", "soft")).lower(),
        "avoid_rand_duplicate": bool(getattr(args, "avoid_rand_duplicate", False)),
        "details": {},
        "summary_mean": {},
    }

    if verbose:
        print("\n" + "=" * 90)
        print("[Compare Scenario Filters - VAE Joint]")
        print("Predictor backbone :", "Stage-2 trained model")
        print("Learned filter     :", "Stage-3 trained scenario_filter")
        print("Methods            :", method_names)
        print("Eval splits        :", eval_splits)
        print("train_batch_size   :", train_bs)
        print("test_batch_size    :", test_bs)
        print("=" * 90)

    for name in method_names:
        if verbose:
            print(f"\n[Compare] evaluating scenario filter: {name}")

        if name == "learned":
            scenario_filter_i = copy.deepcopy(trained_model.scenario_filter)
        else:
            scenario_filter_i = build_scenario_baseline_filter(name, args, device)

        model_i = rebuild_with_specific_filter(stage2_model, scenario_filter_i)

        stage_tag = f"compare_stage2backbone_{name}"
        eval_out = _eval_on_splits(model_i, stage_tag)

        detail = {
            "method_name": name,
            "scenario_filter_class": (
                type(scenario_filter_i).__name__
                if scenario_filter_i is not None else None
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
            print(f"[Compare][{name}]")
            for split in eval_splits:
                if split in mean_info and mean_info[split] is not None:
                    print(f"  {split} mean loss: {mean_info[split]:.6f}")

    return compare_result