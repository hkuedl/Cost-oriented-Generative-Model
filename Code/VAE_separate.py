import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import time
import scipy.stats as stats
from torch.utils.data import DataLoader, TensorDataset
from utils import *
from data_loader import *
from combined_data_loader import *
from VAE import *
from tqdm import tqdm
from Optimization_multi_node import *
from Optimization_single_node import *
from scenarios_reduce import *


class Runner_vae_separate:
    def __init__(
        self,
        train_set,
        val_set,
        test_set,
        z_dim=16,
        hidden=256,
        dropout=0.0,
        lr=1e-3,
        beta=0.1,
        beta_anneal=True,
        anneal_warmup=50,   # 现在表示 warmup steps（batch 数）
        device=None,
    ):
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set

        self.beta = float(beta)
        self.beta_anneal = bool(beta_anneal)
        self.anneal_warmup = int(anneal_warmup)

        self.global_step = 0  # step-based anneal
        self.kl_w = 1.0       # cache (0~1)

        input_dim = train_set.X.shape[-1]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = CVAE_Single(
            input_dim=input_dim, z_dim=z_dim, hidden=hidden, dropout=dropout
        ).to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)

    def _anneal_factor(self):
        if (not self.beta_anneal) or self.anneal_warmup <= 0:
            return 1.0
        return float(min(1.0, self.global_step / float(self.anneal_warmup)))

    def fit(self, epochs=500, batch_size=128, patience=50, best_path="best.pt", verbose=False):
        train_loader = DataLoader(self.train_set, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(self.val_set, batch_size=batch_size, shuffle=False, drop_last=False)

        best_val = float("inf")
        bad = 0

        for ep in range(1, epochs + 1):
            self.model.train()
            tr_losses, tr_mse, tr_kl = [], [], []

            for X, y in train_loader:
                af = self._anneal_factor()
                beta_step = self.beta * af

                X = X.to(self.device)
                y = y.to(self.device)

                mu_y, mu_q, logvar_q, mu_p, logvar_p = self.model(X, y)
                loss, mse, kl = cvae_point_elbo_loss_condprior(
                    mu_y, y, mu_q, logvar_q, mu_p, logvar_p, beta=beta_step
                )

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                self.global_step += 1

                tr_losses.append(float(loss.item()))
                tr_mse.append(float(mse.item()))
                tr_kl.append(float(kl.item()))

            # val: 用当前退火因子（保持与训练大致一致）
            self.model.eval()
            va_losses = []
            with torch.no_grad():
                af = self._anneal_factor()
                beta_eval = self.beta * af
                for X, y in val_loader:
                    X = X.to(self.device)
                    y = y.to(self.device)
                    mu_y, mu_q, logvar_q, mu_p, logvar_p = self.model(X, y)
                    loss, _, _ = cvae_point_elbo_loss_condprior(
                        mu_y, y, mu_q, logvar_q, mu_p, logvar_p, beta=beta_eval
                    )
                    va_losses.append(float(loss.item()))
            val_loss = float(np.mean(va_losses)) if len(va_losses) else float("inf")

            if verbose and (ep == 1 or ep % 10 == 0):
                print(
                    f"Epoch {ep:4d} | af={self._anneal_factor():.4f} beta={self.beta * self._anneal_factor():.6g} | "
                    f"train={np.mean(tr_losses):.6f} (mse={np.mean(tr_mse):.6f}, kl={np.mean(tr_kl):.6f}) | "
                    f"val={val_loss:.6f}"
                )

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
    def test_samples(self, n_samples=500, return_real_scale=True, z_temp=1.0):
        X = self.test_set.X.to(self.device)  # [D,24,F]
        y = self.test_set.y                  # [D,24] on CPU

        samples = self.model.sample(X, n_samples=n_samples, z_temp=z_temp).cpu()  # [S,D,24]

        if not return_real_scale:
            return samples.numpy(), y.numpy()

        y_real = self.test_set.inverse_transform(y)

        s = samples.numpy().reshape(-1, 1)
        s_real = self.test_set.scaler_y.inverse_transform(s).reshape(n_samples, y.shape[0], y.shape[1])
        return s_real, y_real

@torch.no_grad()
def pinball_loss_ensemble(y_true, y_samp, quantiles=(0.1, 0.5, 0.9)):
    """
    y_true: [B,T] or [B,T,1]
    y_samp: [S,B,T] or [S,B,T,1]
    return: scalar (avg over Q,B,T)
    """
    if y_true.ndim == 3 and y_true.shape[-1] == 1:
        y_true = y_true[..., 0]
    if y_samp.ndim == 4 and y_samp.shape[-1] == 1:
        y_samp = y_samp[..., 0]

    qs = torch.tensor(list(quantiles), device=y_true.device, dtype=y_true.dtype)  # [Q]
    y_q = torch.quantile(y_samp, qs, dim=0)  # [Q,B,T]
    diff = y_true.unsqueeze(0) - y_q  # [Q,B,T]
    qv = qs.view(-1, 1, 1)
    loss = torch.maximum(qv * diff, (qv - 1.0) * diff)  # [Q,B,T]
    return loss.mean().item()


@torch.no_grad()
def select_best_ztemp_for_single_node(
    model,
    val_set,
    device,
    ztemps,
    n_samples=50,
    batch_size=256,
    quantiles=None,
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

    for t in list(ztemps):
        scores = []
        for X, y in loader:
            X = X.to(device).float()  # [B,24,F]
            y = y.to(device).float()  # [B,24] or [B,24,1]

            samp = model.sample(X, n_samples=n_samples, z_temp=float(t))  # [S,B,24] or [S,B,24,1]
            pb = pinball_loss_ensemble(y, samp, quantiles=quantiles)
            scores.append(pb)

        score = float(np.mean(scores)) if scores else float("inf")
        table.append((float(t), score))

        if verbose:
            print(f"  z_temp={t:>6} | val pinball{tuple(quantiles)}={score:.6f}")

        if score < best_score:
            best_score = score
            best_t = float(t)

    return best_t, best_score, table

def run_vae_separate(
    data_path,
    node_cols,
    DataHandler,  
    device=None,
    epochs=1000,
    batch_size=128,
    lr=1e-4,
    patience=50,
    ckpt_dir="../Model/VAE/ckpt_nodes_vae",
    verbose=False,
    train_length=8760,
    val_ratio=0.2,
    seed=42,
    z_dim=64,
    hidden=256,
    dropout=0.1,
    beta=0.01,
    beta_anneal=True,
    anneal_warmup=50,
    select_z_temp=True,
    ztemps=np.linspace(0.5, 3, 8),
    ztemp_n_samples=50,
    ztemp_batch_size=256,
    ztemp_quantiles=None,
):
    os.makedirs(ckpt_dir, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(seed)

    models_s, handlers_s, pack_data_s = {}, {}, {}
    best_ztemp_s = {}

    # helper: pack splits like multi-node code
    def _mk_splits(train_set, val_set, test_set):
        return {
            "train": {"X": train_set.X, "Y": train_set.y},
            "val":   {"X": val_set.X,   "Y": val_set.y},
            "test":  {"X": test_set.X,  "Y": test_set.y},
        }

    for node in node_cols:
        train_set = DataHandler(
            data_path, node, flag="train",
            train_length=train_length, val_ratio=val_ratio, seed=seed
        )
        val_set = DataHandler(
            data_path, node, flag="val",
            train_length=train_length, val_ratio=val_ratio, seed=seed
        )
        test_set = DataHandler(
            data_path, node, flag="test",
            train_length=train_length, val_ratio=val_ratio, seed=seed
        )

        runner = Runner_vae_separate(
            train_set=train_set,
            val_set=val_set,
            test_set=test_set,
            z_dim=z_dim,
            hidden=hidden,
            dropout=dropout,
            lr=lr,
            beta=beta,
            beta_anneal=beta_anneal,
            anneal_warmup=anneal_warmup,
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

        model = runner.model.to(device).eval()

        # ---- pick per-node z_temp on VAL ----
        if select_z_temp:
            best_t, best_score, table = select_best_ztemp_for_single_node(
                model=model,
                val_set=val_set,
                device=torch.device(device) if isinstance(device, str) else device,
                ztemps=ztemps,
                n_samples=ztemp_n_samples,
                batch_size=ztemp_batch_size,
                quantiles=ztemp_quantiles,
                verbose=verbose,
            )
        else:
            best_t, best_score, table = None, None, None

        best_ztemp_s[node] = best_t
        if verbose and select_z_temp:
            print(f"[{node}] best z_temp={best_t} | val pinball={best_score:.6f}")

        # ---- NEW: store all splits (train/val/test) ----
        models_s[node] = model
        handlers_s[node] = {"train": train_set, "val": val_set, "test": test_set}
        pack_data_s[node] = {
            "splits": _mk_splits(train_set, val_set, test_set),
            "node": node,
            "best_path": best_path,
            "best_z_temp": (None if best_t is None else float(best_t)),
            "best_val_pinball": (None if best_score is None else float(best_score)),
            "ztemp_table": table,
            "ztemp_quantiles": (None if ztemp_quantiles is None else tuple(ztemp_quantiles)),
            "runner": runner,  # optional: keep for debugging
        }

    if verbose and select_z_temp:
        print("best_z_temp per node:")
        for node in node_cols:
            print(f"  {node}: {best_ztemp_s[node]}")

    return models_s, handlers_s, pack_data_s

@torch.no_grad()
def sample_window_vae_separate(
    models_s,
    handlers_s,
    pack_data_s,
    target_nodes,
    horizon_days=7,
    start_day=0,
    n_samples=50,
    seq_len=24,
    z_temp=None,     # None -> per-node best_z_temp
    split="test",    # NEW: "train" | "val" | "test"
):
    split = str(split)
    node_names = list(target_nodes)
    N = len(node_names)

    # --- get D from first node / split ---
    pack0 = pack_data_s[node_names[0]]
    if "splits" not in pack0 or split not in pack0["splits"]:
        raise KeyError(
            f"pack_data_s[{node_names[0]}]['splits'] missing split={split}. "
            f"Available: {list(pack0.get('splits', {}).keys())}"
        )

    X0 = pack0["splits"][split]["X"]
    D = int(X0.shape[0])
    if start_day < 0 or start_day >= D:
        raise ValueError(f"start_day={start_day} out of range, total_{split}_days={D}")

    actual_days = min(int(horizon_days), D - int(start_day))
    L = actual_days * int(seq_len)

    Y_true = np.zeros((N, L), dtype=np.float32)
    Y_pred = np.zeros((int(n_samples), N, L), dtype=np.float32)

    for j, node in enumerate(node_names):
        model = models_s[node]

        pack = pack_data_s[node]
        if "splits" not in pack or split not in pack["splits"]:
            raise KeyError(
                f"pack_data_s[{node}]['splits'] missing split={split}. "
                f"Available: {list(pack.get('splits', {}).keys())}"
            )

        # dataset handler for this split (provides inverse_transform)
        handler = handlers_s[node][split]

        X_all = pack["splits"][split]["X"]  # [D,T,F]
        y_all = pack["splits"][split]["Y"]  # [D,T] or [D,T,1]

        # z_temp: per node best by default
        zt = z_temp
        if zt is None:
            zt = pack.get("best_z_temp", 1.0)
            if zt is None:
                zt = 1.0

        device = next(model.parameters()).device
        model.eval()

        for d in range(start_day, start_day + actual_days):
            idx0 = (d - start_day) * seq_len
            idx1 = idx0 + seq_len

            # ---- TRUE ----
            y_day_norm = y_all[d]  # [T] or [T,1] (torch)
            # keep EXACT old behavior: use handler.inverse_transform
            y_day_real = handler.inverse_transform(y_day_norm)
            Y_true[j, idx0:idx1] = np.asarray(y_day_real).reshape(-1)

            # ---- SAMPLES ----
            X = X_all[d].unsqueeze(0).to(device).float()  # [1,T,F]
            samp_norm = model.sample(X, n_samples=int(n_samples), z_temp=float(zt)).squeeze(1)
            # samp_norm: [S,T] or [S,T,1]

            if hasattr(handler, "inverse_transform_y"):
                samp_real = handler.inverse_transform_y(samp_norm)
            else:
                # keep EXACT old fallback logic
                samp_real = handler.inverse_transform(
                    samp_norm.detach().cpu().numpy().reshape(-1)
                ).reshape(int(n_samples), int(seq_len))

            samp_real = np.asarray(samp_real)
            if samp_real.ndim == 3 and samp_real.shape[-1] == 1:
                samp_real = samp_real[..., 0]  # [S,T]

            Y_pred[:, j, idx0:idx1] = samp_real.reshape(int(n_samples), -1)

    return dict(
        mode="single_window_vae_single_split",
        split=split,
        target_nodes=node_names,
        start_day=int(start_day),
        horizon_days=int(actual_days),
        seq_len=int(seq_len),
        n_samples=int(n_samples),
        z_temp=z_temp,  # compatibility: actual might be per-node when None
        Y_true=Y_true,
        Y_pred=Y_pred,
    )


class Multi_vae_predictor(nn.Module):
    """
    多节点 VAE 封装：每节点一个 CVAE_Single。
    输入:
      X_all: [B,N,24,F]
    输出:
      scenarios: [S,B,N,24] (real scale if return_real_scale=True)

    关键：用 grad_enabled 控制是否保留计算图；训练时必须 grad_enabled=True 且 cpu_offload=False。
    """
    def __init__(self, node_models: dict, scaler_y_map: dict, node_order: list, dtype=torch.float64):
        super().__init__()
        self.node_order = list(node_order)
        self.models = nn.ModuleDict({str(k): node_models[k] for k in self.node_order})
        self.dtype = dtype

        # register mean/scale buffers for differentiable inverse-transform
        means, scales = [], []
        for node in self.node_order:
            sc = scaler_y_map[node]
            m = np.asarray(sc.mean_, dtype=np.float64).reshape(1)
            s = np.asarray(getattr(sc, "scale_", getattr(sc, "std_", None)), dtype=np.float64).reshape(1)
            if s is None:
                raise ValueError(f"Scaler for node={node} has no scale_/std_.")
            means.append(torch.tensor(m))
            scales.append(torch.tensor(s))

        mean = torch.stack(means, dim=0).reshape(len(self.node_order), 1, 1)   # [N,1,1]
        scale = torch.stack(scales, dim=0).reshape(len(self.node_order), 1, 1) # [N,1,1]
        self.register_buffer("y_mean_N11", mean.to(dtype))
        self.register_buffer("y_scale_N11", scale.to(dtype))

    def inverse_y(self, y_norm_BNT: torch.Tensor) -> torch.Tensor:
        """
        y_norm_BNT: [B,N,T] -> real-scale [B,N,T]
        训练时保持输入dtype以避免不必要的cast导致图/稳定性问题。
        """
        mean = self.y_mean_N11.to(device=y_norm_BNT.device, dtype=y_norm_BNT.dtype).view(1, -1, 1)
        scale = self.y_scale_N11.to(device=y_norm_BNT.device, dtype=y_norm_BNT.dtype).view(1, -1, 1)
        return y_norm_BNT * scale + mean

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
        cpu_offload=False,
        return_real_scale=True,
        grad_enabled=None,
        z_temp=None,   # dict: node -> z_temp
        shared_noise=False,    # 留接口；要实现 shared_noise 需 CVAE.sample 支持传 eps
    ):
        if grad_enabled is None:
            grad_enabled = torch.is_grad_enabled()
        if grad_enabled and cpu_offload:
            raise ValueError("Training (grad_enabled=True) requires cpu_offload=False.")

        B, N, T, F = X_all.shape
        S = int(n_samples)

        node_samples = []
        for j, node in enumerate(self.node_order):
            key = str(node)
            model = self.models[key]
            X_j = X_all[:, j, :, :]  # [B,24,F]

            zt = 1.0
            if z_temp is not None:
                zt = float(z_temp.get(node, z_temp.get(key, 1.0)) or 1.0)

            # 关键：训练路径不能 no_grad / detach / cpu / numpy
            y_s_norm = model.sample(X_j, n_samples=S, z_temp=zt, grad_enabled=grad_enabled)

            if y_s_norm.ndim == 4:
                if y_s_norm.shape[-1] != 1:
                    raise ValueError(f"Expect last dim=1, got {tuple(y_s_norm.shape)} for node={node}.")
                y_s_norm = y_s_norm[..., 0]

            if y_s_norm.shape != (S, B, T):
                raise ValueError(f"Expect y_s_norm shape {(S,B,T)} but got {tuple(y_s_norm.shape)} for node={node}")

            if cpu_offload:
                y_s_norm = y_s_norm.detach().cpu()

            node_samples.append(y_s_norm)

            del y_s_norm, X_j
            if X_all.is_cuda and not grad_enabled:
                torch.cuda.empty_cache()

        YS = torch.stack(node_samples, dim=2)  # [S,B,N,T]

        if not return_real_scale:
            return YS

        mean = self.y_mean_N11.to(YS.device)
        scale = self.y_scale_N11.to(YS.device)

        # 训练：保持YS dtype（通常float32）确保反传稳定
        if grad_enabled:
            mean = mean.to(dtype=YS.dtype).view(1, 1, -1, 1)
            scale = scale.to(dtype=YS.dtype).view(1, 1, -1, 1)
            return YS * scale + mean

        # 推理：可以转成self.dtype（如float64）方便后处理/精度
        mean = mean.to(dtype=self.dtype).view(1, 1, -1, 1)
        scale = scale.to(dtype=self.dtype).view(1, 1, -1, 1)
        return YS.to(self.dtype) * scale + mean

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
        hourly_load_min_sys=None,   # 占位：兼容 DRO 接口
        hourly_load_max_sys=None,   # 占位：兼容 DRO 接口
        eps=None,                   # 占位：兼容 DRO 接口
        solver=None,
        return_aux=False,
        return_filter_aux=False,
        predictor_n_samples=None,
        predictor_kwargs=None,
        filter_kwargs=None,
    ):
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

        # deterministic forecast: scenario mean
        forecast_sys = Y_scen.sum(dim=2).mean(dim=0)   # [B,T]

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
        hourly_load_min_sys=None,   # 占位：兼容 DRO 接口
        hourly_load_max_sys=None,   # 占位：兼容 DRO 接口
        eps=None,                   # 占位：兼容 DRO 接口
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

        # deterministic forecast = scenario mean
        forecast11 = Y_scen_11.mean(dim=0)      # [B,11,T]
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
            Y_scen_11, aux_filter = self.scenario_filter(Y_scen_11, is_train=self.training, **filter_kwargs)
            Y_scen_11 = Y_scen_11.to(device=device, dtype=dtype)

        forecast11 = Y_scen_11.mean(dim=0)                  # [B,11,T]
        scen11 = Y_scen_11.permute(1, 0, 2, 3).contiguous() # [B,S,11,T]

        forecast14 = self._map_11_to_14(forecast11)
        scen14 = self._map_11_to_14(scen11)
        omega14 = scen14 - forecast14.unsqueeze(1)

        sol_DA = self.optnet_DA(forecast14, omega14, solver=solver, return_rt=False, return_cost=True)
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
        cost_reserve = ((R_up * (self.res_up_ratio * bG)[None, :, None]).sum(dim=(1, 2))
                      + (R_dn * (self.res_dn_ratio * bG)[None, :, None]).sum(dim=(1, 2)))
        total_realized_cost = cost_energy + cost_reserve + rt_obj_true

        aux = None
        if return_aux:
            aux = {"forecast11": forecast11.detach(), "forecast14": forecast14.detach(),
                   "P_DA": P_DA.detach(), "R_up": R_up.detach(), "R_dn": R_dn.detach(),
                   "rt_obj_true": rt_obj_true.detach(), "cost_energy": cost_energy.detach(),
                   "cost_reserve": cost_reserve.detach()}

        if return_aux and not return_filter_aux: return total_realized_cost, aux
        if return_filter_aux and not return_aux: return total_realized_cost, aux_filter
        if return_aux and return_filter_aux: return total_realized_cost, aux, aux_filter
        return total_realized_cost


class DFL_model_vae_DRO_MultiNode(nn.Module):
    def __init__(self, mgr, optnet_DA, optnet_RT, predictor, scenario_filter=None, n_scen=50, clamp_min=0.0, solver="ECOS"):
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
        if y_true.dim() != 3: raise ValueError(f"y_true bad shape {tuple(y_true.shape)}")
        if y_true.shape[1] == 11 and y_true.shape[2] == T: return y_true
        if y_true.shape[1] == T and y_true.shape[2] == 11: return y_true.permute(0, 2, 1).contiguous()
        raise ValueError(f"y_true must be [B,11,{T}] or [B,{T},11], got {tuple(y_true.shape)}")

    def _map_11_to_14(self, x_11):
        *prefix, n11, T = x_11.shape
        if n11 != 11: raise ValueError(f"expected 11 nodes, got {n11}")
        out = x_11.new_zeros((*prefix, 14, T))
        out[..., self.bus_indices_11_to_14, :] = x_11
        return out

    def forward(self, X_all, y_true, hourly_load_min_sys, hourly_load_max_sys, eps,
                solver=None, return_aux=False, return_filter_aux=False,
                predictor_n_samples=None, predictor_kwargs=None, filter_kwargs=None):
        solver = solver or self.solver
        device, dtype = X_all.device, self.optnet_dtype
        predictor_kwargs, filter_kwargs = dict(predictor_kwargs or {}), dict(filter_kwargs or {})
        B = X_all.shape[0]
        S = int(predictor_n_samples) if predictor_n_samples is not None else self.n_scen

        Y_scen_11 = self.predictor(
            X_all, n_samples=S, return_real_scale=True, grad_enabled=torch.is_grad_enabled(), **predictor_kwargs
        ).to(device=device, dtype=dtype)

        if self.clamp_min is not None:
            Y_scen_11 = torch.clamp(Y_scen_11, min=float(self.clamp_min))

        aux_filter = None
        if self.scenario_filter is not None:
            Y_scen_11, aux_filter = self.scenario_filter(Y_scen_11, is_train=self.training, **filter_kwargs)
            Y_scen_11 = Y_scen_11.to(device=device, dtype=dtype)

        forecast11 = Y_scen_11.mean(dim=0)
        scen11 = Y_scen_11.permute(1, 0, 2, 3).contiguous()
        forecast14 = self._map_11_to_14(forecast11)
        scen14 = self._map_11_to_14(scen11)
        omega14 = scen14 - forecast14.unsqueeze(1)

        Lmin14 = torch.as_tensor(hourly_load_min_sys, device=device, dtype=dtype)
        Lmax14 = torch.as_tensor(hourly_load_max_sys, device=device, dtype=dtype)
        if Lmin14.ndim == 2: Lmin14 = Lmin14.unsqueeze(0).expand(B, -1, -1).contiguous()
        if Lmax14.ndim == 2: Lmax14 = Lmax14.unsqueeze(0).expand(B, -1, -1).contiguous()

        om_min14 = Lmin14 - forecast14
        om_max14 = Lmax14 - forecast14
        om_min14, om_max14 = torch.minimum(om_min14, om_max14), torch.maximum(om_min14, om_max14)

        eps_t = torch.as_tensor(eps, device=device, dtype=dtype)
        if eps_t.ndim == 0: eps_t = eps_t.view(1).expand(B)

        sol_DA = self.optnet_DA(forecast14, omega14, om_min14, om_max14, eps_t, solver=solver)
        P_DA, R_up, R_dn = sol_DA["P_DA"].to(device=device, dtype=dtype), sol_DA["R_up"].to(device=device, dtype=dtype), sol_DA["R_dn"].to(device=device, dtype=dtype)

        T_len = forecast14.shape[-1]
        y_true_11 = self._canon_y_true_11T(y_true, T=T_len).to(device=device)
        y_true_real11 = self.predictor.inverse_y(y_true_11).to(device=device, dtype=dtype)
        y_true_14 = self._map_11_to_14(y_true_real11)

        omega14_true = y_true_14 - forecast14
        sol_RT = self.optnet_RT(R_up, R_dn, omega14_true, solver=solver)
        rt_obj_true = sol_RT["rt_obj"].to(device=device, dtype=dtype)

        bG = self.b_G.to(device=device, dtype=dtype)
        cost_energy = (P_DA * bG[None, :, None]).sum(dim=(1, 2))
        cost_reserve = ((R_up * (self.res_up_ratio * bG)[None, :, None]).sum(dim=(1, 2))
                      + (R_dn * (self.res_dn_ratio * bG)[None, :, None]).sum(dim=(1, 2)))
        total_realized_cost = cost_energy + cost_reserve + rt_obj_true

        aux = None
        if return_aux:
            aux = {"forecast14": forecast14.detach(), "om_min14": om_min14.detach(), "om_max14": om_max14.detach(),
                   "P_DA": P_DA.detach(), "R_up": R_up.detach(), "R_dn": R_dn.detach(),
                   "rt_obj_true": rt_obj_true.detach(), "cost_energy": cost_energy.detach(),
                   "cost_reserve": cost_reserve.detach()}

        if return_aux and not return_filter_aux: return total_realized_cost, aux
        if return_filter_aux and not return_aux: return total_realized_cost, aux_filter
        if return_aux and return_filter_aux: return total_realized_cost, aux, aux_filter
        return total_realized_cost


def DFL_train(
    dfl, train_dataset, args, z_temp=1.0, problem_mode="saa",
    train_mode="dfl", filter_kwargs=None, lambda_div=1e5,
):
    dfl.train()
    device = next(dfl.parameters()).device

    train_bs = int(
        getattr(
            args,
            "train_batch_size",
            getattr(args, "dfl_batch_size", getattr(args, "batch_size", 8))
        )
    )

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

    if train_mode == "filter_only" and getattr(dfl, "scenario_filter", None) is not None:
        for p in dfl.parameters():
            p.requires_grad = False
        for p in dfl.scenario_filter.parameters():
            p.requires_grad = True
        optim_params_filter = [p for p in dfl.scenario_filter.parameters() if p.requires_grad]
        optim_params_predictor = []
        optim = torch.optim.SGD(
            optim_params_filter,
            lr=getattr(args, "lr", getattr(args, "filter_lr", 1e-3)),
            weight_decay=getattr(args, "weight_decay", 0.0),
        )
    else:
        for p in dfl.parameters():
            p.requires_grad = True
        optim_params_filter = [p for p in getattr(dfl, "scenario_filter", []).parameters()] \
            if getattr(dfl, "scenario_filter", None) is not None else []
        optim_params_predictor = [p for p in dfl.predictor.parameters() if p.requires_grad]
        optim = torch.optim.Adam(
            [
                {"params": optim_params_filter, "lr": getattr(args, "filter_lr", 1e-3)},
                {"params": optim_params_predictor, "lr": getattr(args, "dfl_lr", 1e-5)},
            ],
            weight_decay=getattr(args, "weight_decay", 0.0),
        )

    loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_bs,
        shuffle=True,
    )
    epochs = int(getattr(args, "epochs", 1))
    train_losses = []

    for ep in tqdm(range(epochs), desc=f"Train ({train_mode})", leave=True):
        epoch_task_loss, epoch_div_loss, samples_cnt = 0.0, 0.0, 0
        pbar = tqdm(loader, desc=f"Ep {ep+1}/{epochs}", leave=False)

        for X_all, y_true in pbar:
            X_all, y_true = X_all.to(device).float(), y_true.to(device).float()
            optim.zero_grad(set_to_none=True)

            kwargs = dict(
                solver=getattr(args, "solver", None),
                return_aux=False,
                return_filter_aux=True,
                predictor_n_samples=getattr(args, "S_full", getattr(args, "N_scen", 50)),
                predictor_kwargs={"z_temp": z_temp, "cpu_offload": False},
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

            loss = task_loss_val + lambda_div * div_loss_val
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
            pbar.set_postfix(Task=float(task_loss_val), Div=float(div_loss_val), Type=div_type, BS=train_bs)

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
    z_temp,
    problem_mode="saa",
    filter_kwargs=None,
    return_filter_aux=False,
):
    """
    兼容 scenario_filter 的测试函数。
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

    # ---- DRO 参数 ----
    if str(problem_mode).lower() not in ["saa", "so"]:
        Lmin = args.Lmin
        Lmax = args.Lmax
        eps_value = args.eps_value
    else:
        Lmin = Lmax = eps_value = None

    # ---- filter kwargs ----
    if filter_kwargs is None:
        filter_kwargs = {
            "tau_gumbel": float(getattr(args, "tau_mix", 1.0)),
            "eps_uniform": float(getattr(args, "eps_uniform", 0.10)),
        }

    loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        drop_last=False,
    )

    losses = []
    all_filter_aux = [] if return_filter_aux else None

    pbar = tqdm(loader, desc="Test", leave=True)
    for X_all, y_true in pbar:
        X_all = X_all.to(device).float()
        y_true = y_true.to(device).float()

        # 固定 test-time randomness
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
            if Lmin is None or Lmax is None or eps_value is None:
                raise ValueError("mode='dro' requires Lmin, Lmax, eps_value")
            dtype = getattr(dfl, "optnet_dtype", torch.float64)
            kwargs.update(
                hourly_load_min_sys=torch.as_tensor(Lmin, device=device, dtype=dtype),
                hourly_load_max_sys=torch.as_tensor(Lmax, device=device, dtype=dtype),
                eps=torch.as_tensor(eps_value, device=device, dtype=dtype),
            )

        out = dfl(X_all, y_true, **kwargs)

        if return_filter_aux:
            # 兼容 (loss_vec, aux_filter) 或 (loss_vec, aux, aux_filter)
            if isinstance(out, (tuple, list)):
                loss_vec = out[0]
                aux_filter = out[-1]
            else:
                loss_vec = out
                aux_filter = None
            all_filter_aux.append(aux_filter)
        else:
            loss_vec = out[0] if isinstance(out, (tuple, list)) else out

        batch_mean = loss_vec.mean().detach().cpu()
        losses.append(loss_vec.detach().cpu())
        pbar.set_postfix(loss=float(batch_mean))

    loss_all = torch.cat(losses, dim=0)

    if return_filter_aux:
        return loss_all, all_filter_aux
    return loss_all


def run_DFL_vae_separate(
    args,
    problem_mode,
    optimization_mode,
    data_path,
    target_nodes,
    pack_data_s,
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
        raise ValueError(f"eval_flags must be length-5, got {eval_flags}")

    eval_det_before, eval_random_before, eval_stage2_after, eval_stage3_before, eval_stage3_after = map(bool, eval_flags)

    args.eval_mode = str(getattr(args, "eval_mode", "soft")).lower()
    args.avoid_rand_duplicate = bool(getattr(args, "avoid_rand_duplicate", False))

    print("\n[ScenarioFilter eval config]")
    print("eval_mode =", args.eval_mode)
    print("avoid_rand_duplicate =", args.avoid_rand_duplicate)

    train_bs = int(
        getattr(args, "train_batch_size",
            getattr(args, "dfl_batch_size",
                getattr(args, "batch_size", 8)))
    )
    test_bs = int(getattr(args, "test_batch_size", getattr(args, "batch_size", 8)))

    print("args.train_batch_size =", train_bs)
    print("args.test_batch_size  =", test_bs)

    run_stage2 = args.run_stage2
    run_stage3 = args.run_stage3
    reused_stage2 = stage2_artifact is not None
    multi = is_multi(optimization_mode)
    
    if multi:
        SO_Manager, DRO_Manager = IEEE14_Reserve_SO_Manager_MultiNode, IEEE14_Reserve_DRO_Manager_MultiNode
        DET_Manager = IEEE14_Reserve_Deterministic_DA_Manager_MultiNode

        SAA_DA_OptNet, DRO_DA_OptNet = MultiNode_Reserve_SAA_DA_OptNet, MultiNode_Reserve_DRO_DA_OptNet
        DET_DA_OptNet = MultiNode_Reserve_Deterministic_DA_OptNet

        RT_OptNet = MultiNode_Reserve_RT_OptNet
        DFL_SO_Class, DFL_DRO_Class = DFL_model_vae_MultiNode, DFL_model_vae_DRO_MultiNode
        DET_Class = DFL_model_vae_Deterministic_MultiNode
    else:
        SO_Manager, DRO_Manager = IEEE14_Reserve_SO_Manager_SingleNode, IEEE14_Reserve_DRO_Manager_SingleNode
        DET_Manager = IEEE14_Reserve_Deterministic_DA_Manager_SingleNode

        SAA_DA_OptNet, DRO_DA_OptNet = SingleNode_Reserve_SAA_DA_OptNet, SingleNode_Reserve_DRO_DA_OptNet
        DET_DA_OptNet = SingleNode_Reserve_Deterministic_DA_OptNet

        RT_OptNet = SingleNode_Reserve_RT_OptNet
        DFL_SO_Class, DFL_DRO_Class = DFL_model_vae_SingleNode, DFL_model_vae_DRO_SingleNode
        DET_Class = DFL_model_vae_Deterministic_SingleNode

    args.S_full = int(getattr(args, "S_full", 50))
    K = args.N_scen
    K_rand = args.K_rand
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
        best_z = stage2_artifact["best_z"]
    else:
        train_data = Combined_dataset_parametric(
            data_path, target_nodes, "train", 8760, 0.2, 42, y_real=False
        )
        test_data = Combined_dataset_parametric(
            data_path, target_nodes, "test", 8760, 0.2, 42, y_real=False
        )
        best_z = {n: pack_data_s[n]["best_z_temp"] for n in target_nodes}

    def build_predictor(node_models, scaler_y_map):
        return Multi_vae_predictor(
            node_models=copy.deepcopy(node_models),
            scaler_y_map=scaler_y_map,
            node_order=target_nodes,
        ).to(device)

    predictor_before = build_predictor(models_s, train_data.scaler_y_map)
    predictor_train = build_predictor(models_s, train_data.scaler_y_map)

    # ---------- build core ----------
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
            solver="ECOS",
        ).to(device)

    def rebuild_model_like(src_model, predictor_builder, filter_builder, cls_builder):
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
                model, test_data, args, z_temp=best_z, problem_mode=problem_mode
            )
        if "train" in eval_splits:
            set_seed(seed)
            out[f"train_losses_{stage_tag}"] = DFL_test(
                model, train_data, args, z_temp=best_z, problem_mode=problem_mode
            )
        return out

    # 先做前置 baseline eval（仅在不复用 artifact 时）
    if not reused_stage2:
        if eval_det_before:
            stage0_det_eval = eval_on_splits(dfl_det_before, "deterministic_before")
            if "test" in eval_splits:
                print_mean_loss(
                    "Deterministic baseline (before DFL training)",
                    stage0_det_eval,
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

        stage0_det_eval = copy.deepcopy(stage2_artifact.get("stage0_det_eval", stage0_det_eval))
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
        args.epochs = int(getattr(args, "dfl_epochs", 2))
        args.dfl_lr = float(getattr(args, "dfl_lr", 1e-7))
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
            z_temp=best_z,
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

        dfl_after_stage2_snapshot = rebuild_model_like(
            src_model=dfl,
            predictor_builder=lambda: build_predictor(models_s, train_data.scaler_y_map),
            filter_builder=make_filter,
            cls_builder=make_dfl_model,
        )

        dfl_det_before_snapshot = rebuild_model_like(
            src_model=dfl_det_before,
            predictor_builder=lambda: build_predictor(models_s, train_data.scaler_y_map),
            filter_builder=make_filter,
            cls_builder=make_det_model,
        )

        dfl_before_snapshot = rebuild_model_like(
            src_model=dfl_before,
            predictor_builder=lambda: build_predictor(models_s, train_data.scaler_y_map),
            filter_builder=make_filter,
            cls_builder=make_dfl_model,
        )

    else:
        print("\n ---> [skip] Stage A disabled (run_stage2=False), use pre-Stage-A model and jump to Stage B")

        dfl = rebuild_model_like(
            src_model=dfl_before,
            predictor_builder=lambda: build_predictor(models_s, train_data.scaler_y_map),
            filter_builder=make_filter,
            cls_builder=make_dfl_model,
        )

        dfl_after_stage2_snapshot = dfl
        dfl_det_before_snapshot = rebuild_model_like(
            src_model=dfl_det_before,
            predictor_builder=lambda: build_predictor(models_s, train_data.scaler_y_map),
            filter_builder=make_filter,
            cls_builder=make_det_model,
        )
        dfl_before_snapshot = rebuild_model_like(
            src_model=dfl_before,
            predictor_builder=lambda: build_predictor(models_s, train_data.scaler_y_map),
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
        "dfl_det_before": dfl_det_before_snapshot if dfl_det_before_snapshot is not None else None,
        "dfl_before": dfl_before_snapshot if dfl_before_snapshot is not None else None,
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
        "dfl_det_before": dfl_det_before,
        "dfl_before": dfl_before,
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
        "eval_mode": str(getattr(args, "eval_mode", "soft")).lower(),
        "avoid_rand_duplicate": bool(getattr(args, "avoid_rand_duplicate", False)),
        "stage2_artifact": stage2_artifact_out,
    }

    result.update(stage0_det_eval)
    result.update(stage1_eval)
    result.update(stage2_eval)
    result.update(stage3_before_eval)
    result.update(stage3_eval)

    return result



import copy
import torch
from utils import _parse_compare_method_names

def compare_scenario_filters_with_stage3_learned_vae(
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
    Fair comparison under GAN protocol:

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

    multi = is_multi(optimization_mode)

    # -------- managers / classes --------
    if multi:
        SO_Manager, DRO_Manager = IEEE14_Reserve_SO_Manager_MultiNode, IEEE14_Reserve_DRO_Manager_MultiNode
        SAA_DA_OptNet, DRO_DA_OptNet = MultiNode_Reserve_SAA_DA_OptNet, MultiNode_Reserve_DRO_DA_OptNet
        RT_OptNet = MultiNode_Reserve_RT_OptNet
        DFL_SO_Class, DFL_DRO_Class = DFL_model_vae_MultiNode, DFL_model_vae_DRO_MultiNode
    else:
        SO_Manager, DRO_Manager = IEEE14_Reserve_SO_Manager_SingleNode, IEEE14_Reserve_DRO_Manager_SingleNode
        SAA_DA_OptNet, DRO_DA_OptNet = SingleNode_Reserve_SAA_DA_OptNet, SingleNode_Reserve_DRO_DA_OptNet
        RT_OptNet = SingleNode_Reserve_RT_OptNet
        DFL_SO_Class, DFL_DRO_Class = DFL_model_vae_SingleNode, DFL_model_vae_DRO_SingleNode

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
        return Multi_vae_predictor(
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
        "backbone_predictor_source": "stage2_artifact['dfl_after_stage2']",
        "learned_filter_source": "base_result['dfl_trained'].scenario_filter",
        "method_names": list(method_names),
        "eval_splits": eval_splits,
        "details": {},
        "summary_mean": {},
    }

    if verbose:
        print("\n" + "=" * 90)
        print("[Compare Scenario Filters - GAN]")
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