import os
import json
import time
import copy
import random
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm, chi2
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from scenarios_reduce import *

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def load_ieee14_data():
    """
    返回 IEEE 14 节点的原始物理参数 (Standard IEEE Data)
    """
    # 原始发电机数据 (Min MW, Max MW, Cost Coeffs)
    # 注意：这里的 Max 是原始值，后续会乘以 args.capacity_scale
    gen_raw = {
        1: {'a': 0.04, 'b': 20, 'c': 0, 'min': 0, 'max': 332},
        2: {'a': 0.25, 'b': 20, 'c': 0, 'min': 0, 'max': 140},
        3: {'a': 0.01, 'b': 40, 'c': 0, 'min': 0, 'max': 100},
        6: {'a': 0.01, 'b': 40, 'c': 0, 'min': 0, 'max': 100},
        8: {'a': 0.01, 'b': 40, 'c': 0, 'min': 0, 'max': 100}
    }
    
    # 原始线路数据: (from, to): Reactance(p.u.)
    line_raw = {
        (1, 2): 0.05917, (1, 5): 0.22304, (2, 3): 0.19797, (2, 4): 0.17632, (2, 5): 0.17388,
        (3, 4): 0.17103, (4, 5): 0.04211, (4, 7): 0.20912, (4, 9): 0.55618,
        (5, 6): 0.25202, (6, 11): 0.19890, (6, 12): 0.25581, (6, 13): 0.13027,
        (7, 8): 0.17615, (7, 9): 0.11001, (9, 10): 0.08450, (9, 14): 0.27038,
        (10, 11): 0.19207, (12, 13): 0.19988, (13, 14): 0.34802
    }
    
    return gen_raw, line_raw



def choose_grid(n: int):
    """
    Choose (rows, cols) for nicer layout.
    Preferences:
      - 1 -> 1x1
      - 2 -> 1x2
      - 3 -> 1x3
      - 4 -> 2x2
      - 5/6 -> 2x3
      - 7 -> 3x3 (2*3 + 1, last one centered)
      - 8 -> 2x4
      - 9 -> 3x3
      - >=10 -> near-square (favor cols=4 then 5)
    """
    if n <= 0:
        raise ValueError("n must be positive")

    hard = {
        1: (1, 1),
        2: (1, 2),
        3: (1, 3),
        4: (2, 2),
        5: (2, 3),
        6: (2, 3),
        7: (3, 3),  # 2*3 + 1 (center last)
        8: (2, 4),
        9: (3, 3),
    }
    if n in hard:
        return hard[n]

    # generic near-square for n>=10
    # try cols in [4,5,3,6] order
    for cols in [4, 5, 3, 6]:
        rows = (n + cols - 1) // cols
        if (rows - 1) * cols < n <= rows * cols:
            return rows, cols

    # fallback
    cols = int(np.ceil(np.sqrt(n)))
    rows = (n + cols - 1) // cols
    return rows, cols

def center_last_single(fig, axes, rows, cols):
    """
    If last row has exactly 1 active subplot (i.e., n = (rows-1)*cols + 1),
    center it by moving its position to the middle columns.
    Only works when cols >= 3.
    """
    if cols < 3:
        return

    # axes is a 1D list length rows*cols
    # last row indices: (rows-1)*cols ... rows*cols-1
    last_row_start = (rows - 1) * cols
    last_row_axes = axes[last_row_start: last_row_start + cols]

    # find the only visible ax in that last row
    visible = [ax for ax in last_row_axes if ax.axison]
    if len(visible) != 1:
        return
    ax = visible[0]

    # take the middle cell position as target
    mid_col = cols // 2
    target_ax = last_row_axes[mid_col]
    pos = target_ax.get_position()

    ax.set_position(pos)


def compute_metrics_window(window_pack, quantiles=None):
    """
    window_pack from sample_window_single or sample_window_multi
    returns df_metrics per node
    """
    if quantiles is None:
        quantiles = [i / 10 for i in range(1, 10)]

    Y_true = np.asarray(window_pack["Y_true"])  # [N,L]
    Y_pred = np.asarray(window_pack["Y_pred"])  # [S,N,L]
    target_nodes = list(window_pack["target_nodes"])
    N, L = Y_true.shape

    def pinball(y_true, y_q, q):
        diff = y_true - y_q
        return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))

    rows = []
    for i in range(N):
        y_true = Y_true[i]
        y_mean = Y_pred[:, i, :].mean(axis=0)
        mse = float(np.mean((y_mean - y_true) ** 2))
        rmse = float(np.sqrt(mse))

        pb = []
        for q in quantiles:
            y_q = np.quantile(Y_pred[:, i, :], q, axis=0)
            pb.append(pinball(y_true, y_q, q))

        rows.append(dict(node=target_nodes[i], L=L, mse=mse, rmse=rmse, pinball_avg=float(np.mean(pb))))

    return pd.DataFrame(rows)



def plot_window_curve(window_pack, figsize=None, print_metrics=True):
    Y_true = np.asarray(window_pack["Y_true"])   # [N,L]
    Y_pred = np.asarray(window_pack["Y_pred"])   # [S,N,L]
    nodes = list(window_pack["target_nodes"])
    N, L = Y_true.shape

    Y_mean = Y_pred.mean(axis=0)
    Y_p10  = np.percentile(Y_pred, 10, axis=0)
    Y_p90  = np.percentile(Y_pred, 90, axis=0)

    if print_metrics:
        dfm = compute_metrics_window(window_pack)
        print(dfm.to_string(index=False))

    rows, cols = choose_grid(N)
    if figsize is None:
        figsize = (cols * 6.0, rows * 3.6)
    fig, axes = plt.subplots(rows, cols, figsize=figsize, sharex=False)
    axes = np.atleast_1d(axes).reshape(-1)

    t = np.arange(L)
    for i in range(N):
        ax = axes[i]
        ax.plot(t, Y_true[i], color="black", lw=1.6, label="GT")
        ax.plot(t, Y_mean[i], color="blue", lw=1.2, ls="--", label="Mean")
        ax.fill_between(t, Y_p10[i], Y_p90[i], color="blue", alpha=0.25, label="10-90%")

        for day in range(1, int(np.ceil(L / 24))):
            ax.axvline(day * 24, color="gray", ls=":", alpha=0.6)

        ax.set_title(nodes[i])
        ax.grid(True, alpha=0.25)
        if i == 0:
            ax.legend(loc="upper right", frameon=True)
        if i % cols == 0:
            ax.set_ylabel("Load")
        if i >= (rows - 1) * cols:
            ax.set_xlabel("Hour")

    for j in range(N, len(axes)):
        axes[j].axis("off")

    if N == (rows - 1) * cols + 1:
        center_last_single(fig, axes, rows, cols)

    plt.tight_layout()
    plt.suptitle(
        f"Single Window | start_day={window_pack['start_day']}, days={window_pack['horizon_days']}, S={window_pack['n_samples']}",
        y=1.02
    )
    plt.show()



def fit_ellipsoid_topk_scipy(points_2d, keep=0.9):
    pts = np.asarray(points_2d, dtype=np.float64)
    mu = pts.mean(axis=0)
    Sigma = np.cov(pts.T) + 1e-12 * np.eye(2)
    kappa = chi2.ppf(keep, df=2)
    return mu, Sigma, kappa, None

def fit_ellipsoid_topk(points, keep=0.95, ridge=1e-6):
    """
    points: (N,d) torch.Tensor (or array-like, will be converted)
    fits (x-mu)^T Sigma^{-1} (x-mu) <= kappa covering keep fraction (empirical).
    returns: mu(d,), Sigma(d,d), kappa(scalar), d2(N,)
    """
    X = points if torch.is_tensor(points) else torch.as_tensor(points)
    X = X.float() if X.dtype in (torch.float16, torch.bfloat16) else X

    N, d = X.shape
    mu = X.mean(dim=0)                 # (d,)
    Xc = X - mu                        # (N,d)

    Sigma = (Xc.T @ Xc) / N            # (d,d)
    Sigma = Sigma + ridge * torch.eye(d, device=X.device, dtype=X.dtype)

    Y = torch.linalg.solve(Sigma, Xc.T).T
    d2 = (Xc * Y).sum(dim=1)           # (N,)

    kappa = torch.quantile(d2, keep)
    return mu, Sigma, kappa, d2

def plot_ellipsoid_2d(ax, mu, Sigma, kappa, color="crimson", lw=2, label=None):
    """plots boundary of (x-mu)^T Sigma^{-1} (x-mu) = kappa for 2D."""
    vals, vecs = np.linalg.eigh(Sigma)
    vals = np.maximum(vals, 1e-12)

    t = np.linspace(0, 2*np.pi, 400)
    circle = np.vstack([np.cos(t), np.sin(t)])

    A = vecs @ np.diag(np.sqrt(kappa * vals))
    ell = mu.reshape(2,1) + A @ circle
    ax.plot(ell[0], ell[1], color=color, lw=lw, label=label)


def draw_ellipsoid(x,y,mu,Sigma,kappa):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.scatter(x, y, s=12, alpha=0.5, label="samples")
    ax.scatter([mu[0]], [mu[1]], c="black", s=60, marker="x", label="mu")
    plot_ellipsoid_2d(ax, mu, Sigma, kappa, color="crimson", lw=2, label="95% ellipsoid")

    ax.set_title("Node1 vs Node2 at time=3 (top-k 95% ellipsoid)")
    ax.set_xlabel("node 1 value (t=3)")
    ax.set_ylabel("node 2 value (t=3)")
    ax.axis("equal")
    ax.grid(True, ls="--", alpha=0.3)
    ax.legend()
    plt.show()

def MAPE(y_actual,y_predicted):
    if isinstance(y_actual, torch.Tensor):
        y_actual = y_actual.detach().cpu().numpy()
    if isinstance(y_predicted, torch.Tensor):
        y_predicted = y_predicted.detach().cpu().numpy()
    MAPE=np.mean(abs((y_actual-y_predicted)/y_actual))
    return MAPE

def R2(y_actual,y_predicted):
    if isinstance(y_actual, torch.Tensor):
        y_actual = y_actual.detach().cpu().numpy()
    if isinstance(y_predicted, torch.Tensor):
        y_predicted = y_predicted.detach().cpu().numpy()
    R2 = 1 - np.sum(np.square(y_actual-y_predicted)) / np.sum(np.square(y_actual-np.mean(y_actual)))
    return R2

def RMSE(y_actual,y_predicted):
    if isinstance(y_actual, torch.Tensor):
        y_actual = y_actual.detach().cpu().numpy()
    if isinstance(y_predicted, torch.Tensor):
        y_predicted = y_predicted.detach().cpu().numpy()
    RMSE = np.sqrt(np.mean(np.square(y_actual-y_predicted)))
    return RMSE

def MAE(y_actual,y_predicted):
    if isinstance(y_actual, torch.Tensor):
        y_actual = y_actual.detach().cpu().numpy()
    if isinstance(y_predicted, torch.Tensor):
        y_predicted = y_predicted.detach().cpu().numpy()
    return np.mean(np.abs(y_actual-y_predicted))

def get_load_data(args,flag='train'):
    if flag=='train':
        shuffle_flag=True
        drop_last=True
    elif flag == 'val':
        shuffle_flag=False
        drop_last=False
    else:
        shuffle_flag=False
        drop_last=False
    data_set=Dataset_load(flag=flag,size=[args.seq_len,args.label_len,args.pred_len])
    
    data_loader = DataLoader(
        data_set,
        batch_size=args.batch_size,
        shuffle=shuffle_flag,
        drop_last=drop_last)
    
    return data_set,data_loader

def pinball_loss_calculation(quantiles_list_inversed, label, quantile=[]):
    loss=[]
    for i in range(len(quantile)):
        errors = label - quantiles_list_inversed[quantile[i]]
        loss.append(np.maximum(quantile[i] * errors, (quantile[i] - 1) * errors))
    return np.mean(loss)

def corr_mats_mean_and_residual(Y_true, Y_scenarios, *, eps=1e-12, method="pearson"):
    """
    Compute (1) mean-correlation and (2) residual-correlation across nodes.

    Parameters
    ----------
    Y_true : array-like, shape [N, L] or [L, N]
        Ground-truth series for N nodes over L timestamps (e.g., L=303*24).
    Y_scenarios : array-like, shape [S, N, L] or [S, L, N]
        Scenario samples from the model.
    eps : float
        Numerical stabilizer.
    method : {"pearson"} currently

    Returns
    -------
    out : dict with keys
        - C_true:      [N,N] corr of true series
        - C_pred_mean: [N,N] corr of scenario-mean series
        - C_pred_eps:  [N,N] corr of scenario residuals (joint dependence of uncertainty)
        - mean_series: [L,N] scenario mean time series
        - eps_samples: [(S*L),N] stacked residual samples used for C_pred_eps
    """
    Yt = np.asarray(Y_true)
    Ys = np.asarray(Y_scenarios)

    # --- normalize shapes ---
    # Y_true -> [L,N]
    if Yt.ndim != 2:
        raise ValueError("Y_true must be 2D.")
    if Yt.shape[0] < Yt.shape[1]:  # likely [N,L]
        Yt_LN = Yt.T
    else:
        Yt_LN = Yt

    # Y_scenarios -> [S,N,L]
    if Ys.ndim != 3:
        raise ValueError("Y_scenarios must be 3D.")
    if Ys.shape[1] == Yt_LN.shape[1] and Ys.shape[2] == Yt_LN.shape[0]:
        Ys_SNL = Ys
    elif Ys.shape[2] == Yt_LN.shape[1] and Ys.shape[1] == Yt_LN.shape[0]:
        Ys_SNL = np.transpose(Ys, (0, 2, 1))  # [S,L,N] -> [S,N,L]
    else:
        raise ValueError(
            f"Cannot align shapes. Got Y_true [L,N]={Yt_LN.shape}, Y_scenarios={Ys.shape}."
        )

    S, N, L = Ys_SNL.shape

    # --- mean correlation (point forecast from scenario mean) ---
    mean_series = Ys_SNL.mean(axis=0).T  # [L,N]

    # --- residual correlation (uncertainty dependence) ---
    mu = Ys_SNL.mean(axis=0, keepdims=True)   # [1,N,L]
    eps_snl = Ys_SNL - mu                     # [S,N,L]
    # stack (S,L) as samples: [(S*L), N]
    eps_samples = np.transpose(eps_snl, (0, 2, 1)).reshape(S * L, N)

    # corrcoef expects rows as variables if rowvar=True; we want variables in columns => rowvar=False
    if method == "pearson":
        C_true = np.corrcoef(Yt_LN, rowvar=False)
        C_pred_mean = np.corrcoef(mean_series, rowvar=False)

        # residual corr: handle near-constant columns (rare but possible)
        # add eps by slight jitter if needed
        if np.any(np.std(eps_samples, axis=0) < eps):
            eps_samples = eps_samples + eps * np.random.randn(*eps_samples.shape)
        C_pred_eps = np.corrcoef(eps_samples, rowvar=False)
    else:
        raise NotImplementedError("Only pearson implemented.")

    return dict(
        C_true=C_true,
        C_pred_mean=C_pred_mean,
        C_pred_eps=C_pred_eps,
        mean_series=mean_series,
        eps_samples=eps_samples,
    )


def system_hourly_load_minmax(df, datetime_col="DATETIME", node_cols=None, hours=24):
    d = df.copy()
    d[datetime_col] = pd.to_datetime(d[datetime_col])

    #d["sys_load"] = d[node_cols].sum(axis=1)
    d["hour"] = d[datetime_col].dt.hour

    Lmin=np.zeros((len(node_cols),hours))
    Lmax=np.zeros((len(node_cols),hours))
    for i in range(len(node_cols)):
        grp = d.groupby("hour")[node_cols[i]]
        Lmin[i] = np.array(grp.min())
        Lmax[i] = np.array(grp.max())

    return Lmin, Lmax


def save_run_result(
    result,
    forecasting_mode="joint",
    optimization_mode=None,
    problem_mode=None,
    out_dir="./Result/VAE",
):
    import os, pickle
    import numpy as np
    import torch

    os.makedirs(out_dir, exist_ok=True)

    def as_np_float32(x):
        if x is None:
            return None
        if hasattr(x, "detach"):
            x = x.detach().cpu().numpy()
        return np.asarray(x, dtype=np.float32)

    if optimization_mode is None:
        optimization_mode = result["optimization_mode"]
    if problem_mode is None:
        problem_mode = result["problem_mode"]

    tag = f"{forecasting_mode}_{optimization_mode}_{problem_mode}"

    # ---------- summary print ----------
    stages = [
        {
            "name": "Deterministic Baseline (before DFL training)",
            "test_key": "test_losses_deterministic_before",
            "train_key": "train_losses_deterministic_before",
        },
        {
            "name": "Random Baseline (before DFL training)",
            "test_key": "test_losses_stage1_after",
            "train_key": "train_losses_stage1_after",
        },
        {
            "name": "Random Filter AFTER DFL training",
            "test_key": "test_losses_stage2_after",
            "train_key": "train_losses_stage2_after",
        },
        {
            "name": "Fresh ScenarioFilter BEFORE training",
            "test_key": "test_losses_stage3_before",
            "train_key": "train_losses_stage3_before",
        },
        {
            "name": "ScenarioFilter AFTER training",
            "test_key": "test_losses_stage3_after",
            "train_key": "train_losses_stage3_after",
        },
    ]

    print("\n" + "=" * 60)
    print(f"  Final Results Summary ({forecasting_mode.upper()} | {optimization_mode.upper()} | {problem_mode.upper()})")
    print("=" * 60)
    for s in stages:
        tv, rv = result.get(s["test_key"]), result.get(s["train_key"])
        if tv is not None or rv is not None:
            print(f" -> {s['name']}:")
            if tv is not None:
                print(f"    [TEST ] Mean Loss: {float(np.mean(as_np_float32(tv))):.6f}")
            if rv is not None:
                print(f"    [TRAIN] Mean Loss: {float(np.mean(as_np_float32(rv))):.6f}")
    print("=" * 60 + "\n")

    ckpt_path = os.path.join(out_dir, f"DFL_model_trained_{tag}.pkl")
    with open(ckpt_path, 'wb') as f:
        pickle.dump(result, f)

    print(f"Saved at: {ckpt_path}")



import copy
import torch

def build_scenario_baseline_filter(name, args, device):
    """
    Build one scenario selection baseline filter by name.

    Supported:
        - random
        - kmeans
        - kmedoids
        - hierarchical
    """
    name = str(name).lower()
    K = args.N_scen

    feature_mode = str(getattr(args, "cluster_feature_mode", "sum")).lower()
    random_state = int(getattr(args, "cluster_random_state", 0))
    kmeans_n_init = int(getattr(args, "kmeans_n_init", 10))
    kmedoids_metric = str(getattr(args, "kmedoids_metric", "euclidean")).lower()
    hierarchical_linkage = str(getattr(args, "hierarchical_linkage", "ward")).lower()

    if name == "random":
        return RandomScenarioSelector(n_scen=K).to(device)

    elif name == "kmeans":
        return KMeansScenarioFilter(
            K=K,
            feature_mode=feature_mode,
            random_state=random_state,
            n_init=kmeans_n_init,
        ).to(device)

    elif name == "kmedoids":
        return KMedoidsScenarioFilter(
            K=K,
            feature_mode=feature_mode,
            metric=kmedoids_metric,
            random_state=random_state,
        ).to(device)

    elif name in {"hierarchical", "hc", "agglomerative"}:
        return HierarchicalScenarioFilter(
            K=K,
            feature_mode=feature_mode,
            linkage=hierarchical_linkage,
        ).to(device)

    else:
        raise ValueError(
            f"Unknown baseline filter: {name}. "
            f"Supported: random, kmeans, kmedoids, hierarchical"
        )

def _parse_compare_method_names(method_names):
    """
    Parse method names from:
        - None
        - 'learned,random,kmeans'
        - ['learned', 'random', 'kmeans']
    """
    if method_names is None:
        return ["learned", "random", "kmeans", "kmedoids", "hierarchical"]

    if isinstance(method_names, str):
        method_names = [x.strip().lower() for x in method_names.split(",") if x.strip()]
    elif isinstance(method_names, (list, tuple, set)):
        method_names = [str(x).strip().lower() for x in method_names if str(x).strip()]
    else:
        raise ValueError(
            f"Unsupported method_names={method_names}. "
            f"Use None, 'learned,random,kmeans', or a list/tuple/set."
        )

    alias = {
        "hc": "hierarchical",
        "agglomerative": "hierarchical",
    }
    method_names = [alias.get(x, x) for x in method_names]

    valid = {"learned", "random", "kmeans", "kmedoids", "hierarchical"}
    for x in method_names:
        if x not in valid:
            raise ValueError(f"Unknown method name: {x}, valid={sorted(valid)}")

    return method_names


import copy

def merge_compare_results_learned_as_variants(compare_res_single_dro_kl,
                                             compare_res_single_dro_inner,
                                             compare_res_single_dro_entropy,
                                             learned_key="learned",
                                             labels=("KL", "Inner", "Entropy"),
                                             keep_baselines_from="kl"):
    """
    合并三份 compare_result，输出结构与原来一致，只是把 learned 拆成：
      learned (KL), learned (Inner), learned (Entropy)
    其它 random/kmeans/... 只保留一份（默认取 KL 那份）。

    返回：merged_compare_result (dict)
    """
    def rename_learned(res, new_learned_name):
        res = copy.deepcopy(res)

        # method_names
        res["method_names"] = [new_learned_name if m == learned_key else m
                               for m in res.get("method_names", [])]

        # summary_mean
        if "summary_mean" in res and learned_key in res["summary_mean"]:
            res["summary_mean"][new_learned_name] = res["summary_mean"].pop(learned_key)

        # details：把 learned 这一项整体挪到新名字，同时把内部 key 的后缀也替换掉
        if "details" in res and learned_key in res["details"]:
            old_block = res["details"].pop(learned_key)
            new_block = {}
            for k, v in old_block.items():
                # k 形如 test_losses_compare_stage2backbone_learned
                new_k = k.replace(f"_{learned_key}", f"_{new_learned_name}")
                new_block[new_k] = v
            res["details"][new_learned_name] = new_block

        return res

    r_kl   = rename_learned(compare_res_single_dro_kl,      f"{learned_key} ({labels[0]})")
    r_in   = rename_learned(compare_res_single_dro_inner,   f"{learned_key} ({labels[1]})")
    r_ent  = rename_learned(compare_res_single_dro_entropy, f"{learned_key} ({labels[2]})")

    base = {"kl": r_kl, "inner": r_in, "entropy": r_ent}[keep_baselines_from]
    merged = copy.deepcopy(base)

    # baselines：非 learned(...) 的方法名（只从 base 取一份）
    baselines = [m for m in merged.get("method_names", [])
                 if not (isinstance(m, str) and m.startswith(f"{learned_key} ("))]

    # 清掉 base 里 learned(...)，后面按顺序重新加 3 个 learned(...)
    for m in list(merged.get("method_names", [])):
        if isinstance(m, str) and m.startswith(f"{learned_key} ("):
            merged["method_names"].remove(m)

    for key in list(merged.get("details", {}).keys()):
        if isinstance(key, str) and key.startswith(f"{learned_key} ("):
            merged["details"].pop(key)

    for key in list(merged.get("summary_mean", {}).keys()):
        if isinstance(key, str) and key.startswith(f"{learned_key} ("):
            merged["summary_mean"].pop(key)

    # 追加三份 learned(...)
    for rr in (r_kl, r_in, r_ent):
        learned_names = [m for m in rr.get("method_names", [])
                         if isinstance(m, str) and m.startswith(f"{learned_key} (")]
        if len(learned_names) != 1:
            raise ValueError(f"Cannot uniquely find learned variant in one result: {learned_names}")
        ln = learned_names[0]

        merged.setdefault("details", {})[ln] = rr.get("details", {}).get(ln, {})
        merged.setdefault("summary_mean", {})[ln] = rr.get("summary_mean", {}).get(ln, {})
        merged.setdefault("method_names", []).append(ln)

    # 最终 method_names 顺序：learned(KL/Inner/Entropy) + baselines
    merged["method_names"] = [f"{learned_key} ({labels[0]})",
                              f"{learned_key} ({labels[1]})",
                              f"{learned_key} ({labels[2]})"] + baselines

    return merged



def summarize_compare_result(compare_result):
    rows = []
    summary_mean = compare_result.get("summary_mean", {})
    for method, info in summary_mean.items():
        row = {"method": method}
        if isinstance(info, dict):
            row.update(info)
        rows.append(row)
    return rows