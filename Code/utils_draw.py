import matplotlib.pyplot as plt
import numpy as np
from utils import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, chi2



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


def plot_3metrics_compare_bar(df1, df2, name1="dfm_m", name2="dfm_s",
                              metrics=("mse","rmse","pinball_avg"),
                              node_col="node",
                              y_from_zero=False,
                              pad=0.08):
    def node_id(s):
        return int(str(s).split("-")[-1])

    df1 = df1.copy()
    df2 = df2.copy()
    df1["node_id"] = df1[node_col].map(node_id)
    df2["node_id"] = df2[node_col].map(node_id)

    df1 = df1.sort_values("node_id")
    df2 = df2.sort_values("node_id")

    x = np.arange(len(df1))
    w = 0.38
    xticklabels = df1["node_id"].to_numpy()

    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 10), sharex=True)
    if len(metrics) == 1:
        axes = [axes]

    label_map = {"mse": "MSE", "rmse": "RMSE", "pinball_avg": "Pinball loss"}

    for ax, met in zip(axes, metrics):
        v1 = df1[met].to_numpy(dtype=float)
        v2 = df2[met].to_numpy(dtype=float)

        ax.bar(x - w/2, v1, width=w, label=name1, color="#8ECFC9")
        ax.bar(x + w/2, v2, width=w, label=name2, color="#FFBE7A")

        ax.set_ylabel(label_map.get(met, met))
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()

        # 关键：y轴不从0开始
        ymin = min(v1.min(), v2.min())
        ymax = max(v1.max(), v2.max())
        rng = max(1e-12, ymax - ymin)

        if y_from_zero:
            ax.set_ylim(0, ymax + pad * rng)
        else:
            ax.set_ylim(ymin - pad * rng, ymax + pad * rng)

    axes[-1].set_xlabel("node id (0-10)")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(xticklabels)

    plt.tight_layout()
    plt.show()


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

    legend_handles, legend_labels = None, None

    for i in range(N):
        ax = axes[i]
        ax.plot(t, Y_true[i], color="black", lw=1.6, label="Actual load")
        ax.plot(t, Y_mean[i], color="blue", lw=1.2, ls="--", label="Mean")
        ax.fill_between(t, Y_p10[i], Y_p90[i], color="blue", alpha=0.25, label="10-90%")

        for day in range(1, int(np.ceil(L / 24))):
            ax.axvline(day * 24, color="gray", ls=":", alpha=0.6)

        ax.set_title(nodes[i])
        ax.grid(True, alpha=0.25)
        if i % cols == 0:
            ax.set_ylabel("Load")
        if i >= (rows - 1) * cols:
            ax.set_xlabel("Hour")

        # 只从第一个子图收集legend元素
        if i == 0:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    for j in range(N, len(axes)):
        axes[j].axis("off")

    if N == (rows - 1) * cols + 1:
        center_last_single(fig, axes, rows, cols)

    # 给底部legend留空间：rect的bottom调大一点
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))

    # 全局标题
    fig.suptitle(
        f"Single Window | start_day={window_pack['start_day']}, days={window_pack['horizon_days']}, S={window_pack['n_samples']}",
        y=0.995
    )

    # 全图底部legend
    if legend_handles is not None:
        fig.legend(
            legend_handles, legend_labels,
            loc="lower center", ncol=3, frameon=True,
            bbox_to_anchor=(0.5, 0.01)
        )

    plt.show()


def plot_joint_ellipsoid_no_norm(
    packs,                 # pack 或 [pack1, pack2, ...]
    node_a,
    node_b,
    time_step=0,
    labels=None,           # 多个 packs 时可传，如 ["Single","Multi"]
    node_names=None,
    confidence=0.95,
    figsize=(7.5, 7.5),
    alpha_scatter=0.70,
    s_scatter=16,
    show_truth=True,
    truth_pack_index=0,    # truth 从第几个pack取
):
    # --- 把单个 pack 包成 list（这不是数据标准化，只是为了统一循环） ---
    if not isinstance(packs, (list, tuple)):
        packs = [packs]

    if labels is None:
        labels = [f"Model{i}" for i in range(len(packs))]
    if len(labels) != len(packs):
        raise ValueError("labels 长度必须和 packs 数量一致")

    fig, ax = plt.subplots(figsize=figsize)

    # --- truth（只画一次）---
    if show_truth:
        Y_true = packs[truth_pack_index]["Y_true"]
        if hasattr(Y_true, "cpu"): Y_true = Y_true.cpu().numpy()
        if Y_true.ndim == 3:       Y_true = Y_true[0]  # (1,N,T) -> (N,T)

        true_a = float(Y_true[node_a, time_step])
        true_b = float(Y_true[node_b, time_step])

        ax.scatter(true_a, true_b, c="black", s=260, marker="*",
                   edgecolors="white", linewidths=1.2, zorder=10, label="Ground Truth")

    # 置信椭圆通用量
    scale = np.sqrt(chi2.ppf(confidence, df=2))
    t = np.linspace(0, 2*np.pi, 240)
    circle = np.vstack([np.cos(t), np.sin(t)])  # (2,M)

    # --- 每个 pack：散点 + 椭圆 ---
    for i, pack in enumerate(packs):
        lab = labels[i]

        Y_pred = pack["Y_pred"]
        if hasattr(Y_pred, "cpu"): Y_pred = Y_pred.cpu().numpy()

        # 你的数据形状： (S, N, T)
        samp_a = Y_pred[:, node_a, time_step]
        samp_b = Y_pred[:, node_b, time_step]

        corr_val, _ = pearsonr(samp_a, samp_b)
        mu = np.array([np.mean(samp_a), np.mean(samp_b)])
        cov = np.cov(samp_a, samp_b)

        # 椭圆：eigh 更稳
        eig_vals, eig_vecs = np.linalg.eigh(cov)
        eig_vals = np.maximum(eig_vals, 1e-12)
        A = eig_vecs @ np.diag(np.sqrt(eig_vals) * scale)
        ell = mu.reshape(2, 1) + A @ circle

        ax.scatter(samp_a, samp_b, s=s_scatter, alpha=alpha_scatter,
                   edgecolors="none", label=f"{lab} samples (corr={corr_val:.2f})")
        ax.plot(ell[0], ell[1], lw=2, label=f"{lab} {int(confidence*100)}% ellipse")

    name_a = f"Node {node_a}" if node_names is None else node_names[node_a]
    name_b = f"Node {node_b}" if node_names is None else node_names[node_b]
    ax.set_xlabel(f"{name_a} value")
    ax.set_ylabel(f"{name_b} value")
    ax.set_title(f"Joint @ time_step={time_step}")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.legend(loc="best", frameon=True)

    plt.tight_layout()
    plt.show()


def _annotate_heatmap(ax, M, fmt="{:.2f}", fontsize=9, color="black",
                      thresh=None, diag=True):
    n, m = M.shape
    for i in range(n):
        for j in range(m):
            if (not diag) and (i == j):
                continue
            v = M[i, j]
            if np.isnan(v):
                continue
            if (thresh is not None) and (abs(v) < thresh):
                continue
            ax.text(j, i, fmt.format(v), ha="center", va="center",
                    fontsize=fontsize, color=color)

def plot_corr_heatmaps(
    C_single, C_multi, *,
    title,
    model_name=None,                 # 例如 "VAE" / "GAN"
    node_names=None,
    vlim=(-1, 1),
    cmap="coolwarm",
    figsize=(9, 4.2),
    annot=False,
    fmt="{:.2f}",
    annot_fontsize=10,               # 数字变大
    annot_color="black",
    annot_thresh=None,
    annot_offdiag_only=False,
    grid=True,       
    cbar_pad=0.05                # 加格子线，读数更清晰
):
    mats = [C_single, C_multi]
    panel_names = ["Separate", "Joint"]

    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    #fig.subplots_adjust(wspace=)   # <<< 控制两子图间距

    ims = []
    for ax, M, nm in zip(axes, mats, panel_names):
        im = ax.imshow(M, vmin=vlim[0], vmax=vlim[1], cmap=cmap, interpolation="nearest")
        ims.append(im)

        # 标题里带上 model 名称
        if model_name is None:
            ax.set_title(nm)
        else:
            ax.set_title(f"{model_name}-{nm}")

        # ticks / labels
        n = M.shape[0]
        if node_names is not None:
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(node_names, rotation=90)
            ax.set_yticklabels(node_names)
        else:
            ax.set_xticks([])
            ax.set_yticks([])

        # 画格子线（不影响 imshow）
        if grid:
            ax.set_xticks(np.arange(-.5, n, 1), minor=True)
            ax.set_yticks(np.arange(-.5, n, 1), minor=True)
            ax.grid(which="minor", color="k", linestyle="-", linewidth=0.3, alpha=0.25)
            ax.tick_params(which="minor", bottom=False, left=False)

        # 数字标注
        if annot:
            _annotate_heatmap(
                ax, M,
                fmt=fmt,
                fontsize=annot_fontsize,
                color=annot_color,
                thresh=annot_thresh,
                diag=not annot_offdiag_only
            )

    # 总标题
    #fig.suptitle(title if model_name is None else f"{title} ({model_name})", fontsize=16)

    # colorbar
    cbar = fig.colorbar(ims[-1], ax=axes, shrink=0.92, pad=cbar_pad)
    cbar.ax.tick_params(labelsize=12)

    return fig, axes


def _confidence_ellipse_points(x, y, confidence=0.95, n_theta=240):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.zeros((2, n_theta), dtype=float)

    mu = np.array([x.mean(), y.mean()], dtype=float)
    cov = np.cov(x, y)
    eig_vals, eig_vecs = np.linalg.eigh(cov)
    eig_vals = np.maximum(eig_vals, 1e-12)

    scale = np.sqrt(chi2.ppf(confidence, df=2))
    theta = np.linspace(0, 2 * np.pi, n_theta)
    circle = np.vstack([np.cos(theta), np.sin(theta)])
    A = eig_vecs @ np.diag(np.sqrt(eig_vals) * scale)
    ell = mu.reshape(2, 1) + A @ circle
    return ell

def plot_test_sampling_joint_one_time(
    data,
    nufilter,
    day,
    hour,
    prob_type="single",
    node_a=0,
    node_b=6,
    T=24,
    K=20,
    K_rand=10,
    tau_fixed=1.0,
    tau_mix=1.0,
    eps_uniform=0.10,
    confidence=0.95,
    figsize=(8, 7),
    alpha_full=0.15,
    alpha_rand=1.0,   # 改：随机点更醒目
    alpha_mix=0.65,   # 改：橙色稍透明
    s_full=12,
    s_rand=38,        # 改：随机点稍大
    s_mix=45,
    show_truth=True,
    title_prefix="Scenario Selection Visualization",
    save_path=None,
    device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Y_pred = np.asarray(data["Y_pred"])  # (S, N, L)
    if Y_pred.ndim != 3:
        raise ValueError(f"data['Y_pred'] must be (S,N,L), got {Y_pred.shape}")
    S_full, N_nodes, L = Y_pred.shape

    t_idx = day * T + hour
    if t_idx < 0 or t_idx >= L:
        raise IndexError(f"t_idx={t_idx} out of range for L={L}")
    if not (0 <= node_a < N_nodes and 0 <= node_b < N_nodes):
        raise IndexError(f"node_a/node_b out of range [0,{N_nodes-1}]")

    sl = slice(day * T, (day + 1) * T)
    if sl.stop > L:
        raise IndexError(f"day={day} with T={T} exceeds sequence length L={L}")

    Y_full_SNT = Y_pred[:, :, sl]
    Y_scen_t = torch.tensor(Y_full_SNT, dtype=torch.float32, device=device).unsqueeze(1)  # [S,1,N,T]

    nufilter = nufilter.to(device).eval()

    aux = {}
    idx_rand_np = np.array([], dtype=int)
    Y_mix_KNT = None

    with torch.no_grad():
        try:
            Y_sel, aux_filter = nufilter(Y_scen_t, is_train=False)
            if isinstance(Y_sel, torch.Tensor) and Y_sel.dim() == 4:
                Y_mix_KNT = Y_sel[:, 0].detach().cpu().numpy()  # [K,N,T]
                aux = aux_filter if isinstance(aux_filter, dict) else {}
                if isinstance(aux, dict) and "idx_rand" in aux and aux["idx_rand"] is not None:
                    idx_rand = aux["idx_rand"]
                    if torch.is_tensor(idx_rand):
                        idx_rand_np = idx_rand.detach().cpu().numpy().astype(int)
                        if idx_rand_np.ndim > 1:   # 兼容 [B,K]
                            idx_rand_np = idx_rand_np[0]
                    else:
                        idx_rand_np = np.asarray(idx_rand, dtype=int)
            else:
                raise RuntimeError("ScenarioFilter output format mismatch.")
        except Exception:
            if prob_type.lower() == "single":
                Y_feature = torch.tensor(Y_full_SNT.sum(axis=1), dtype=torch.float32, device=device).unsqueeze(0)
                Y_work = torch.tensor(Y_full_SNT.sum(axis=1), dtype=torch.float32, device=device).unsqueeze(0)  # [1,S,T]
            else:
                Y_feature = torch.tensor(Y_full_SNT, dtype=torch.float32, device=device).unsqueeze(0).reshape(1, S_full, -1)
                Y_work = torch.tensor(Y_full_SNT, dtype=torch.float32, device=device).unsqueeze(0)  # [1,S,N,T]

            w = nufilter(Y_feature, tau=tau_fixed)
            _, aux_g = select_scenarios_train_gumbel(
                Y_work, w, K=K, K_rand=K_rand,
                tau_gumbel=tau_mix, eps_uniform=eps_uniform
            )
            aux = aux_g if isinstance(aux_g, dict) else {}

            if "idx_rand" in aux and aux["idx_rand"] is not None:
                idx_rand_np = aux["idx_rand"][0].detach().cpu().numpy().astype(int)

            A_t = aux.get("A", None)
            if A_t is None:
                raise RuntimeError("NuFilter path requires aux['A'] but not found.")
            A = A_t[0].detach().cpu().numpy() if A_t.dim() == 3 else A_t.detach().cpu().numpy()
            Y_mix_KNT = np.einsum("ks,snt->knt", A, Y_full_SNT)

    if Y_mix_KNT is None:
        raise RuntimeError("Failed to obtain mixed/selected scenarios for plotting.")

    full_a = Y_pred[:, node_a, t_idx]
    full_b = Y_pred[:, node_b, t_idx]

    if idx_rand_np.size > 0:
        idx_rand_np = idx_rand_np[(idx_rand_np >= 0) & (idx_rand_np < S_full)]
        rand_a = Y_pred[idx_rand_np, node_a, t_idx]
        rand_b = Y_pred[idx_rand_np, node_b, t_idx]
    else:
        rand_a = np.array([])
        rand_b = np.array([])

    mix_a = Y_mix_KNT[:, node_a, hour]
    mix_b = Y_mix_KNT[:, node_b, hour]

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(
        full_a, full_b,
        s=s_full, alpha=alpha_full, c="#1f77b4",
        label=f"Full Pool (S={S_full})",
        edgecolors="none", zorder=1
    )
    # 2) 橙色中层
    ax.scatter(
        mix_a, mix_b,
        s=s_mix, alpha=0.65, c="#ff7f0e",
        label=f"Model Mixed/Selected (K={len(mix_a)})",
        edgecolors="white", lw=0.5, zorder=3
    )
    # 3) 绿色最后画（顶层）——就这一条最关键
    if len(rand_a) > 0:
        ax.scatter(
            rand_a, rand_b,
            s=s_rand + 12, alpha=0.95, c="#2ca02c",
            label=f"Random Raw (K_r={len(rand_a)})",
            edgecolors="white", lw=0.6, zorder=6
        )
        ax.scatter(
            rand_a, rand_b,
            s=s_rand * 0.75, c="#2ca02c", marker="x",
            lw=1.3, alpha=alpha_rand, zorder=7
        )

    if show_truth and ("Y_true" in data):
        y_true = np.asarray(data["Y_true"])
        if y_true.ndim == 2 and y_true.shape[0] > max(node_a, node_b) and y_true.shape[1] > t_idx:
            ax.scatter(
                y_true[node_a, t_idx], y_true[node_b, t_idx],
                c="black", s=150, marker="*", label="Ground Truth",
                zorder=10, edgecolors="white"
            )

    def _plot_ell(x, y, color, label, z=2):
        if len(x) >= 3 and np.std(x) > 1e-12 and np.std(y) > 1e-12:
            pts = _confidence_ellipse_points(x, y, confidence=confidence)
            ax.plot(pts[0], pts[1], color=color, lw=2, label=label, zorder=z)

    _plot_ell(full_a, full_b, "#1f77b4", f"{int(confidence*100)}% Full", z=2)
    if len(rand_a) >= 3:
        _plot_ell(rand_a, rand_b, "#2ca02c", f"{int(confidence*100)}% Random", z=5)
    _plot_ell(mix_a, mix_b, "#ff7f0e", f"{int(confidence*100)}% Mixed", z=4)

    def _safe_corr(x, y):
        if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
            return np.nan
        return pearsonr(x, y)[0]

    c_f = _safe_corr(full_a, full_b)
    c_m = _safe_corr(mix_a, mix_b)

    ax.set_title(
        f"{title_prefix} [{prob_type.upper()}]\n"
        f"Day {day}, Hour {hour} | Node {node_a} vs {node_b}\n"
        f"Corr: Full={c_f:.3f}, Mixed={c_m:.3f}"
    )
    ax.set_xlabel(f"Power at Node {node_a} [p.u.]")
    ax.set_ylabel(f"Power at Node {node_b} [p.u.]")
    ax.legend(loc="best", fontsize="small", frameon=True, ncol=2)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)

    return fig, ax, aux