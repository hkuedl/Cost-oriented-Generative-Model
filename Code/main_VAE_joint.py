import os
import math
import copy
import pickle
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from data_loader import *
from utils import *
from VAE import *
from VAE_joint import *
from combined_data_loader import *
from Optimization_multi_node import *
from Optimization_single_node import *

# ----------------------------
# CLI
# ----------------------------
def parse_cli():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--optimization-mode",
        choices=["multi", "single"],
        default="multi",
        help="default: multi",
    )
    parser.add_argument(
        "--problem-mode",
        choices=["dro", "so", "ro"],
        default="dro",
        help="default: dro",
    )
    parser.add_argument(
        "--forecasting-mode",
        choices=["joint", "separate", "combined"],
        default="joint",
        help="default: joint",
    )
    parser.add_argument("--seed", type=int, default=0, help="default: 0")

    return parser.parse_args()

class Args:
    def __init__(self):
        # ----- problem / optnet -----
        self.T = 24
        self.base_mva = 100.0
        self.capacity_scale = 4.5
        self.ramp_rate = 0.5
        self.voll = 200.0
        self.vosp = 50.0
        self.M_beta = 1e4
        self.pwl_segments = 10

        # IMPORTANT: add these to match gurobi
        self.reserve_up_ratio = 0.05
        self.reserve_dn_ratio = 0.02
        self.rt_up_ratio = 3.0
        self.rt_dn_ratio = 0.5

        self.N_scen = 20  # OptNet真正求解的场景池 (K)
        self.S_full = 100  # VAE吐出候选场景数 (S池)
        self.K_rand = 0
        self.tau_gumbel = 1.0
        self.eps_uniform = 0.1
        self.lambda_div = 1e5

        # ----- 分段训练控制参数 -----
        self.device = "cuda"
        self.train_batch_size = 8
        self.test_batch_size = 8
        self.solver = "ECOS"

        self.filter_epochs = 5
        self.filter_lr = 1e-3
        self.dfl_epochs = 1
        self.dfl_lr = 1e-6

def main():
    cli = parse_cli()

    args = Args()
    set_seed(42)

    optimization_mode = cli.optimization_mode
    problem_mode = cli.problem_mode
    forecasting_mode = cli.forecasting_mode
    seed = cli.seed

    print(
        f"[CLI] optimization_mode={optimization_mode}, problem_mode={problem_mode}, "
        f"forecasting_mode={forecasting_mode}, seed={seed}"
    )

    data_path = "../Data/load_data_city_4_2.csv"
    target_nodes = [f"4-2-{i}" for i in range(11)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    eps_search = pd.read_csv("../Result/eps_search.csv")
    eps = int(eps_search[eps_search["model"] == "vae_m"]["eps"])

    data = pd.read_csv(data_path)
    data["DATETIME"] = pd.to_datetime(data["DATETIME"], errors="coerce")
    data_2022 = data[data["DATETIME"].dt.year == 2022].copy()
    Lmin, Lmax = system_hourly_load_minmax(
        data_2022, datetime_col="DATETIME", node_cols=target_nodes
    )
    Lmax_total = Lmax.sum(0)  # (24,)
    Lmin_total = Lmin.sum(0)  # (24,)
    args.eps_value = eps

    # ---- load models / data packs ----
    with open("../Result/VAE/models_m.pkl", "rb") as f:
        models_m = pickle.load(f)
    with open("../Result/VAE/handlers_m.pkl", "rb") as f:
        handlers_m = pickle.load(f)
    with open("../Result/VAE/pack_data_m.pkl", "rb") as f:
        pack_data_m = pickle.load(f)

    # 这些 window_pack 在你贴的脚本里没被使用；这里仅按“语义正确”读入，防你后续要用
    try:
        with open("../Result/VAE/window_pack_m_vae_test.pkl", "rb") as f:
            window_pack_m_vae_test = pickle.load(f)
    except FileNotFoundError:
        window_pack_m_vae_test = None

    try:
        with open("../Result/VAE/window_pack_m_vae_train.pkl", "rb") as f:
            window_pack_m_vae_train = pickle.load(f)
    except FileNotFoundError:
        window_pack_m_vae_train = None

    try:
        with open("../Result/VAE/window_pack_m_vae_val.pkl", "rb") as f:
            window_pack_m_vae_val = pickle.load(f)
    except FileNotFoundError:
        window_pack_m_vae_val = None

    # ---- build manager ----
    if optimization_mode == "multi":
        mgr = IEEE14_Reserve_SO_Manager_MultiNode(args)
    else:
        mgr = IEEE14_Reserve_SO_Manager_SingleNode(args)

    # 写入 args 边界（DRO 会用到）
    if optimization_mode == "multi":
        args.Lmin = mgr.map_11load_to_14bus(Lmin)
        args.Lmax = mgr.map_11load_to_14bus(Lmax)
    else:
        args.Lmin = Lmin_total
        args.Lmax = Lmax_total

    args.run_stage2 = True
    args.run_stage3 = True

    # 超参（按你给的 joint 配置）
    args.eval_mode = "discrete"
    args.dfl_epochs = 3
    args.dfl_lr = 1e-6
    args.filter_lr = 1e-3
    args.N_scen = 20
    args.K_rand = 10
    args.filter_epochs = 3
    args.clip_predictor = 1.0
    args.S_full = 200

    args.lambda_div = 1e5
    args.lambda_div_stage3 = 1e5
    args.eps_uniform = 0.1
    args.tau_gumbel = 0.1

    # 用字典存结果，避免 single_dro_xxx 这种硬编码变量名
    results = {}
    compares = {}

    # =========================================================
    # 1) KL：完整跑 + compare，并导出 stage2_artifact
    # =========================================================
    args.div_type = "kl"
    key = f"{optimization_mode}_{problem_mode}_{args.div_type}"

    results[key] = run_DFL_vae_joint(
        args=args,
        optimization_mode=optimization_mode,
        problem_mode=problem_mode,
        data_path=data_path,
        target_nodes=target_nodes,
        pack_data_m=pack_data_m,
        models_m=models_m,
        device=device,
        seed=seed,
        eval_splits=("test",),
        eval_flags=(False, True, True, True, True),
        stage2_artifact=None,
    )

    save_run_result(results[key], forecasting_mode, out_dir="../Result copy/VAE/KL")

    compares[key] = compare_scenario_filters_with_stage3_learned_vae(
        base_result=results[key],
        args=args,
        problem_mode=problem_mode,
        optimization_mode=optimization_mode,
        models_m=models_m,
        target_nodes=target_nodes,
        device=device,
        eval_splits=("test",),
        method_names=["learned", "random", "kmeans", "kmedoids", "hierarchical"],
        seed=seed,
        verbose=True,
    )

    shared_stage2_artifact = results[key]["stage2_artifact"]

    # =========================================================
    # 2) Entropy：复用 KL 的 stage2_artifact（通常只 compare learned）
    # =========================================================
    args.div_type = "entropy"
    key = f"{optimization_mode}_{problem_mode}_{args.div_type}"

    results[key] = run_DFL_vae_joint(
        args=args,
        optimization_mode=optimization_mode,
        problem_mode=problem_mode,
        data_path=data_path,
        target_nodes=target_nodes,
        pack_data_m=pack_data_m,
        models_m=models_m,
        device=device,
        seed=seed,
        eval_splits=("test",),
        eval_flags=(False, False, False, False, False),
        stage2_artifact=shared_stage2_artifact,
    )

    save_run_result(results[key], forecasting_mode, out_dir="../Result copy/VAE/Entropy")

    compares[key] = compare_scenario_filters_with_stage3_learned_vae(
        base_result=results[key],
        args=args,
        problem_mode=problem_mode,
        optimization_mode=optimization_mode,
        models_m=models_m,
        target_nodes=target_nodes,
        device=device,
        eval_splits=("test",),
        method_names=["learned"],
        seed=seed,
        verbose=True,
    )

    # =========================================================
    # 3) Inner：复用 KL 的 stage2_artifact（inner 常需要更大 penalty）
    # =========================================================
    args.lambda_div = 5e8
    args.lambda_div_stage3 = 5e8
    args.div_type = "inner"
    key = f"{optimization_mode}_{problem_mode}_{args.div_type}"

    results[key] = run_DFL_vae_joint(
        args=args,
        optimization_mode=optimization_mode,
        problem_mode=problem_mode,
        data_path=data_path,
        target_nodes=target_nodes,
        pack_data_m=pack_data_m,
        models_m=models_m,
        device=device,
        seed=seed,
        eval_splits=("test",),
        eval_flags=(False, False, False, False, False),
        stage2_artifact=shared_stage2_artifact,
    )

    save_run_result(results[key], forecasting_mode, out_dir="../Result copy/VAE/Inner")

    compares[key] = compare_scenario_filters_with_stage3_learned_vae(
        base_result=results[key],
        args=args,
        problem_mode=problem_mode,
        optimization_mode=optimization_mode,
        models_m=models_m,
        target_nodes=target_nodes,
        device=device,
        eval_splits=("test",),
        method_names=["learned"],
        seed=seed,
        verbose=True,
    )

    # =========================================================
    # 4) 合并 learned variants + 导出 pkl
    # =========================================================
    compare_res_all = merge_compare_results_learned_as_variants(
        compares[f"{optimization_mode}_{problem_mode}_kl"],
        compares[f"{optimization_mode}_{problem_mode}_inner"],
        compares[f"{optimization_mode}_{problem_mode}_entropy"],
    )

    out_pkl = f"../Result copy/VAE/compare_res_{optimization_mode}_{problem_mode}_all_joint.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump(compare_res_all, f)

    print("\n[Comparison Summary Rows]")
    summary_rows = summarize_compare_result(compares[f"{optimization_mode}_{problem_mode}_kl"])
    for row in summary_rows:
        print(row)

    print(f"\n[Saved] {out_pkl}")

if __name__ == "__main__":
    #python main_VAE_joint_python.py
    #python main_VAE_joint_python.py --optimization-mode single --problem-mode dro
    main()