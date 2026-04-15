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
from tqdm import tqdm

from data_loader import *
from utils import *
from utils_draw import *
from GAN import *
from GAN_separate import *
from combined_data_loader import *
from Optimization_single_node import *
from Optimization_multi_node import *

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
        choices=["separate", "joint", "combined"],
        default="separate",
        help="default: separate",
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

        self.N_scen = 10
        self.S_full = 100
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

    eps_search = pd.read_csv("../Result/eps_search.csv")
    eps = int(eps_search[eps_search["model"] == "gan_s"]["eps"])

    target_nodes = [f"4-2-{i}" for i in range(11)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = pd.read_csv(data_path)
    data["DATETIME"] = pd.to_datetime(data["DATETIME"], errors="coerce")
    data_2022 = data[data["DATETIME"].dt.year == 2022].copy()

    Lmin, Lmax = system_hourly_load_minmax(
        data_2022, datetime_col="DATETIME", node_cols=target_nodes
    )
    Lmax_total = Lmax.sum(0)  # (24,)
    Lmin_total = Lmin.sum(0)  # (24,)
    args.Lmax_total = Lmax_total
    args.Lmin_total = Lmin_total
    args.eps_value = eps

    # ---- load artifacts ----
    with open("../Result/GAN/models_s.pkl", "rb") as f:
        models_s = pickle.load(f)
    with open("../Result/GAN/handlers_s.pkl", "rb") as f:
        handlers_s = pickle.load(f)
    with open("../Result/GAN/pack_data_s.pkl", "rb") as f:
        pack_data_s = pickle.load(f)

    with open("../Result/GAN/window_pack_s_gan_test.pkl", "rb") as f:
        window_pack_s_gan_test = pickle.load(f)
    with open("../Result/GAN/window_pack_s_gan_train.pkl", "rb") as f:
        window_pack_s_gan_train = pickle.load(f)
    with open("../Result/GAN/window_pack_s_gan_val.pkl", "rb") as f:
        window_pack_s_gan_val = pickle.load(f)

    # ---- build manager ----
    if optimization_mode == "multi":
        mgr = IEEE14_Reserve_SO_Manager_MultiNode(args)
    else:
        mgr = IEEE14_Reserve_SO_Manager_SingleNode(args)

    # 写入 args 配置边界（DRO 会用到）
    if optimization_mode == "multi":
        args.Lmin = mgr.map_11load_to_14bus(Lmin)
        args.Lmax = mgr.map_11load_to_14bus(Lmax)
    else:
        args.Lmin = Lmin_total
        args.Lmax = Lmax_total

    args.run_stage2 = True
    args.run_stage3 = True

    # 超参（按你给的 GAN 配置）
    args.eval_mode = "discrete"
    args.dfl_epochs = 3
    args.dfl_lr = 1e-6
    args.filter_lr = 1e-3
    args.N_scen = 20
    args.K_rand = 10
    args.filter_epochs = 3
    args.clip_predictor = 1.0
    args.S_full = 200
    args.tau_gumbel = 0.1
    args.eps_uniform = 0.1
    results = {}
    compares = {}

    # =========
    # 1) KL：完整跑 + compare，拿 stage2_artifact
    # =========
    args.lambda_div = 1e5
    args.lambda_div_stage3 = 1e5
    args.div_type = "kl"
    key = f"{optimization_mode}_{problem_mode}_{args.div_type}"

    results[key] = run_DFL_gan_separate(
        args=args,
        optimization_mode=optimization_mode,
        problem_mode=problem_mode,
        data_path=data_path,
        target_nodes=target_nodes,
        pack_data_s=pack_data_s,
        models_s=models_s,
        device=device,
        seed=seed,
        eval_splits=("test",),
        eval_flags=(True, True, True, True, True),
        stage2_artifact=None,
    )

    save_run_result(results[key], forecasting_mode, out_dir="../Result/GAN/KL")

    compares[key] = compare_scenario_filters_with_stage3_learned_gan(
        base_result=results[key],
        args=args,
        problem_mode=problem_mode,
        optimization_mode=optimization_mode,
        models_s=models_s,
        target_nodes=target_nodes,
        device=device,
        eval_splits=("test",),
        method_names=["learned", "random", "kmeans", "kmedoids", "hierarchical"],
        seed=seed,
        verbose=True,
    )

    shared_stage2_artifact = results[key]["stage2_artifact"]

    # =========
    # 2) Entropy：复用 stage2_artifact
    # =========
    args.div_type = "entropy"
    key = f"{optimization_mode}_{problem_mode}_{args.div_type}"

    results[key] = run_DFL_gan_separate(
        args=args,
        optimization_mode=optimization_mode,
        problem_mode=problem_mode,
        data_path=data_path,
        target_nodes=target_nodes,
        pack_data_s=pack_data_s,
        models_s=models_s,
        device=device,
        seed=seed,
        eval_splits=("test",),
        eval_flags=(False, False, False, False, False),
        stage2_artifact=shared_stage2_artifact,
    )

    save_run_result(results[key], forecasting_mode, out_dir="../Result/GAN/Entropy")

    compares[key] = compare_scenario_filters_with_stage3_learned_gan(
        base_result=results[key],
        args=args,
        problem_mode=problem_mode,
        optimization_mode=optimization_mode,
        models_s=models_s,
        target_nodes=target_nodes,
        device=device,
        eval_splits=("test",),
        method_names=["learned"],
        seed=seed,
        verbose=True,
    )

    # =========
    # 3) Inner：复用 stage2_artifact（更大 penalty）
    # =========
    args.lambda_div = 5e8
    args.lambda_div_stage3 = 5e8
    args.div_type = "inner"
    key = f"{optimization_mode}_{problem_mode}_{args.div_type}"

    results[key] = run_DFL_gan_separate(
        args=args,
        optimization_mode=optimization_mode,
        problem_mode=problem_mode,
        data_path=data_path,
        target_nodes=target_nodes,
        pack_data_s=pack_data_s,
        models_s=models_s,
        device=device,
        seed=seed,
        eval_splits=("test",),
        eval_flags=(False, False, False, False, False),
        stage2_artifact=shared_stage2_artifact,
    )

    save_run_result(results[key], forecasting_mode, out_dir="../Result/GAN/Inner")

    compares[key] = compare_scenario_filters_with_stage3_learned_gan(
        base_result=results[key],
        args=args,
        problem_mode=problem_mode,
        optimization_mode=optimization_mode,
        models_s=models_s,
        target_nodes=target_nodes,
        device=device,
        eval_splits=("test",),
        method_names=["learned"],
        seed=seed,
        verbose=True,
    )

    # =========
    # 4) merge + export
    # =========
    compare_res_all = merge_compare_results_learned_as_variants(
        compares[f"{optimization_mode}_{problem_mode}_kl"],
        compares[f"{optimization_mode}_{problem_mode}_inner"],
        compares[f"{optimization_mode}_{problem_mode}_entropy"],
    )

    out_pkl = f"../Result/GAN/compare_res_{optimization_mode}_{problem_mode}_all_separate.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump(compare_res_all, f)

    print("\n[Comparison Summary Rows]")
    summary_rows = summarize_compare_result(compares[f"{optimization_mode}_{problem_mode}_kl"])
    for row in summary_rows:
        print(row)

    print(f"\n[Saved] {out_pkl}")

if __name__ == "__main__":
    #python main_GAN_separate.py
    #python main_GAN_separate.py --optimization-mode single --problem-mode dro
    #python main_GAN_separate.py --optimization-mode multi --problem-mode dro --seed 0
    main()