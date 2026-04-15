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

from utils import *
from diffusion import *
from diff_joint import *
from data_loader import *
from combined_data_loader import *
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
        choices=["joint", "separate", "combined"],
        default="joint",
        help="default: joint",
    )
    parser.add_argument("--seed", type=int, default=0, help="default: 0")
    return parser.parse_args()

class Args:
    def __init__(self):
        self.T = 24
        self.base_mva = 100.0
        self.capacity_scale = 8
        self.ramp_rate = 0.5
        self.voll = 200.0
        self.vosp = 50.0
        self.M_beta = 1e4

        self.reserve_up_ratio = 0.05
        self.reserve_dn_ratio = 0.02
        self.rt_up_ratio = 3.0
        self.rt_dn_ratio = 0.5

        self.device = "cuda"
        self.epochs = 1
        self.train_batch_size = 8
        self.test_batch_size = 8
        self.batch_size = 4
        self.lr = 1e-8
        self.solver = "ECOS"

        self.N_scen = 20
        self.S_full = 100
        self.K_rand = 10
        self.tau_gumbel = 0.1
        self.eps_uniform = 0.1
        self.lambda_div = 1e5

        self.sample_chunk = 10
        self.cpu_offload = False
        self.shared_noise = True

        self.diffusion_mode = "ddpm"  # "ddpm" or "ddim"
        self.n_steps = 50
        self.eta = 0.0

        self.trunc_steps = 1
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

    DTYPE = torch.float64
    data_path = "../Data/load_data_city_4_2.csv"
    target_nodes = [f"4-2-{i}" for i in range(11)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = pd.read_csv(data_path)
    data["DATETIME"] = pd.to_datetime(data["DATETIME"], errors="coerce")
    data_2022 = data[data["DATETIME"].dt.year == 2022].copy()
    Lmin, Lmax = system_hourly_load_minmax(
        data_2022, datetime_col="DATETIME", node_cols=target_nodes
    )
    Lmax_total = Lmax.sum(0)
    Lmin_total = Lmin.sum(0)

    eps_search = pd.read_csv("../Result/eps_search.csv")
    eps = int(eps_search[eps_search["model"] == "diffusion_m"]["eps"])
    args.eps_value = eps

    # ---- load artifacts ----
    with open("../Result/Diffusion/models_m.pkl", "rb") as f:
        models_m = pickle.load(f)
    with open("../Result/Diffusion/handlers_m.pkl", "rb") as f:
        handlers_m = pickle.load(f)
    with open("../Result/Diffusion/pack_data_m.pkl", "rb") as f:
        pack_data_m = pickle.load(f)

    with open("../Result/Diffusion/window_pack_m_diff_val.pkl", "rb") as f:
        window_pack_m_diff_val = pickle.load(f)
    with open("../Result/Diffusion/window_pack_m_diff_test.pkl", "rb") as f:
        window_pack_m_diff_test = pickle.load(f)
    with open("../Result/Diffusion/window_pack_m_diff_train.pkl", "rb") as f:
        window_pack_m_diff_train = pickle.load(f)

    args.eval_mode = "discrete"

    # ---- manager ----
    if optimization_mode == "multi":
        mgr = IEEE14_Reserve_SO_Manager_MultiNode(args)
        args.Lmin = mgr.map_11load_to_14bus(Lmin)
        args.Lmax = mgr.map_11load_to_14bus(Lmax)
    else:
        # 你原脚本只 import 了 multi 的 manager，这里仍保留 single 选项，
        # 但如果你没有 single manager/映射，就只用 total 边界。
        args.Lmin = Lmin_total
        args.Lmax = Lmax_total

    # 你原脚本这里强制 eps_value = 0；保持一致（若想从 eps_search 生效，就删掉这一段）
    args.eps_value = 0

    args.run_stage2 = True
    args.run_stage3 = True

    # lr decay
    args.lr_decay = 1
    args.filter_lr_decay = 1
    args.dfl_lr_decay = 1

    # 超参
    args.dfl_epochs = 3
    args.dfl_lr = 1e-6
    args.filter_lr = 1e-3
    args.N_scen = 20
    args.K_rand = 10
    args.filter_epochs = 3
    args.clip_predictor = 1.0
    args.S_full = 200

    args.eps_uniform = 0.1
    args.tau_gumbel = 0.1

    args.train_batch_size = 8
    args.test_batch_size = 8

    results = {}
    compares = {}

    # =========================================================
    # 1) KL
    # =========================================================
    args.div_type = "kl"
    args.lambda_div = 1e5
    args.lambda_div_stage3 = 1e5
    key = f"{optimization_mode}_{problem_mode}_{args.div_type}"

    results[key] = run_DFL_diffusion_joint(
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
        eval_flags=(True, True, True, True, True),
        stage2_artifact=None,
    )

    save_run_result(results[key], forecasting_mode, out_dir="./Result/Diffusion/KL")

    compares[key] = compare_scenario_filters_with_stage3_learned_diff(
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
    # 2) Entropy
    # =========================================================
    args.div_type = "entropy"
    key = f"{optimization_mode}_{problem_mode}_{args.div_type}"

    results[key] = run_DFL_diffusion_joint(
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

    save_run_result(results[key], forecasting_mode, out_dir="./Result/Diffusion/Entropy")

    compares[key] = compare_scenario_filters_with_stage3_learned_diff(
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
    # 3) Inner
    # =========================================================
    args.div_type = "inner"
    args.lambda_div = 5e8
    args.lambda_div_stage3 = 5e8
    key = f"{optimization_mode}_{problem_mode}_{args.div_type}"

    results[key] = run_DFL_diffusion_joint(
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

    save_run_result(results[key], forecasting_mode, out_dir="./Result/Diffusion/Inner")

    compares[key] = compare_scenario_filters_with_stage3_learned_diff(
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
    # 4) merge + export pkl
    # =========================================================
    compare_res_all = merge_compare_results_learned_as_variants(
        compares[f"{optimization_mode}_{problem_mode}_kl"],
        compares[f"{optimization_mode}_{problem_mode}_inner"],
        compares[f"{optimization_mode}_{problem_mode}_entropy"],
    )

    out_pkl = f"./Result/Diffusion/compare_res_{optimization_mode}_{problem_mode}_all_joint.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump(compare_res_all, f)

    print("\n[Comparison Summary Rows]")
    summary_rows = summarize_compare_result(compares[f"{optimization_mode}_{problem_mode}_kl"])
    for row in summary_rows:
        print(row)

    print(f"\n[Saved] {out_pkl}")

if __name__ == "__main__":
    #python main_diffusion_joint.py
    #python main_diffusion_joint.py --optimization-mode multi --problem-mode dro
    main()