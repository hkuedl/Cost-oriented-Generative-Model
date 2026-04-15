import argparse
import numpy as np
import pandas as pd
import gurobipy as gp

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


def get_ieee14_args():
    parser = argparse.ArgumentParser(description="IEEE 14 System Args")

    # --- 物理系统配置 ---
    parser.add_argument('--case_name', type=str, default='IEEE14', help='选择电网算例')
    parser.add_argument('--base_mva', type=float, default=100.0, help='基准容量，用于计算B矩阵')
    parser.add_argument('--capacity_scale', type=float, default=2.0, help='发电机容量放大倍数')
    parser.add_argument('--ramp_rate', type=float, default=0.8, help='爬坡率系数')

    # --- 经济参数 (修正后) ---
    parser.add_argument('--act_price', type=float, default=60.0, help='实时备用激活价 (> MC)')
    parser.add_argument('--cap_price', type=float, default=8.0, help='备用容量预留价')
    parser.add_argument('--voll', type=float, default=5000.0, help='切负荷惩罚')
    parser.add_argument('--vosp', type=float, default=10.0, help='弃电惩罚')

    # --- 维度 ---
    parser.add_argument('--T', type=int, default=24)
    parser.add_argument('--enforce_network', type=bool, default=True)

    args = parser.parse_args(args=[]) # Jupyter中用 args=[]
    return args
