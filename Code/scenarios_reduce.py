import os, copy
from dataclasses import dataclass
from typing import Optional, List, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from joblib import Parallel, delayed

from cvxpylayers.torch import CvxpyLayer
import cvxpy as cp
from utils import *
from Optimization_multi_node import *
from Optimization_single_node import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import torch
from joblib import Parallel, delayed


import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import numpy as np

from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.spatial.distance import cdist
from sklearn_extra.cluster import KMedoids

class _BaseClusterScenarioFilter(nn.Module):
    def __init__(self, K, feature_mode="sum"):
        super().__init__()
        self.K = int(K)
        self.feature_mode = feature_mode

    def _build_feature(self, Y_b):
        """
        Y_b: [S, N, T]
        return: [S, d]
        """
        if self.feature_mode == "sum":
            feat = Y_b.sum(dim=1)          # [S, T]
        elif self.feature_mode == "mean":
            feat = Y_b.mean(dim=1)         # [S, T]
        elif self.feature_mode == "flat":
            feat = Y_b.reshape(Y_b.shape[0], -1)  # [S, N*T]
        else:
            raise ValueError(f"Unknown feature_mode={self.feature_mode}")
        return feat

    def _make_aux_and_output(self, Y_full, idx_all):
        """
        Y_full: [B, S, N, T]
        idx_all: [B, K]
        return:
            Y_sel: [K, B, N, T]
            aux: dict
        """
        B, S, N, T = Y_full.shape
        K = idx_all.shape[1]
        device = Y_full.device
        dtype = Y_full.dtype

        Y_sel = torch.gather(
            Y_full, 1,
            idx_all[:, :, None, None].expand(B, K, N, T)
        )  # [B, K, N, T]

        A = torch.zeros((B, K, S), device=device, dtype=dtype)
        A.scatter_(dim=-1, index=idx_all.unsqueeze(-1), value=1.0)

        aux = {
            "idx": idx_all,   # [B, K]
            "A": A,           # [B, K, S]
            "p": A,           # 保持接口兼容
        }

        Y_sel = Y_sel.permute(1, 0, 2, 3).contiguous()  # [K, B, N, T]
        return Y_sel, aux



class KMeansScenarioFilter(_BaseClusterScenarioFilter):
    def __init__(self, K, feature_mode="sum", random_state=0, n_init=10):
        super().__init__(K=K, feature_mode=feature_mode)
        self.random_state = random_state
        self.n_init = n_init

    def _select_indices_one_batch(self, feat_np, K):
        """
        feat_np: [S, d]
        return idx: [K]
        """
        S = feat_np.shape[0]
        kmeans = KMeans(
            n_clusters=K,
            random_state=self.random_state,
            n_init=self.n_init
        )
        labels = kmeans.fit_predict(feat_np)         # [S]
        centers = kmeans.cluster_centers_            # [K, d]

        # 选择每个 centroid 最近的真实样本，并避免重复
        dist = cdist(centers, feat_np, metric="euclidean")  # [K, S]

        idx = []
        used = set()
        for k in range(K):
            order = np.argsort(dist[k])
            picked = None
            for s in order:
                s = int(s)
                if s not in used:
                    idx.append(s)
                    used.add(s)
                    picked = s
                    break

            if picked is None:
                for s in range(S):
                    if s not in used:
                        idx.append(s)
                        used.add(s)
                        break

        return idx

    def forward(self, Y_scen, **kwargs):
        """
        Y_scen: [S, B, N, T]
        return:
            Y_sel: [K, B, N, T]
            aux
        """
        S, B, N, T = Y_scen.shape
        K = min(self.K, S)
        device = Y_scen.device

        Y_full = Y_scen.permute(1, 0, 2, 3).contiguous()  # [B, S, N, T]

        idx_all = []
        for b in range(B):
            Y_b = Y_full[b]                                # [S, N, T]
            feat = self._build_feature(Y_b)                # [S, d]
            feat_np = feat.detach().cpu().numpy()

            idx = self._select_indices_one_batch(feat_np, K)
            idx = torch.tensor(idx, device=device, dtype=torch.long)
            idx_all.append(idx)

        idx_all = torch.stack(idx_all, dim=0)              # [B, K]
        return self._make_aux_and_output(Y_full, idx_all)

class KMedoidsScenarioFilter(_BaseClusterScenarioFilter):
    def __init__(self, K, feature_mode="sum", metric="euclidean", random_state=0):
        super().__init__(K=K, feature_mode=feature_mode)
        self.metric = metric
        self.random_state = random_state

    def _select_indices_one_batch(self, feat_np, K):
        """
        feat_np: [S, d]
        return idx: [K]
        """
        model = KMedoids(
            n_clusters=K,
            metric=self.metric,
            random_state=self.random_state
        )
        model.fit(feat_np)
        idx = model.medoid_indices_.tolist()   # [K]
        return idx

    def forward(self, Y_scen, **kwargs):
        """
        Y_scen: [S, B, N, T]
        """
        S, B, N, T = Y_scen.shape
        K = min(self.K, S)
        device = Y_scen.device

        Y_full = Y_scen.permute(1, 0, 2, 3).contiguous()  # [B, S, N, T]

        idx_all = []
        for b in range(B):
            Y_b = Y_full[b]
            feat = self._build_feature(Y_b)
            feat_np = feat.detach().cpu().numpy()

            idx = self._select_indices_one_batch(feat_np, K)
            idx = torch.tensor(idx, device=device, dtype=torch.long)
            idx_all.append(idx)

        idx_all = torch.stack(idx_all, dim=0)             # [B, K]
        return self._make_aux_and_output(Y_full, idx_all)


class HierarchicalScenarioFilter(_BaseClusterScenarioFilter):
    def __init__(self, K, feature_mode="sum", linkage="ward"):
        super().__init__(K=K, feature_mode=feature_mode)
        self.linkage = linkage

    def _select_indices_one_batch(self, feat_np, K):
        """
        feat_np: [S, d]
        return idx: [K]
        """
        S = feat_np.shape[0]

        hc = AgglomerativeClustering(
            n_clusters=K,
            linkage=self.linkage
        )
        labels = hc.fit_predict(feat_np)  # [S]

        idx = []
        used = set()

        for k in range(K):
            members = np.where(labels == k)[0]

            # 理论上不会空，但还是做保护
            if len(members) == 0:
                for s in range(S):
                    if s not in used:
                        idx.append(s)
                        used.add(s)
                        break
                continue

            cluster_feat = feat_np[members]                        # [m, d]
            center = cluster_feat.mean(axis=0, keepdims=True)      # [1, d]
            dist = cdist(center, cluster_feat, metric="euclidean")[0]  # [m]
            order = np.argsort(dist)

            picked = None
            for j in order:
                s = int(members[j])
                if s not in used:
                    idx.append(s)
                    used.add(s)
                    picked = s
                    break

            if picked is None:
                for s in members:
                    s = int(s)
                    if s not in used:
                        idx.append(s)
                        used.add(s)
                        picked = s
                        break

            if picked is None:
                for s in range(S):
                    if s not in used:
                        idx.append(s)
                        used.add(s)
                        break

        return idx

    def forward(self, Y_scen, **kwargs):
        """
        Y_scen: [S, B, N, T]
        """
        S, B, N, T = Y_scen.shape
        K = min(self.K, S)
        device = Y_scen.device

        Y_full = Y_scen.permute(1, 0, 2, 3).contiguous()  # [B, S, N, T]

        idx_all = []
        for b in range(B):
            Y_b = Y_full[b]
            feat = self._build_feature(Y_b)
            feat_np = feat.detach().cpu().numpy()

            idx = self._select_indices_one_batch(feat_np, K)
            idx = torch.tensor(idx, device=device, dtype=torch.long)
            idx_all.append(idx)

        idx_all = torch.stack(idx_all, dim=0)             # [B, K]
        return self._make_aux_and_output(Y_full, idx_all)

class RandomScenarioSelector(torch.nn.Module):
    def __init__(self, n_scen: int):
        super().__init__()
        self.n_scen = int(n_scen)

    def forward(self, Y_scen, is_train=False, **kwargs):
        # Y_scen: [S_full, B, ...]
        S_full, B = Y_scen.shape[0], Y_scen.shape[1]
        K = min(self.n_scen, S_full)

        # choose K unique indices (same for the whole batch), simplest baseline
        perm = torch.randperm(S_full, device=Y_scen.device)
        idx = perm[:K]                      # [K]
        Y_sel = Y_scen.index_select(0, idx) # [K, B, ...]

        # Build aux to be compatible with your DFL_train diversity regularization
        # p: [B, K, S_full] (one-hot selection)
        p = Y_scen.new_zeros((B, K, S_full))
        p[:, torch.arange(K, device=Y_scen.device), idx] = 1.0

        aux = {"idx": idx,   # [K]
            "p": p,       # [B, K, S_full]
            "A": p,       # optional: some code checks "A" in aux_filter
        }
        return Y_sel, aux


class ScenarioFilter(nn.Module):
    def __init__(self, args, prob_type="multi", T=24, N_nodes=11, K=20, K_rand=10, hidden=128):
        super().__init__()
        self.prob_type = prob_type
        self.K = int(K)
        self.K_rand = int(K_rand)
        self.K_model = self.K - self.K_rand

        print("K_rand", self.K_rand)
        print("K_model", self.K_model)

        # 实际输入是 sum(dim=2) 后的 [B,S,T]
        input_dim = T
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.K_model),
        )

        self.eps_uniform = float(getattr(args, "eps_uniform", 0.10))
        self.tau_gumbel = float(getattr(args, "tau_gumbel", 1.0))

        # 新增：由 args 控制 eval 行为
        self.eval_mode = str(getattr(args, "eval_mode", "soft")).lower()
        self.avoid_rand_duplicate = bool(getattr(args, "avoid_rand_duplicate", False))

        print("eps_uniform", self.eps_uniform)
        print("tau_gumbel", self.tau_gumbel)
        print("eval_mode", self.eval_mode)
        print("avoid_rand_duplicate", self.avoid_rand_duplicate)

    def _discrete_select_indices(self, p, idx_rand=None):
        """
        p: [B, K_model, S]
        idx_rand: [B, K_rand] or None

        return:
            idx_model: [B, K_model]
        """
        B, K_model, S = p.shape
        device = p.device

        idx_model_list = []

        for b in range(B):
            chosen = []
            used = set()

            # 是否避免和 random 部分重复，由 args 控制
            if self.avoid_rand_duplicate and idx_rand is not None:
                used.update(idx_rand[b].tolist())

            for k in range(K_model):
                order = torch.argsort(p[b, k], descending=True)
                picked = None

                for s in order.tolist():
                    if s not in used:
                        picked = s
                        chosen.append(s)
                        used.add(s)
                        break

                # 理论补位：如果全撞了，就按平均分补
                if picked is None:
                    avg_score = p[b].mean(dim=0)  # [S]
                    order_global = torch.argsort(avg_score, descending=True)
                    for s in order_global.tolist():
                        if s not in used:
                            chosen.append(s)
                            used.add(s)
                            picked = s
                            break

                # 极端保险
                if picked is None:
                    for s in range(S):
                        if s not in used:
                            chosen.append(s)
                            used.add(s)
                            picked = s
                            break

            idx_model_list.append(torch.tensor(chosen, device=device, dtype=torch.long))

        idx_model = torch.stack(idx_model_list, dim=0)  # [B,K_model]
        return idx_model

    def forward(self, Y_scen, tau_gumbel=1.0, eps_uniform=0.10, is_train=True):
        """
        Y_scen: [S, B, N, T]

        return:
            Y_sel: [K, B, N, T]
            aux:
                p: [B, K_model, S]
                A: [B, K_model, S]
                idx_rand: [B, K_rand] or None
                idx_model: [B, K_model] or None
        """
        eps_uniform = self.eps_uniform
        tau_gumbel = self.tau_gumbel

        Y_full = Y_scen.permute(1, 0, 2, 3).contiguous()  # [B,S,N,T]
        B, S, N, T_len = Y_full.shape
        device, dtype = Y_full.device, Y_full.dtype

        # 先聚合节点维度，得到 [B,S,T]
        Y_feature = Y_full.sum(dim=2)  # [B,S,T]

        # logits: [B,S,K_model] -> [B,K_model,S]
        logits = self.net(Y_feature.float()).transpose(1, 2)

        # 基础概率 p
        p = torch.softmax(logits / max(float(tau_gumbel), 1e-6), dim=-1).to(dtype)

        # uniform smoothing
        if eps_uniform > 0:
            p = (1 - eps_uniform) * p + eps_uniform * (1.0 / S)
            p = p / p.sum(dim=-1, keepdim=True)

        # -----------------------------
        # random keep
        # -----------------------------
        if self.K_rand > 0:
            idx_rand = torch.stack(
                [torch.randperm(S, device=device)[:self.K_rand] for _ in range(B)],
                dim=0
            )  # [B,K_rand]

            Y_rand = torch.gather(
                Y_full, 1,
                idx_rand[:, :, None, None].expand(B, self.K_rand, N, T_len)
            )  # [B,K_rand,N,T]
        else:
            idx_rand = None
            Y_rand = Y_full[:, :0, ...]

        # -----------------------------
        # model part
        # -----------------------------
        idx_model = None

        if self.K_model > 0:
            if is_train:
                log_p = torch.log(torch.clamp(p, 1e-12, 1.0))
                A = F.gumbel_softmax(
                    log_p,
                    tau=max(float(tau_gumbel), 1e-6),
                    hard=False,
                    dim=-1
                )  # [B,K_model,S]

                Y_mix = torch.einsum("bks,bsnt->bknt", A, Y_full)

            else:
                if self.eval_mode == "soft":
                    A = p
                    Y_mix = torch.einsum("bks,bsnt->bknt", A, Y_full)

                elif self.eval_mode == "discrete":
                    idx_model = self._discrete_select_indices(p, idx_rand=idx_rand)  # [B,K_model]

                    Y_mix = torch.gather(
                        Y_full, 1,
                        idx_model[:, :, None, None].expand(B, self.K_model, N, T_len)
                    )  # [B,K_model,N,T]

                    # 为了接口一致性，这里仍返回 A，但只是 one-hot indicator
                    A = torch.zeros((B, self.K_model, S), device=device, dtype=dtype)
                    A.scatter_(dim=-1, index=idx_model.unsqueeze(-1), value=1.0)

                else:
                    raise ValueError(f"Unknown eval_mode={self.eval_mode}, expected 'soft' or 'discrete'")

            Y_sel = torch.cat([Y_rand, Y_mix], dim=1)  # [B,K,N,T]

        else:
            A = torch.empty((B, 0, S), device=device, dtype=dtype)
            Y_sel = Y_rand

        aux = {
            "p": p,
            "A": A,
            "idx_rand": idx_rand,
            "idx_model": idx_model,
        }

        Y_sel = Y_sel.permute(1, 0, 2, 3).contiguous()  # [K,B,N,T]
        return Y_sel, aux

# class ScenarioFilter(nn.Module):
#     def __init__(self, prob_type="multi", T=24, N_nodes=11, K=20, K_rand=10, hidden=128):
#         super().__init__()
#         self.prob_type = prob_type
#         self.K = int(K)
#         self.K_rand = int(K_rand)
#         self.K_model = self.K - self.K_rand

#         # 以你给的基准逻辑为准：实际输入是 sum(dim=2) 后的 [B,S,T]
#         input_dim = T
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, hidden),
#             nn.ReLU(),
#             nn.Linear(hidden, self.K_model),
#         )

#     def forward(self, Y_scen, tau_gumbel=1.0, eps_uniform=0.10, is_train=True):
#         """
#         Y_scen: [S, B, N, T]
#         return:
#             Y_sel: [K, B, N, T]
#             aux:
#                 p: [B, K_model, S]
#                 A: [B, K_model, S]  (train时为gumbel soft sample, eval时为p)
#                 idx_rand: [B, K_rand] or None
#         """
#         Y_full = Y_scen.permute(1, 0, 2, 3).contiguous()   # [B,S,N,T]
#         B, S, N, T_len = Y_full.shape
#         device, dtype = Y_full.device, Y_full.dtype

#         # 与基准代码对齐：先聚合节点维度，得到 [B,S,T]
#         Y_feature = Y_full.sum(dim=2)                      # [B,S,T]

#         # logits: [B,S,K_model] -> [B,K_model,S]
#         logits = self.net(Y_feature.float()).transpose(1, 2)

#         # 基础概率 w/p
#         p = torch.softmax(logits / max(float(tau_gumbel), 1e-6), dim=-1).to(dtype)

#         # uniform smoothing
#         if eps_uniform > 0:
#             p = (1 - eps_uniform) * p + eps_uniform * (1.0 / S)
#             p = p / p.sum(dim=-1, keepdim=True)

#         # 随机保底场景
#         if self.K_rand > 0:
#             idx = torch.stack(
#                 [torch.randperm(S, device=device)[:self.K_rand] for _ in range(B)],
#                 dim=0
#             )  # [B,K_rand]
#             Y_rand = torch.gather(
#                 Y_full, 1,
#                 idx[:, :, None, None].expand(B, self.K_rand, N, T_len)
#             )  # [B,K_rand,N,T]
#         else:
#             idx = None
#             Y_rand = Y_full[:, :0, ...]

#         # 模型生成部分
#         if self.K_model > 0:
#             if is_train:
#                 log_p = torch.log(torch.clamp(p, 1e-12, 1.0))
#                 A = F.gumbel_softmax(
#                     log_p,
#                     tau=max(float(tau_gumbel), 1e-6),
#                     hard=False,
#                     dim=-1
#                 )  # [B,K_model,S]
#                 Y_mix = torch.einsum("bks,bsnt->bknt", A, Y_full)
#             else:
#                 A = p
#                 Y_mix = torch.einsum("bks,bsnt->bknt", p, Y_full)

#             Y_sel = torch.cat([Y_rand, Y_mix], dim=1)      # [B,K,N,T]
#         else:
#             A = torch.empty((B, 0, S), device=device, dtype=dtype)
#             Y_sel = Y_rand

#         aux = {
#             "p": p,
#             "A": A,
#             "idx_rand": idx,
#         }

#         Y_sel = Y_sel.permute(1, 0, 2, 3).contiguous()     # [K,B,N,T]
#         return Y_sel, aux


# class RandomScenarioSelector(torch.nn.Module):
#     def __init__(self, n_scen: int):
#         super().__init__()
#         self.n_scen = int(n_scen)

#     def forward(self, Y_scen, is_train=False, **kwargs):
#         # Y_scen: [S_full, B, ...]
#         S_full, B = Y_scen.shape[0], Y_scen.shape[1]
#         K = min(self.n_scen, S_full)
#         print(Y_scen)
#         print(Y_scen.shape)
#         # choose K unique indices (same for the whole batch), simplest baseline
#         perm = torch.randperm(S_full, device=Y_scen.device)
#         print(perm)
#         idx = perm[:K]                      # [K]
#         Y_sel = Y_scen.index_select(0, idx) # [K, B, ...]

#         # Build aux to be compatible with your DFL_train diversity regularization
#         # p: [B, K, S_full] (one-hot selection)
#         p = Y_scen.new_zeros((B, K, S_full))
#         p[:, torch.arange(K, device=Y_scen.device), idx] = 1.0

#         aux = {"idx": idx,   # [K]
#             "p": p,       # [B, K, S_full]
#             "A": p,       # optional: some code checks "A" in aux_filter
#         }
#         return Y_sel, aux


# class ScenarioFilter(nn.Module):
#     def __init__(self, prob_type="multi", T=24, N_nodes=11, K=20, K_rand=10, hidden=128):
#         super().__init__()
#         print('new version')
#         self.prob_type = prob_type
#         self.K = int(K)
#         self.K_rand = int(K_rand)
#         self.K_model = self.K - self.K_rand

#         input_dim = T #if prob_type == "single" else N_nodes * T
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, hidden),
#             nn.ReLU(),
#             nn.Linear(hidden, self.K_model),
#         )

#     def forward(self, Y_scen, tau_gumbel=1.0, eps_uniform=0.10, is_train=True):
#         # Y_scen: [S, B, N, T]
#         print(eps_uniform)
#         Y_full = Y_scen.permute(1, 0, 2, 3).contiguous()   # [B, S, N, T]
#         B, S, N, T_len = Y_full.shape
#         device, dtype = Y_full.device, Y_full.dtype

#         #if self.prob_type == "single":
#         Y_feature = Y_full.sum(dim=2)                  # [B, S, T]
#         #else:
#         #    Y_feature = Y_full.reshape(B, S, -1)           # [B, S, N*T]

#         # logits: [B, S, K_model] -> [B, K_model, S]
#         logits = self.net(Y_feature.float()).transpose(1, 2)
#         #print('logits shape',logits.shape)
#         p = torch.softmax(logits / max(float(tau_gumbel), 1e-6), dim=-1).to(dtype)
#         #print('p shape',p.shape)

#         if eps_uniform > 0:
#             p = (1 - eps_uniform) * p + eps_uniform * (1.0 / S)
#             p = p / p.sum(dim=-1, keepdim=True)

#         # 随机部分（保留）
#         #et_seed(0)
#         if self.K_rand > 0:
#             idx = torch.stack([torch.randperm(S, device=device)[:self.K_rand] for _ in range(B)], dim=0)  # [B,K_rand]
#             #print(idx)
#             Y_rand = torch.gather(Y_full, 1, idx[:, :, None, None].expand(B, self.K_rand, N, T_len))      # [B,K_rand,N,T]
#         else:
#             idx = None
#             Y_rand = Y_full[:, :0, ...]

#         # 关键：train/eval统一都用 p 混合
#         if self.K_model > 0:
#             Y_mix = torch.einsum("bks,bsnt->bknt", p, Y_full)   # [B,K_model,N,T]
#             Y_sel = torch.cat([Y_rand, Y_mix], dim=1)           # [B,K,N,T]
#         else:
#             Y_sel = Y_rand

#         aux = {
#             "p": p,                    # [B,K_model,S]
#             "A": p,                    # 兼容旧代码
#             "idx_rand": idx
#         }

#         Y_sel = Y_sel.permute(1, 0, 2, 3).contiguous()          # [K,B,N,T]
#         return Y_sel, aux







# class ScenarioFilter(nn.Module):
#     def __init__(self, prob_type="multi", T=24, N_nodes=11, K=20, K_rand=10, hidden=128):
#         super().__init__()
#         self.prob_type = prob_type
#         self.K = int(K)
#         self.K_rand = int(K_rand)
#         self.K_model = self.K - self.K_rand

#         input_dim = T #if prob_type == "single" else N_nodes * T
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, hidden),
#             nn.ReLU(),
#             nn.Linear(hidden, self.K_model),
#         )

#     def forward(self, Y_scen, tau_gumbel=1.0, eps_uniform=0.10, is_train=True):
#         # Y_scen: [S, B, N, T]
        
#         Y_full = Y_scen.permute(1, 0, 2, 3).contiguous()   # [B, S, N, T]
#         B, S, N, T_len = Y_full.shape
#         device, dtype = Y_full.device, Y_full.dtype

#         #if self.prob_type == "single":
#         Y_feature = Y_full.sum(dim=2)                  # [B, S, T]
#         #else:
#         #    Y_feature = Y_full.reshape(B, S, -1)           # [B, S, N*T]

#         # logits: [B, S, K_model] -> [B, K_model, S]
#         logits = self.net(Y_feature.float()).transpose(1, 2)
#         #print('logits shape',logits.shape)
#         p = torch.softmax(logits / max(float(tau_gumbel), 1e-6), dim=-1).to(dtype)
#         #print('p shape',p.shape)

#         if eps_uniform > 0:
#             p = (1 - eps_uniform) * p + eps_uniform * (1.0 / S)
#             p = p / p.sum(dim=-1, keepdim=True)

#         # 随机部分（保留）
#         #et_seed(0)
#         if self.K_rand > 0:
#             idx = torch.stack([torch.randperm(S, device=device)[:self.K_rand] for _ in range(B)], dim=0)  # [B,K_rand]
#             #print(idx)
#             Y_rand = torch.gather(Y_full, 1, idx[:, :, None, None].expand(B, self.K_rand, N, T_len))      # [B,K_rand,N,T]
#         else:
#             idx = None
#             Y_rand = Y_full[:, :0, ...]

#         # 关键：train/eval统一都用 p 混合
#         if self.K_model > 0:
#             Y_mix = torch.einsum("bks,bsnt->bknt", p, Y_full)   # [B,K_model,N,T]
#             Y_sel = torch.cat([Y_rand, Y_mix], dim=1)           # [B,K,N,T]
#         else:
#             Y_sel = Y_rand

#         aux = {
#             "p": p,                    # [B,K_model,S]
#             "A": p,                    # 兼容旧代码
#             "idx_rand": idx
#         }

#         Y_sel = Y_sel.permute(1, 0, 2, 3).contiguous()          # [K,B,N,T]
#         return Y_sel, aux


# ============================================================
# 保留你原来的对齐组装函数
# ============================================================
@torch.no_grad()
def select_scenarios_eval_aligned(Y_full, w, K, K_rand, eps_uniform):
    B, S = Y_full.shape[0], Y_full.shape[1]
    device = Y_full.device
    K_model = K - K_rand
    is_multi = (Y_full.ndim == 4)

    if K_rand > 0:
        idx_rand = torch.stack([torch.randperm(S, device=device)[:K_rand] for _ in range(B)], 0)
        if is_multi:
            _, _, N_nodes, T = Y_full.shape
            Y_rand = torch.gather(Y_full, 1, idx_rand[:, :, None, None].expand(B, K_rand, N_nodes, T))
        else:
            _, _, T = Y_full.shape
            Y_rand = torch.gather(Y_full, 1, idx_rand[:, :, None].expand(B, K_rand, T))
    else:
        Y_rand = Y_full[:, :0, ...]

    if K_model == 0:
        return Y_rand
        
    p = w
    if eps_uniform > 0:
        p = (1 - eps_uniform) * p + eps_uniform * (1.0 / S)
        p = p / p.sum(dim=-1, keepdim=True)

    if is_multi:
        Y_mix = torch.einsum("bks,bsnt->bknt", p, Y_full)
    else:
        Y_mix = torch.einsum("bks,bst->bkt", p, Y_full)
        
    return torch.cat([Y_rand, Y_mix], dim=1) 

# ============================================================
# 无缝对接 E2E 架构的离线过滤效果验证函数
# ============================================================
def eval_filter_performance(
    args, 
    data,               # 例如: window_pack_m_vae_test_200 
    trained_filter,     # E2E 训练好的 dfl_trained.scenario_filter
    mgr_class,          # 例如: IEEE14_Reserve_SO_Manager_MultiNode
    prob_type="multi",
    K=20, 
    K_rand=10, 
    S_full=200, 
    n_jobs=8, 
    seed=42, 
    eps_uniform=0.10, 
    mode=None
):
    print(f"\n[验证] 正在进入离线 Benchmark 评估 (并行数: {n_jobs})...")
    set_seed(seed)
    prob_type = str(prob_type).lower()
    
    Y_pred = data["Y_pred"]            # (S, N_nodes, L)
    Y_true_all = data["Y_true"]        # (N_nodes, L)
    S, N_nodes, L = Y_pred.shape
    T = args.T
    days = list(range(L // T))
    pred_mean_all = Y_pred.mean(axis=0)  

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 切换为 Eval 模式
    trained_filter.eval()

    args_full = copy.deepcopy(args); args_full.N_scen = S_full
    args_K = copy.deepcopy(args); args_K.N_scen = K
    
    scen_map = {}
    
    # ================= 1. 预计算内存中的选景 =================
    for day in days:
        torch.manual_seed(seed + day)
        sl = slice(T * day, T * day + T)
        
        if prob_type == "single":
            Y_full_work_np = Y_pred[:, :, sl].sum(axis=1)           # (S, T)
            Y_full_eval_tensor = torch.tensor(Y_full_work_np, dtype=torch.float32, device=device).unsqueeze(0)
            Y_feature_flat = Y_full_eval_tensor
        else:
            Y_full_work_np = Y_pred[:, :, sl]                       # (S, 11, T)
            Y_full_eval_tensor = torch.tensor(Y_full_work_np, dtype=torch.float32, device=device).unsqueeze(0)
            Y_feature_flat = Y_full_eval_tensor.reshape(1, S, N_nodes * T)
            
        with torch.no_grad():
            # 核心：复用新架构里 Filter 内部的神经网络 net 计算权重 w
            logits = trained_filter.net(Y_feature_flat)            # (1, S, K_model)
            logits = logits.transpose(1, 2)                        # (1, K_model, S)
            w = torch.softmax(logits / 1.0, dim=-1)                # 温度 tau=1.0

            Y_sel_work = select_scenarios_eval_aligned(
                Y_full_eval_tensor, w, K=K, K_rand=K_rand, eps_uniform=eps_uniform
            )
        scen_map[day] = Y_sel_work[0].cpu().numpy()  # (K,T) or (K,11,T)

    # 包装执行数据的辅助函数
    def get_eval_numpy_vars(d, scen_work):
        sl = slice(T * d, T * d + T)
        if prob_type == "single":
            Y_true_np = Y_true_all[:, sl].sum(axis=0)       
            forecast_np = pred_mean_all[:, sl].sum(axis=0)  
        else:
            Y_true_np = Y_true_all[:, sl]                   
            forecast_np = pred_mean_all[:, sl]              
        
        if mode == "ideal":
            forecast_np = Y_true_np.copy()
            scen_work = Y_true_np.reshape((1, -1) if prob_type=="single" else (1, N_nodes, T))
            
        return scen_work, forecast_np, Y_true_np

    # ================= 2. 并行评估 =================
    def _job_full(day):
        scen_full = Y_pred[:, :, slice(T*day, T*day+T)].sum(axis=1) if prob_type=="single" else Y_pred[:, :, slice(T*day, T*day+T)]
        scen_full, forecast_np, Y_true_np = get_eval_numpy_vars(day, scen_full)
        mgr = mgr_class(args_full)
        mgr.build_model(forecast_np, scen_full, output_flag=0)
        mgr.solve(threads=1, method=1, warm_start=False, output_flag=0)
        return float(mgr.compute_true_cost(Y_true_np, output_flag=0, threads=1, method=1)["total_cost"])

    def _job_rand(day):
        rng = np.random.RandomState(seed + day)
        idx = rng.choice(S_full, size=K, replace=False)
        scen_rand = Y_pred[idx, :, slice(T*day, T*day+T)].sum(axis=1) if prob_type=="single" else Y_pred[idx, :, slice(T*day, T*day+T)]
        scen_rand, forecast_np, Y_true_np = get_eval_numpy_vars(day, scen_rand)
        mgr = mgr_class(args_K)
        mgr.build_model(forecast_np, scen_rand, output_flag=0)
        mgr.solve(threads=1, method=1, warm_start=False, output_flag=0)
        return float(mgr.compute_true_cost(Y_true_np, output_flag=0, threads=1, method=1)["total_cost"])

    def _job_model(day):
        scen_model, forecast_np, Y_true_np = get_eval_numpy_vars(day, scen_map[day])
        mgr = mgr_class(args_K)
        mgr.build_model(forecast_np, scen_model, output_flag=0)
        mgr.solve(threads=1, method=1, warm_start=False, output_flag=0)
        return float(mgr.compute_true_cost(Y_true_np, output_flag=0, threads=1, method=1)["total_cost"])

    cost_full = Parallel(n_jobs=n_jobs, backend="loky")(delayed(_job_full)(d) for d in days)
    cost_rand = Parallel(n_jobs=n_jobs, backend="loky")(delayed(_job_rand)(d) for d in days)
    cost_model = Parallel(n_jobs=n_jobs, backend="loky")(delayed(_job_model)(d) for d in days)

    full_vals, rand_vals, model_vals = np.asarray(cost_full), np.asarray(cost_rand), np.asarray(cost_model)

    print("-" * 50)
    print(f"gap(random-full)    = {(rand_vals.mean()-full_vals.mean())/abs(full_vals.mean()+1e-9):.6f}")
    print(f"gap(model-full)     = {(model_vals.mean()-full_vals.mean())/abs(full_vals.mean()+1e-9):.6f}")
    print(f"improve(model-rand) = {rand_vals.mean()-model_vals.mean():.6f} (越大越好)")
    print(f"winrate(model<rand) = {(model_vals < rand_vals).mean():.3f}")
    print("-" * 50)

    return {
        "fullS":  {d: float(c) for d, c in zip(days, cost_full)},
        "randomK":{d: float(c) for d, c in zip(days, cost_rand)},
        "modelK": {d: float(c) for d, c in zip(days, cost_model)},
    }