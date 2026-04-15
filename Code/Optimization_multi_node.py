import torch
import cvxpy as cp
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from cvxpylayers.torch import CvxpyLayer

import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from joblib import Parallel, delayed
import os


class IEEE14_Ideal_Manager_MultiNode:
    """
    IEEE14 ideal multi-node dispatch, NO network, perfect forecast.

    Features:
      - use true load directly in DA optimization
      - no stochastic optimization
      - no reserve requirement for uncertainty
      - no theta / no branch flow
      - keep nodal balance via system pool exchange S_pool
      - linear DA energy cost: Cost_DA = b*P + c

    This is a perfect-information lower-bound baseline.
    """

    def __init__(self, args):
        self.args = args

        self.column_mapping = {
            "4-2-1": 3, "4-2-3": 4, "4-2-8": 9, "4-2-2": 2, "4-2-5": 14,
            "4-2-6": 13, "4-2-9": 6, "4-2-4": 10, "4-2-0": 5,
            "4-2-7": 12, "4-2-10": 11
        }
        self.default_bus_order = list(self.column_mapping.values())

        self.all_buses = list(range(1, 15))
        self.ref_bus = 1

        gen_raw = {
            1: {"a": 0.04, "b": 20, "c": 0, "min": 0, "max": 332},
            2: {"a": 0.25, "b": 20, "c": 0, "min": 0, "max": 140},
            3: {"a": 0.01, "b": 40, "c": 0, "min": 0, "max": 100},
            6: {"a": 0.01, "b": 40, "c": 0, "min": 0, "max": 100},
            8: {"a": 0.01, "b": 40, "c": 0, "min": 0, "max": 100},
        }
        self.gen_info = {}
        cap_scale = float(getattr(args, "capacity_scale", 1.0))
        for g_id, info in gen_raw.items():
            self.gen_info[g_id] = info.copy()
            self.gen_info[g_id]["max"] = info["max"] * cap_scale

        self.gen_bus_list = list(self.gen_info.keys())
        self.gen_bus_of = {g: g for g in self.gen_bus_list}

        self.ramp_factor = float(getattr(args, "ramp_factor", 1.0))

        self.model = None
        self.vars = {}
        self.T = None
        self._true14 = None

    # ----------------- mapping helpers -----------------
    def map_11load_to_14bus(self, arr_11_T):
        arr_11_T = np.asarray(arr_11_T, dtype=float)
        if arr_11_T.ndim != 2 or arr_11_T.shape[0] != len(self.default_bus_order):
            raise ValueError(f"arr_11_T must be (11,T), got {arr_11_T.shape}")
        T = arr_11_T.shape[1]
        out = np.zeros((14, T), dtype=float)
        for j, bus in enumerate(self.default_bus_order):
            out[bus - 1, :] = arr_11_T[j, :]
        return out

    def _coerce_forecast14(self, arr):
        arr = np.asarray(arr, dtype=float)
        if arr.ndim != 2:
            raise ValueError(f"input must be 2D, got {arr.shape}")
        if arr.shape[0] == 14:
            return arr
        if arr.shape[0] == len(self.default_bus_order):
            return self.map_11load_to_14bus(arr)
        raise ValueError(f"input must be (11,T) or (14,T), got {arr.shape}")

    # ----------------- build/solve -----------------
    def build_model(self, actual, output_flag=0):
        """
        actual: (11,T) or (14,T), treated as true load known in advance
        """
        actual14 = self._coerce_forecast14(actual)
        T = actual14.shape[1]

        self.T = int(T)
        self._true14 = actual14.copy()

        m = gp.Model("IEEE14_Ideal_NoNetwork")
        m.setParam("OutputFlag", int(output_flag))

        P_DA = m.addVars(self.gen_bus_list, range(T), lb=0, name="P_DA")
        Cost_DA = m.addVars(self.gen_bus_list, range(T), lb=0, name="Cost_DA")

        # keep reserve vars for output compatibility, but force 0
        R_up = m.addVars(self.gen_bus_list, range(T), lb=0, ub=0, name="R_up")
        R_dn = m.addVars(self.gen_bus_list, range(T), lb=0, ub=0, name="R_dn")

        S_pool = m.addVars(self.all_buses, range(T), lb=-GRB.INFINITY, name="S_pool")

        ramp_factor = float(self.ramp_factor)

        obj = gp.quicksum(Cost_DA[g, t] for g in self.gen_bus_list for t in range(T))
        m.setObjective(obj, GRB.MINIMIZE)

        for g in self.gen_bus_list:
            pmin, pmax = float(self.gen_info[g]["min"]), float(self.gen_info[g]["max"])
            b_lin = float(self.gen_info[g]["b"])
            c0 = float(self.gen_info[g]["c"])

            for t in range(T):
                m.addConstr(P_DA[g, t] >= pmin, name=f"pmin_{g}_{t}")
                m.addConstr(P_DA[g, t] <= pmax, name=f"pmax_{g}_{t}")
                m.addConstr(Cost_DA[g, t] == b_lin * P_DA[g, t] + c0, name=f"lin_cost_{g}_{t}")

            for t in range(1, T):
                m.addConstr(P_DA[g, t] - P_DA[g, t - 1] <= pmax * ramp_factor, name=f"ramp_up_{g}_{t}")
                m.addConstr(P_DA[g, t - 1] - P_DA[g, t] <= pmax * ramp_factor, name=f"ramp_dn_{g}_{t}")

        for t in range(T):
            m.addConstr(gp.quicksum(S_pool[b, t] for b in self.all_buses) == 0.0, name=f"pool_{t}")
            for b in self.all_buses:
                gen_inj = gp.quicksum(P_DA[g, t] for g in self.gen_bus_list if self.gen_bus_of[g] == b)
                load = float(actual14[b - 1, t])
                m.addConstr(gen_inj - load == S_pool[b, t], name=f"nodalBal_{b}_{t}")

        m.update()
        self.model = m
        self.vars = {
            "P_DA": P_DA,
            "R_up": R_up,
            "R_dn": R_dn,
            "Cost_DA": Cost_DA,
            "S_pool": S_pool,
        }

    def solve(self, warm_start=False, threads=None, method=None, output_flag=None):
        if self.model is None:
            raise RuntimeError("Call build_model() first.")

        if output_flag is not None:
            self.model.setParam("OutputFlag", int(output_flag))
        if warm_start:
            self.model.setParam("LPWarmStart", 1)
        if threads is not None:
            self.model.setParam("Threads", int(threads))
        if method is not None:
            self.model.setParam("Method", int(method))

        self.model.optimize()
        if self.model.status != GRB.OPTIMAL:
            print(f"Status: {self.model.status}")

    def get_stage1_results(self):
        if self.model is None or self.model.status != GRB.OPTIMAL:
            return None

        T = self.T
        G_list = self.gen_bus_list
        P = np.zeros((T, len(G_list)))
        Rup = np.zeros((T, len(G_list)))
        Rdn = np.zeros((T, len(G_list)))
        for i, g in enumerate(G_list):
            for t in range(T):
                P[t, i] = float(self.vars["P_DA"][g, t].X)
                Rup[t, i] = float(self.vars["R_up"][g, t].X)
                Rdn[t, i] = float(self.vars["R_dn"][g, t].X)

        return (
            pd.DataFrame(P, columns=[f"Gen_{g}" for g in G_list]),
            pd.DataFrame(Rup, columns=[f"Rup_{g}" for g in G_list]),
            pd.DataFrame(Rdn, columns=[f"Rdn_{g}" for g in G_list]),
            {"Objective": float(self.model.ObjVal)},
        )

    def compute_true_cost(self, actual=None, **kwargs):
        """
        Perfect forecast => realized cost equals DA solved cost.
        """
        if self.model is None or self.model.status != GRB.OPTIMAL:
            raise RuntimeError("Model not solved to OPTIMAL. Call solve() first.")

        T = self.T
        G_list = self.gen_bus_list

        P_DA = np.array([[self.vars["P_DA"][g, t].X for g in G_list] for t in range(T)], dtype=float)
        R_up = np.zeros_like(P_DA)
        R_dn = np.zeros_like(P_DA)
        Cost_DA = np.array([[self.vars["Cost_DA"][g, t].X for g in G_list] for t in range(T)], dtype=float)

        stage1_energy_cost_pwl = float(Cost_DA.sum())
        stage1_reserve_cost = 0.0
        rt_deploy_cost = 0.0
        rt_penalty_cost = 0.0
        total_cost = stage1_energy_cost_pwl

        Omega14 = np.zeros((14, T), dtype=float)
        deploy_up = np.zeros_like(P_DA)
        deploy_dn = np.zeros_like(P_DA)
        LS = np.zeros((T, 14), dtype=float)
        SP = np.zeros((T, 14), dtype=float)
        S_pool_rt = np.zeros((T, 14), dtype=float)

        return {
            "stage1_energy_cost_pwl": float(stage1_energy_cost_pwl),
            "stage1_reserve_cost": float(stage1_reserve_cost),
            "rt_deploy_cost": float(rt_deploy_cost),
            "rt_penalty_cost": float(rt_penalty_cost),
            "total_cost": float(total_cost),

            "Omega14": Omega14,
            "P_DA": P_DA,
            "R_up": R_up,
            "R_dn": R_dn,
            "Cost_DA": Cost_DA,
            "deploy_up": deploy_up,
            "deploy_dn": deploy_dn,
            "LS": LS,
            "SP": SP,
            "S_pool_rt": S_pool_rt,
            "LS_sys_t": LS.sum(axis=1),
            "SP_sys_t": SP.sum(axis=1),
        }


class IEEE14_Reserve_Deterministic_DA_Manager_MultiNode:
    """
    IEEE14 deterministic reserve scheduling, multi-node, NO network.

    Features:
      - no stochastic optimization / no SAA
      - no theta / no branch flow
      - keep nodal balance via system pool exchange S_pool
      - linear DA energy cost: Cost_DA = b*P + c
      - system-wide reserve requirements:
            sum_g R_up[g,t] >= 2% * total forecast load at t
            sum_g R_dn[g,t] >= 2% * total forecast load at t
      - compute_true_cost uses the same no-network + nodal balance + pool assumption
    """

    def __init__(self, args):
        self.args = args

        self.column_mapping = {
            "4-2-1": 3, "4-2-3": 4, "4-2-8": 9, "4-2-2": 2, "4-2-5": 14,
            "4-2-6": 13, "4-2-9": 6, "4-2-4": 10, "4-2-0": 5,
            "4-2-7": 12, "4-2-10": 11
        }
        self.default_bus_order = list(self.column_mapping.values())

        self.all_buses = list(range(1, 15))
        self.ref_bus = 1

        gen_raw = {
            1: {"a": 0.04, "b": 20, "c": 0, "min": 0, "max": 332},
            2: {"a": 0.25, "b": 20, "c": 0, "min": 0, "max": 140},
            3: {"a": 0.01, "b": 40, "c": 0, "min": 0, "max": 100},
            6: {"a": 0.01, "b": 40, "c": 0, "min": 0, "max": 100},
            8: {"a": 0.01, "b": 40, "c": 0, "min": 0, "max": 100},
        }
        self.gen_info = {}
        cap_scale = float(getattr(args, "capacity_scale", 1.0))
        for g_id, info in gen_raw.items():
            self.gen_info[g_id] = info.copy()
            self.gen_info[g_id]["max"] = info["max"] * cap_scale

        self.gen_bus_list = list(self.gen_info.keys())
        self.gen_bus_of = {g: g for g in self.gen_bus_list}

        self.ramp_factor = float(getattr(args, "ramp_factor", 1.0))

        self.model = None
        self.vars = {}
        self.T = None
        self._forecast14 = None

    # ----------------- mapping helpers -----------------
    def map_11load_to_14bus(self, arr_11_T):
        arr_11_T = np.asarray(arr_11_T, dtype=float)
        if arr_11_T.ndim != 2 or arr_11_T.shape[0] != len(self.default_bus_order):
            raise ValueError(f"arr_11_T must be (11,T), got {arr_11_T.shape}")
        T = arr_11_T.shape[1]
        out = np.zeros((14, T), dtype=float)
        for j, bus in enumerate(self.default_bus_order):
            out[bus - 1, :] = arr_11_T[j, :]
        return out

    def _coerce_forecast14(self, forecast):
        forecast = np.asarray(forecast, dtype=float)
        if forecast.ndim != 2:
            raise ValueError(f"forecast must be 2D, got {forecast.shape}")
        if forecast.shape[0] == 14:
            return forecast
        if forecast.shape[0] == len(self.default_bus_order):
            return self.map_11load_to_14bus(forecast)
        raise ValueError(f"forecast must be (11,T) or (14,T), got {forecast.shape}")

    # ----------------- build/solve -----------------
    def build_model(self, forecast, output_flag=0):
        """
        Deterministic DA model:
          - NO network
          - nodal balance via S_pool
          - no scenarios
          - enforce system reserve up/down = 2% of total forecast load
        """
        forecast14 = self._coerce_forecast14(forecast)   # (14,T)
        T = forecast14.shape[1]

        self.T = int(T)
        self._forecast14 = forecast14.copy()

        m = gp.Model("IEEE14_Reserve_DA_NoNetwork")
        m.setParam("OutputFlag", int(output_flag))

        # ---------- Stage-1 vars ----------
        P_DA = m.addVars(self.gen_bus_list, range(T), lb=0, name="P_DA")
        R_up = m.addVars(self.gen_bus_list, range(T), lb=0, name="R_up")
        R_dn = m.addVars(self.gen_bus_list, range(T), lb=0, name="R_dn")
        Cost_DA = m.addVars(self.gen_bus_list, range(T), lb=0, name="Cost_DA")

        # pool exchange (free sign)
        S_pool = m.addVars(self.all_buses, range(T), lb=-GRB.INFINITY, name="S_pool")

        # ---------- parameters ----------
        reserve_req_up_ratio = float(getattr(self.args, "system_reserve_up_ratio", 0.02))
        reserve_req_dn_ratio = float(getattr(self.args, "system_reserve_dn_ratio", 0.02))

        reserve_dn_ratio = float(getattr(self.args, "reserve_dn_ratio", 0.02))
        reserve_up_ratio = float(getattr(self.args, "reserve_up_ratio", 0.05))
        ramp_factor = float(self.ramp_factor)

        b_g = {g: float(self.gen_info[g]["b"]) for g in self.gen_bus_list}
        price_res_up = {g: reserve_up_ratio * b_g[g] for g in self.gen_bus_list}
        price_res_dn = {g: reserve_dn_ratio * b_g[g] for g in self.gen_bus_list}

        # ---------- objective ----------
        obj_stage1_energy = gp.quicksum(Cost_DA[g, t] for g in self.gen_bus_list for t in range(T))
        obj_stage1_reserve = gp.quicksum(
            price_res_up[g] * R_up[g, t] + price_res_dn[g] * R_dn[g, t]
            for g in self.gen_bus_list for t in range(T)
        )

        m.setObjective(obj_stage1_energy + obj_stage1_reserve, GRB.MINIMIZE)

        # ---------- generator constraints ----------
        for g in self.gen_bus_list:
            pmin, pmax = float(self.gen_info[g]["min"]), float(self.gen_info[g]["max"])
            b_lin = float(self.gen_info[g]["b"])
            c0 = float(self.gen_info[g]["c"])

            for t in range(T):
                m.addConstr(P_DA[g, t] >= pmin, name=f"pmin_{g}_{t}")
                m.addConstr(P_DA[g, t] <= pmax, name=f"pmax_{g}_{t}")

                m.addConstr(P_DA[g, t] + R_up[g, t] <= pmax, name=f"cap_up_{g}_{t}")
                m.addConstr(P_DA[g, t] - R_dn[g, t] >= pmin, name=f"cap_dn_{g}_{t}")

                m.addConstr(Cost_DA[g, t] == b_lin * P_DA[g, t] + c0, name=f"lin_cost_{g}_{t}")

            for t in range(1, T):
                m.addConstr(P_DA[g, t] - P_DA[g, t - 1] <= pmax * ramp_factor, name=f"ramp_up_{g}_{t}")
                m.addConstr(P_DA[g, t - 1] - P_DA[g, t] <= pmax * ramp_factor, name=f"ramp_dn_{g}_{t}")

        # ---------- NO NETWORK: nodal balance with system pool ----------
        for t in range(T):
            total_load_t = float(np.sum(forecast14[:, t]))

            # pool conservation
            m.addConstr(gp.quicksum(S_pool[b, t] for b in self.all_buses) == 0.0, name=f"pool_DA_{t}")

            # nodal balance
            for b in self.all_buses:
                gen_inj = gp.quicksum(P_DA[g, t] for g in self.gen_bus_list if self.gen_bus_of[g] == b)
                load = float(forecast14[b - 1, t])
                m.addConstr(gen_inj - load == S_pool[b, t], name=f"nodalBal_DA_{b}_{t}")

            # system reserve requirements: up/down both 2% of total load
            m.addConstr(
                gp.quicksum(R_up[g, t] for g in self.gen_bus_list) >= reserve_req_up_ratio * total_load_t,
                name=f"sysResUp_{t}"
            )
            m.addConstr(
                gp.quicksum(R_dn[g, t] for g in self.gen_bus_list) >= reserve_req_dn_ratio * total_load_t,
                name=f"sysResDn_{t}"
            )

        m.update()
        self.model = m
        self.vars = {
            "P_DA": P_DA,
            "R_up": R_up,
            "R_dn": R_dn,
            "Cost_DA": Cost_DA,
            "S_pool": S_pool,
        }

    def solve(self, warm_start=False, threads=None, method=None, output_flag=None):
        if self.model is None:
            raise RuntimeError("Call build_model() first.")

        if output_flag is not None:
            self.model.setParam("OutputFlag", int(output_flag))
        if warm_start:
            self.model.setParam("LPWarmStart", 1)
        if threads is not None:
            self.model.setParam("Threads", int(threads))
        if method is not None:
            self.model.setParam("Method", int(method))

        self.model.optimize()
        if self.model.status != GRB.OPTIMAL:
            print(f"Status: {self.model.status}")

    def get_stage1_results(self):
        if self.model is None or self.model.status != GRB.OPTIMAL:
            return None

        T = self.T
        G_list = self.gen_bus_list
        P = np.zeros((T, len(G_list)))
        Rup = np.zeros((T, len(G_list)))
        Rdn = np.zeros((T, len(G_list)))
        for i, g in enumerate(G_list):
            for t in range(T):
                P[t, i] = float(self.vars["P_DA"][g, t].X)
                Rup[t, i] = float(self.vars["R_up"][g, t].X)
                Rdn[t, i] = float(self.vars["R_dn"][g, t].X)

        return (
            pd.DataFrame(P, columns=[f"Gen_{g}" for g in G_list]),
            pd.DataFrame(Rup, columns=[f"Rup_{g}" for g in G_list]),
            pd.DataFrame(Rdn, columns=[f"Rdn_{g}" for g in G_list]),
            {"Objective": float(self.model.ObjVal)},
        )

    def compute_true_cost(self, actual, mu=None, output_flag=0, warm_start=False, threads=None, method=None):
        """
        True-cost evaluation with the SAME assumption:
          - NO network
          - nodal balance via system pool exchange
          - LS/SP per bus
          - RT solved in one model for all t
        """
        if self.model is None or self.model.status != GRB.OPTIMAL:
            raise RuntimeError("Model not solved to OPTIMAL. Call solve() first.")

        # --- forecast14 ---
        if mu is None:
            if self._forecast14 is None:
                raise RuntimeError("No stored forecast found. Pass mu explicitly.")
            forecast14 = np.asarray(self._forecast14, dtype=float)
        else:
            forecast14 = self._coerce_forecast14(mu)

        # --- actual14 ---
        actual14 = self._coerce_forecast14(actual)
        if actual14.shape != forecast14.shape:
            raise ValueError(f"actual shape {actual14.shape} != forecast shape {forecast14.shape}")

        T = forecast14.shape[1]
        G_list = self.gen_bus_list
        Buses = self.all_buses

        reserve_dn_ratio = float(getattr(self.args, "reserve_dn_ratio", 0.02))
        reserve_up_ratio = float(getattr(self.args, "reserve_up_ratio", 0.05))
        rt_dn_ratio = float(getattr(self.args, "rt_dn_ratio", 0.5))
        rt_up_ratio = float(getattr(self.args, "rt_up_ratio", 2.0))
        voll = float(getattr(self.args, "voll", 1000.0))
        vosp = float(getattr(self.args, "vosp", 50.0))

        b_g = {g: float(self.gen_info[g]["b"]) for g in G_list}
        price_res_up = {g: reserve_up_ratio * b_g[g] for g in G_list}
        price_res_dn = {g: reserve_dn_ratio * b_g[g] for g in G_list}
        price_rt_up = {g: rt_up_ratio * b_g[g] for g in G_list}
        price_rt_dn = {g: rt_dn_ratio * b_g[g] for g in G_list}

        # --- extract stage-1 decisions ---
        P_DA = np.array([[self.vars["P_DA"][g, t].X for g in G_list] for t in range(T)], dtype=float)
        R_up = np.array([[self.vars["R_up"][g, t].X for g in G_list] for t in range(T)], dtype=float)
        R_dn = np.array([[self.vars["R_dn"][g, t].X for g in G_list] for t in range(T)], dtype=float)
        Cost_DA = np.array([[self.vars["Cost_DA"][g, t].X for g in G_list] for t in range(T)], dtype=float)

        stage1_energy_cost_pwl = float(Cost_DA.sum())
        stage1_reserve_cost = 0.0
        for gi, g in enumerate(G_list):
            stage1_reserve_cost += float(np.sum(price_res_up[g] * R_up[:, gi] + price_res_dn[g] * R_dn[:, gi]))
        stage1_reserve_cost = float(stage1_reserve_cost)

        Omega14 = actual14 - forecast14

        gen_to_idx = {g: i for i, g in enumerate(G_list)}

        # ---- ONE RT model for all T ----
        sm = gp.Model("RT_NoNetwork_AllT")
        sm.setParam("OutputFlag", int(output_flag))
        if warm_start:
            sm.setParam("LPWarmStart", 1)
        if threads is not None:
            sm.setParam("Threads", int(threads))
        if method is not None:
            sm.setParam("Method", int(method))

        du = sm.addVars(G_list, range(T), lb=0, name="du")
        dd = sm.addVars(G_list, range(T), lb=0, name="dd")
        ls = sm.addVars(Buses, range(T), lb=0, name="ls")
        sp = sm.addVars(Buses, range(T), lb=0, name="sp")
        s_pool = sm.addVars(Buses, range(T), lb=-GRB.INFINITY, name="s_pool")

        for t in range(T):
            for g in G_list:
                gi = gen_to_idx[g]
                sm.addConstr(du[g, t] <= float(R_up[t, gi]), name=f"du_le_Rup_{g}_{t}")
                sm.addConstr(dd[g, t] <= float(R_dn[t, gi]), name=f"dd_le_Rdn_{g}_{t}")

            sm.addConstr(gp.quicksum(s_pool[b, t] for b in Buses) == 0.0, name=f"pool_{t}")

            for b in Buses:
                gen_inj = gp.quicksum(
                    float(P_DA[t, gen_to_idx[g]]) + du[g, t] - dd[g, t]
                    for g in G_list if self.gen_bus_of[g] == b
                )
                load = float(actual14[b - 1, t])
                sm.addConstr(
                    gen_inj - sp[b, t] - (load - ls[b, t]) == s_pool[b, t],
                    name=f"nodal_{b}_{t}"
                )

        sm.setObjective(
            gp.quicksum(price_rt_up[g] * du[g, t] + price_rt_dn[g] * dd[g, t]
                        for g in G_list for t in range(T))
            + voll * gp.quicksum(ls[b, t] for b in Buses for t in range(T))
            + vosp * gp.quicksum(sp[b, t] for b in Buses for t in range(T)),
            GRB.MINIMIZE
        )

        sm.optimize()
        if sm.status != GRB.OPTIMAL:
            raise RuntimeError(f"RT all-T subproblem not optimal, status={sm.status}")

        deploy_up = np.zeros((T, len(G_list)), dtype=float)
        deploy_dn = np.zeros((T, len(G_list)), dtype=float)
        LS = np.zeros((T, 14), dtype=float)
        SP = np.zeros((T, 14), dtype=float)
        S_pool_rt = np.zeros((T, 14), dtype=float)

        for t in range(T):
            for g in G_list:
                gi = gen_to_idx[g]
                deploy_up[t, gi] = float(du[g, t].X)
                deploy_dn[t, gi] = float(dd[g, t].X)
            for b in Buses:
                LS[t, b - 1] = float(ls[b, t].X)
                SP[t, b - 1] = float(sp[b, t].X)
                S_pool_rt[t, b - 1] = float(s_pool[b, t].X)

        rt_deploy_cost = float(sum(price_rt_up[g] * du[g, t].X + price_rt_dn[g] * dd[g, t].X
                                   for g in G_list for t in range(T)))
        rt_penalty_cost = float(
            voll * sum(ls[b, t].X for b in Buses for t in range(T))
            + vosp * sum(sp[b, t].X for b in Buses for t in range(T))
        )

        sm.dispose()

        total_cost = stage1_energy_cost_pwl + stage1_reserve_cost + rt_deploy_cost + rt_penalty_cost

        return {
            "stage1_energy_cost_pwl": float(stage1_energy_cost_pwl),
            "stage1_reserve_cost": float(stage1_reserve_cost),
            "rt_deploy_cost": float(rt_deploy_cost),
            "rt_penalty_cost": float(rt_penalty_cost),
            "total_cost": float(total_cost),

            "Omega14": Omega14,
            "P_DA": P_DA,
            "R_up": R_up,
            "R_dn": R_dn,
            "Cost_DA": Cost_DA,
            "deploy_up": deploy_up,
            "deploy_dn": deploy_dn,
            "LS": LS,
            "SP": SP,
            "S_pool_rt": S_pool_rt,
            "LS_sys_t": LS.sum(axis=1),
            "SP_sys_t": SP.sum(axis=1),
        }


class IEEE14_Reserve_SO_Manager_MultiNode:
    """
    IEEE14 reserve-based RT recourse (d_up/d_dn + LS/SP) two-stage SAA.

    MODIFIED:
      - 去掉网络(不建theta/f，不建线路潮流)
      - 仍然保留“每个节点的平衡”，但用“系统池交换”变量 S_pool / S_pool_s 来闭合各节点
      - true cost 的 RT 子问题也用同样的“无网络+节点平衡+系统池交换”假设，保证口径统一
      - 去掉PWL；一阶段能量成本：Cost_DA = b*P + c，并在目标中 sum(Cost_DA)

    SPEEDUP (新增):
      - compute_true_cost_fast(): 1 个 Gurobi 模型一次性解完所有 t=0..T-1 的 RT true-cost
        （替代旧版 compute_true_cost 每个 t 建一个模型并 optimize 一次）
      - compute_true_cost(): 默认调用 compute_true_cost_fast（保留旧版为 compute_true_cost_slow）
    """

    def __init__(self, args):
        self.args = args

        # --- 11负荷列 -> 14母线号 的映射（确保你11维顺序与此一致）---
        self.column_mapping = {
            "4-2-1": 3, "4-2-3": 4, "4-2-8": 9, "4-2-2": 2, "4-2-5": 14,
            "4-2-6": 13, "4-2-9": 6, "4-2-4": 10, "4-2-0": 5,
            "4-2-7": 12, "4-2-10": 11
        }
        self.default_bus_order = list(self.column_mapping.values())  # length=11

        self.all_buses = list(range(1, 15))
        self.ref_bus = 1

        # --- generators (at bus id = gen id) ---
        gen_raw = {
            1: {"a": 0.04, "b": 20, "c": 0, "min": 0, "max": 332},
            2: {"a": 0.25, "b": 20, "c": 0, "min": 0, "max": 140},
            3: {"a": 0.01, "b": 40, "c": 0, "min": 0, "max": 100},
            6: {"a": 0.01, "b": 40, "c": 0, "min": 0, "max": 100},
            8: {"a": 0.01, "b": 40, "c": 0, "min": 0, "max": 100},
        }
        self.gen_info = {}
        cap_scale = float(getattr(args, "capacity_scale", 1.0))
        for g_id, info in gen_raw.items():
            self.gen_info[g_id] = info.copy()
            self.gen_info[g_id]["max"] = info["max"] * cap_scale

        self.gen_bus_list = list(self.gen_info.keys())
        self.gen_bus_of = {g: g for g in self.gen_bus_list}  # gen g at bus g

        self.ramp_factor = float(getattr(args, "ramp_factor", 1.0))

        self.model = None
        self.vars = {}
        self.T = None
        self.N_scen = None
        self._forecast14 = None

    # ----------------- mapping helpers -----------------
    def map_11load_to_14bus(self, arr_11_T):
        arr_11_T = np.asarray(arr_11_T, dtype=float)
        if arr_11_T.ndim != 2 or arr_11_T.shape[0] != len(self.default_bus_order):
            raise ValueError(f"arr_11_T must be (11,T), got {arr_11_T.shape}")
        T = arr_11_T.shape[1]
        out = np.zeros((14, T), dtype=float)
        for j, bus in enumerate(self.default_bus_order):
            out[bus - 1, :] = arr_11_T[j, :]
        return out

    def map_scenarios_to_14bus(self, scen_N_11_T):
        scen_N_11_T = np.asarray(scen_N_11_T, dtype=float)
        if scen_N_11_T.ndim != 3 or scen_N_11_T.shape[1] != len(self.default_bus_order):
            raise ValueError(f"scen_N_11_T must be (N,11,T), got {scen_N_11_T.shape}")
        N, _, T = scen_N_11_T.shape
        out = np.zeros((N, 14, T), dtype=float)
        for j, bus in enumerate(self.default_bus_order):
            out[:, bus - 1, :] = scen_N_11_T[:, j, :]
        return out

    def _coerce_forecast14(self, forecast):
        forecast = np.asarray(forecast, dtype=float)
        if forecast.ndim != 2:
            raise ValueError(f"forecast must be 2D, got {forecast.shape}")
        if forecast.shape[0] == 14:
            return forecast
        if forecast.shape[0] == len(self.default_bus_order):  # 11
            return self.map_11load_to_14bus(forecast)
        raise ValueError(f"forecast must be (11,T) or (14,T), got {forecast.shape}")

    def _coerce_scen14(self, scen):
        scen = np.asarray(scen, dtype=float)
        if scen.ndim != 3:
            raise ValueError(f"scenarios must be 3D, got {scen.shape}")
        if scen.shape[1] == 14:
            return scen
        if scen.shape[1] == len(self.default_bus_order):  # 11
            return self.map_scenarios_to_14bus(scen)
        raise ValueError(f"scenarios must be (N,11,T) or (N,14,T), got {scen.shape}")

    # ----------------- build/solve -----------------
    def build_model(self, forecast, scenarios, pwl_n_points=4, output_flag=0):
        """
        Notes:
          - NO network (no theta/f, no DC flow constraints).
          - Keep nodal balance at each bus using a "system pool exchange" variable:
                gen - load = S_pool[b,t]
            and enforce pool conservation:
                sum_b S_pool[b,t] = 0
          - Scenario RT nodal balance uses S_pool_s similarly.
          - MODIFIED: remove PWL; Cost_DA is linear b*P (+c)
        """
        forecast14 = self._coerce_forecast14(forecast)   # (14,T)
        scen14 = self._coerce_scen14(scenarios)          # (N,14,T)

        N, _, T = scen14.shape
        if forecast14.shape[1] != T:
            raise ValueError("T mismatch between forecast and scenarios")

        self.T = int(T)
        self.N_scen = int(N)
        self._forecast14 = forecast14.copy()

        m = gp.Model("IEEE14_Reserve_SO_NoNetwork")
        m.setParam("OutputFlag", int(output_flag))

        # ---------- Stage-1 vars ----------
        P_DA = m.addVars(self.gen_bus_list, range(T), lb=0, name="P_DA")
        R_up = m.addVars(self.gen_bus_list, range(T), lb=0, name="R_up")
        R_dn = m.addVars(self.gen_bus_list, range(T), lb=0, name="R_dn")
        Cost_DA = m.addVars(self.gen_bus_list, range(T), lb=0, name="Cost_DA")

        # pool exchange (free sign)
        S_pool = m.addVars(self.all_buses, range(T), lb=-GRB.INFINITY, name="S_pool")

        # ---------- Stage-2 vars (scenario) ----------
        d_up = m.addVars(self.gen_bus_list, range(N), range(T), lb=0, name="d_up")
        d_dn = m.addVars(self.gen_bus_list, range(N), range(T), lb=0, name="d_dn")

        LS = m.addVars(range(N), self.all_buses, range(T), lb=0, name="LS")
        SP = m.addVars(range(N), self.all_buses, range(T), lb=0, name="SP")
        S_pool_s = m.addVars(range(N), self.all_buses, range(T), lb=-GRB.INFINITY, name="S_pool_s")

        rt_dn_ratio = float(self.args.rt_dn_ratio)
        rt_up_ratio = float(self.args.rt_up_ratio)
        reserve_dn_ratio = float(self.args.reserve_dn_ratio)
        reserve_up_ratio = float(self.args.reserve_up_ratio)
        vosp = float(self.args.vosp)
        voll = float(self.args.voll)
        ramp_factor = float(self.ramp_factor)

        b_g = {g: float(self.gen_info[g]["b"]) for g in self.gen_bus_list}
        price_res_up = {g: reserve_up_ratio * b_g[g] for g in self.gen_bus_list}
        price_res_dn = {g: reserve_dn_ratio * b_g[g] for g in self.gen_bus_list}
        price_rt_up = {g: rt_up_ratio * b_g[g] for g in self.gen_bus_list}
        price_rt_dn = {g: rt_dn_ratio * b_g[g] for g in self.gen_bus_list}

        obj_stage1_energy = gp.quicksum(Cost_DA[g, t] for g in self.gen_bus_list for t in range(T))
        obj_stage1_reserve = gp.quicksum(
            price_res_up[g] * R_up[g, t] + price_res_dn[g] * R_dn[g, t]
            for g in self.gen_bus_list for t in range(T)
        )
        obj_stage2_deploy = (1.0 / N) * gp.quicksum(
            price_rt_up[g] * d_up[g, n, t] + price_rt_dn[g] * d_dn[g, n, t]
            for g in self.gen_bus_list for n in range(N) for t in range(T)
        )
        obj_stage2_penalty = (1.0 / N) * gp.quicksum(
            voll * LS[n, b, t] + vosp * SP[n, b, t]
            for n in range(N) for b in self.all_buses for t in range(T)
        )
        m.setObjective(
            obj_stage1_energy + obj_stage1_reserve + obj_stage2_deploy + obj_stage2_penalty,
            GRB.MINIMIZE
        )

        for g in self.gen_bus_list:
            pmin, pmax = float(self.gen_info[g]["min"]), float(self.gen_info[g]["max"])
            b_lin = float(self.gen_info[g]["b"])
            c0 = float(self.gen_info[g]["c"])

            for t in range(T):
                m.addConstr(P_DA[g, t] >= pmin)
                m.addConstr(P_DA[g, t] <= pmax)

                # reserve feasibility w.r.t capacity
                m.addConstr(P_DA[g, t] + R_up[g, t] <= pmax, name=f"cap_up_{g}_{t}")
                m.addConstr(P_DA[g, t] - R_dn[g, t] >= pmin, name=f"cap_dn_{g}_{t}")

                # linear cost only
                m.addConstr(Cost_DA[g, t] == b_lin * P_DA[g, t] + c0, name=f"lin_cost_{g}_{t}")

            # ramp on DA schedule
            for t in range(1, T):
                m.addConstr(P_DA[g, t] - P_DA[g, t - 1] <= pmax * ramp_factor)
                m.addConstr(P_DA[g, t - 1] - P_DA[g, t] <= pmax * ramp_factor)

            # deployment bounded by reserved bands
            for n in range(N):
                for t in range(T):
                    m.addConstr(d_up[g, n, t] <= R_up[g, t])
                    m.addConstr(d_dn[g, n, t] <= R_dn[g, t])

        # ---------------- NO NETWORK: nodal balance with system pool ----------------
        # DA: nodal balance per bus + pool conservation per time
        for t in range(T):
            m.addConstr(gp.quicksum(S_pool[b, t] for b in self.all_buses) == 0.0, name=f"pool_DA_{t}")
            for b in self.all_buses:
                gen_inj = gp.quicksum(P_DA[g, t] for g in self.gen_bus_list if self.gen_bus_of[g] == b)
                load = float(forecast14[b - 1, t])
                m.addConstr(gen_inj - load == S_pool[b, t], name=f"nodalBal_DA_{b}_{t}")

        # RT: nodal balance per scenario/bus + pool conservation per time
        for n in range(N):
            for t in range(T):
                m.addConstr(
                    gp.quicksum(S_pool_s[n, b, t] for b in self.all_buses) == 0.0,
                    name=f"pool_RT_{n}_{t}"
                )
                for b in self.all_buses:
                    gen_inj_rt = gp.quicksum(
                        (P_DA[g, t] + d_up[g, n, t] - d_dn[g, n, t])
                        for g in self.gen_bus_list if self.gen_bus_of[g] == b
                    )
                    load_rt = float(scen14[n, b - 1, t])
                    m.addConstr(
                        gen_inj_rt - SP[n, b, t] - (load_rt - LS[n, b, t]) == S_pool_s[n, b, t],
                        name=f"nodalBal_RT_{n}_{b}_{t}"
                    )

        m.update()
        self.model = m
        self.vars = {
            "P_DA": P_DA,
            "R_up": R_up,
            "R_dn": R_dn,
            "Cost_DA": Cost_DA,
            "d_up": d_up,
            "d_dn": d_dn,
            "LS": LS,
            "SP": SP,
            "S_pool": S_pool,
            "S_pool_s": S_pool_s,
        }

    def solve(self, warm_start=False, threads=None, method=None, output_flag=None):
        """
        warm_start: LPWarmStart=1（保留basis/启动信息时可能有帮助）
        threads: 设置并行线程数（与外层并行二选一）
        method: Gurobi Method（0 auto, 1 dual simplex, 2 barrier, 3 concurrent, 4 deterministic concurrent）
        """
        if self.model is None:
            raise RuntimeError("Call build_model() first.")

        if output_flag is not None:
            self.model.setParam("OutputFlag", int(output_flag))
        if warm_start:
            self.model.setParam("LPWarmStart", 1)
        if threads is not None:
            self.model.setParam("Threads", int(threads))
        if method is not None:
            self.model.setParam("Method", int(method))

        self.model.optimize()
        if self.model.status == GRB.OPTIMAL:
            pass
        #   print(f"Solved. Obj: {self.model.ObjVal:.6f}")
        else:
            print(f"Status: {self.model.status}")

    def get_stage1_results(self):
        if self.model is None or self.model.status != GRB.OPTIMAL:
            return None

        T = self.T
        G_list = self.gen_bus_list
        P = np.zeros((T, len(G_list)))
        Rup = np.zeros((T, len(G_list)))
        Rdn = np.zeros((T, len(G_list)))
        for i, g in enumerate(G_list):
            for t in range(T):
                P[t, i] = float(self.vars["P_DA"][g, t].X)
                Rup[t, i] = float(self.vars["R_up"][g, t].X)
                Rdn[t, i] = float(self.vars["R_dn"][g, t].X)

        return (
            pd.DataFrame(P, columns=[f"Gen_{g}" for g in G_list]),
            pd.DataFrame(Rup, columns=[f"Rup_{g}" for g in G_list]),
            pd.DataFrame(Rdn, columns=[f"Rdn_{g}" for g in G_list]),
            {"Objective": float(self.model.ObjVal)},
        )

    def compute_true_cost(self, actual, mu=None, output_flag=0, warm_start=False, threads=None, method=None):
        """
        True-cost evaluation with the SAME assumption as build_model:
          - NO network
          - nodal balance via system pool exchange
          - LS/SP per bus

        SPEEDUP:
          - Build ONE RT model for all t in {0..T-1} and optimize once.
        """
        if self.model is None or self.model.status != GRB.OPTIMAL:
            raise RuntimeError("Model not solved to OPTIMAL. Call solve() first.")

        # --- forecast14 ---
        if mu is None:
            if self._forecast14 is None:
                raise RuntimeError("No stored forecast found. Pass mu explicitly.")
            forecast14 = np.asarray(self._forecast14, dtype=float)
        else:
            forecast14 = self._coerce_forecast14(mu)

        # --- actual14 ---
        actual14 = self._coerce_forecast14(actual)
        if actual14.shape != forecast14.shape:
            raise ValueError(f"actual shape {actual14.shape} != forecast shape {forecast14.shape}")

        T = forecast14.shape[1]
        G_list = self.gen_bus_list
        Buses = self.all_buses

        reserve_dn_ratio = float(self.args.reserve_dn_ratio)
        reserve_up_ratio = float(self.args.reserve_up_ratio)
        rt_dn_ratio = float(self.args.rt_dn_ratio)
        rt_up_ratio = float(self.args.rt_up_ratio)
        voll = float(self.args.voll)
        vosp = float(self.args.vosp)

        b_g = {g: float(self.gen_info[g]["b"]) for g in G_list}
        price_res_up = {g: reserve_up_ratio * b_g[g] for g in G_list}
        price_res_dn = {g: reserve_dn_ratio * b_g[g] for g in G_list}
        price_rt_up = {g: rt_up_ratio * b_g[g] for g in G_list}
        price_rt_dn = {g: rt_dn_ratio * b_g[g] for g in G_list}

        # --- extract stage-1 decisions ---
        P_DA = np.array([[self.vars["P_DA"][g, t].X for g in G_list] for t in range(T)], dtype=float)
        R_up = np.array([[self.vars["R_up"][g, t].X for g in G_list] for t in range(T)], dtype=float)
        R_dn = np.array([[self.vars["R_dn"][g, t].X for g in G_list] for t in range(T)], dtype=float)

        # STRICT stage-1 energy cost: use solved Cost_DA (now linear b*P+c)
        Cost_DA = np.array([[self.vars["Cost_DA"][g, t].X for g in G_list] for t in range(T)], dtype=float)
        stage1_energy_cost_pwl = float(Cost_DA.sum())

        stage1_reserve_cost = 0.0
        for gi, g in enumerate(G_list):
            stage1_reserve_cost += float(np.sum(price_res_up[g] * R_up[:, gi] + price_res_dn[g] * R_dn[:, gi]))
        stage1_reserve_cost = float(stage1_reserve_cost)

        # omega at each bus (actual - forecast)
        Omega14 = actual14 - forecast14  # (14,T)

        gen_to_idx = {g: i for i, g in enumerate(G_list)}

        # ---- ONE RT model for all T ----
        sm = gp.Model("RT_NoNetwork_AllT")
        sm.setParam("OutputFlag", int(output_flag))
        if warm_start:
            sm.setParam("LPWarmStart", 1)
        if threads is not None:
            sm.setParam("Threads", int(threads))
        if method is not None:
            sm.setParam("Method", int(method))

        du = sm.addVars(G_list, range(T), lb=0, name="du")
        dd = sm.addVars(G_list, range(T), lb=0, name="dd")
        ls = sm.addVars(Buses, range(T), lb=0, name="ls")
        sp = sm.addVars(Buses, range(T), lb=0, name="sp")
        s_pool = sm.addVars(Buses, range(T), lb=-GRB.INFINITY, name="s_pool")

        # constraints
        for t in range(T):
            # within reserves
            for g in G_list:
                gi = gen_to_idx[g]
                sm.addConstr(du[g, t] <= float(R_up[t, gi]), name=f"du_le_Rup_{g}_{t}")
                sm.addConstr(dd[g, t] <= float(R_dn[t, gi]), name=f"dd_le_Rdn_{g}_{t}")

            # pool conservation
            sm.addConstr(gp.quicksum(s_pool[b, t] for b in Buses) == 0.0, name=f"pool_{t}")

            # nodal balance
            for b in Buses:
                gen_inj = gp.quicksum(
                    float(P_DA[t, gen_to_idx[g]]) + du[g, t] - dd[g, t]
                    for g in G_list if self.gen_bus_of[g] == b
                )
                load = float(forecast14[b - 1, t] + Omega14[b - 1, t])
                sm.addConstr(
                    gen_inj - sp[b, t] - (load - ls[b, t]) == s_pool[b, t],
                    name=f"nodal_{b}_{t}"
                )

        # objective
        sm.setObjective(
            gp.quicksum(price_rt_up[g] * du[g, t] + price_rt_dn[g] * dd[g, t]
                        for g in G_list for t in range(T))
            + voll * gp.quicksum(ls[b, t] for b in Buses for t in range(T))
            + vosp * gp.quicksum(sp[b, t] for b in Buses for t in range(T)),
            GRB.MINIMIZE
        )

        sm.optimize()
        if sm.status != GRB.OPTIMAL:
            raise RuntimeError(f"RT all-T subproblem not optimal, status={sm.status}")

        deploy_up = np.zeros((T, len(G_list)), dtype=float)
        deploy_dn = np.zeros((T, len(G_list)), dtype=float)
        LS = np.zeros((T, 14), dtype=float)
        SP = np.zeros((T, 14), dtype=float)
        S_pool_rt = np.zeros((T, 14), dtype=float)

        for t in range(T):
            for g in G_list:
                gi = gen_to_idx[g]
                deploy_up[t, gi] = float(du[g, t].X)
                deploy_dn[t, gi] = float(dd[g, t].X)
            for b in Buses:
                LS[t, b - 1] = float(ls[b, t].X)
                SP[t, b - 1] = float(sp[b, t].X)
                S_pool_rt[t, b - 1] = float(s_pool[b, t].X)

        rt_deploy_cost = float(sum(price_rt_up[g] * du[g, t].X + price_rt_dn[g] * dd[g, t].X
                                   for g in G_list for t in range(T)))
        rt_penalty_cost = float(
            voll * sum(ls[b, t].X for b in Buses for t in range(T))
            + vosp * sum(sp[b, t].X for b in Buses for t in range(T))
        )

        sm.dispose()

        total_cost = stage1_energy_cost_pwl + stage1_reserve_cost + rt_deploy_cost + rt_penalty_cost

        return {
            "stage1_energy_cost_pwl": float(stage1_energy_cost_pwl),
            "stage1_reserve_cost": float(stage1_reserve_cost),
            "rt_deploy_cost": float(rt_deploy_cost),
            "rt_penalty_cost": float(rt_penalty_cost),
            "total_cost": float(total_cost),

            "Omega14": Omega14,   # (14,T)
            "P_DA": P_DA,
            "R_up": R_up,
            "R_dn": R_dn,
            "Cost_DA": Cost_DA,
            "deploy_up": deploy_up,
            "deploy_dn": deploy_dn,
            "LS": LS,             # (T,14)
            "SP": SP,             # (T,14)
            "S_pool_rt": S_pool_rt,
            "LS_sys_t": LS.sum(axis=1),
            "SP_sys_t": SP.sum(axis=1),
        }


class IEEE14_Reserve_DRO_Manager_MultiNode:
    """
    Multi-node Wasserstein-DRO (UB/LB extremes) under the SAME "no network + nodal balance + pool exchange"
    modeling assumption as IEEE14_Reserve_SO_Manager_MultiNode.

    Supports inputs in (11,T) or (14,T) for forecast / hourly bounds / fixed omega bounds,
    and (N,11,T) or (N,14,T) for scenarios.
    """

    def __init__(self, args):
        self.args = args

        # --- 11负荷列 -> 14母线号 的映射（与 SO MultiNode 一致）---
        self.column_mapping = {
            "4-2-1": 3, "4-2-3": 4, "4-2-8": 9, "4-2-2": 2, "4-2-5": 14,
            "4-2-6": 13, "4-2-9": 6, "4-2-4": 10, "4-2-0": 5,
            "4-2-7": 12, "4-2-10": 11
        }
        self.default_bus_order = list(self.column_mapping.values())  # 11 -> bus ids

        self.all_buses = list(range(1, 15))
        self.ref_bus = 1

        # generators (at bus id = gen id)
        gen_raw = {
            1: {"a": 0.04, "b": 20, "c": 0, "min": 0, "max": 332},
            2: {"a": 0.25, "b": 20, "c": 0, "min": 0, "max": 140},
            3: {"a": 0.01, "b": 40, "c": 0, "min": 0, "max": 100},
            6: {"a": 0.01, "b": 40, "c": 0, "min": 0, "max": 100},
            8: {"a": 0.01, "b": 40, "c": 0, "min": 0, "max": 100},
        }
        self.gen_info = {}
        cap_scale = float(getattr(args, "capacity_scale", 1.0))
        for g_id, info in gen_raw.items():
            self.gen_info[g_id] = info.copy()
            self.gen_info[g_id]["max"] = info["max"] * cap_scale

        self.gen_bus_list = list(self.gen_info.keys())
        self.gen_bus_of = {g: g for g in self.gen_bus_list}

        self.ramp_factor = float(getattr(args, "ramp_factor", 1.0))

        self.model = None
        self.vars = {}
        self.T = None
        self.N_scen = None

        # stored
        self._forecast14 = None
        self._scen14 = None
        self._Omega14 = None
        self._Omega_min = None
        self._Omega_max = None
        self._dist_max = None
        self._dist_min = None
        self._dro_eps = None
        self._dro_ub_lb_scale = None

    # ----------------- mapping helpers -----------------
    def map_11load_to_14bus(self, arr_11_T):
        arr_11_T = np.asarray(arr_11_T, dtype=float)
        if arr_11_T.ndim != 2 or arr_11_T.shape[0] != len(self.default_bus_order):
            raise ValueError(f"arr_11_T must be (11,T), got {arr_11_T.shape}")
        T = arr_11_T.shape[1]
        out = np.zeros((14, T), dtype=float)
        for j, bus in enumerate(self.default_bus_order):
            out[bus - 1, :] = arr_11_T[j, :]
        return out

    def map_scenarios_to_14bus(self, scen_N_11_T):
        scen_N_11_T = np.asarray(scen_N_11_T, dtype=float)
        if scen_N_11_T.ndim != 3 or scen_N_11_T.shape[1] != len(self.default_bus_order):
            raise ValueError(f"scen_N_11_T must be (N,11,T), got {scen_N_11_T.shape}")
        N, _, T = scen_N_11_T.shape
        out = np.zeros((N, 14, T), dtype=float)
        for j, bus in enumerate(self.default_bus_order):
            out[:, bus - 1, :] = scen_N_11_T[:, j, :]
        return out

    def _coerce_forecast14(self, forecast):
        forecast = np.asarray(forecast, dtype=float)
        if forecast.ndim != 2:
            raise ValueError(f"forecast must be 2D, got {forecast.shape}")
        if forecast.shape[0] == 14:
            return forecast
        if forecast.shape[0] == len(self.default_bus_order):  # 11
            return self.map_11load_to_14bus(forecast)
        raise ValueError(f"forecast must be (11,T) or (14,T), got {forecast.shape}")

    def _coerce_scen14(self, scen):
        scen = np.asarray(scen, dtype=float)
        if scen.ndim != 3:
            raise ValueError(f"scenarios must be 3D, got {scen.shape}")
        if scen.shape[1] == 14:
            return scen
        if scen.shape[1] == len(self.default_bus_order):  # 11
            return self.map_scenarios_to_14bus(scen)
        raise ValueError(f"scenarios must be (N,11,T) or (N,14,T), got {scen.shape}")

    # ----------------- build model -----------------
    def build_model(
        self,
        forecast,
        scenarios,
        eps,
        ub_lb_scale=1.0,
        fixed_om_min=None,    # (11,T) or (14,T)
        fixed_om_max=None,    # (11,T) or (14,T)
        hourly_load_min=None, # (11,T) or (14,T)
        hourly_load_max=None, # (11,T) or (14,T)
        output_flag=0,
    ):
        forecast14 = self._coerce_forecast14(forecast)   # (14,T)
        scen14 = self._coerce_scen14(scenarios)          # (N,14,T)

        N, _, T = scen14.shape
        if forecast14.shape[1] != T:
            raise ValueError("T mismatch between forecast and scenarios")

        self.T = int(T)
        self.N_scen = int(N)

        Omega14 = scen14 - forecast14[None, :, :]        # (N,14,T)

        # -------- choose omega_min/max (TIME-VARYING, bus-wise) --------
        # Priority:
        #   1) hourly_load_min/max -> omega = load_bound - forecast
        #      BUT: if hourly implied bounds are tighter than sample extrema,
        #      use sample extrema instead (optnet-style: final bounds cover BOTH).
        #   2) fixed_om_min/max    -> omega bounds directly
        #   3) from samples min/max scaled
        if hourly_load_min is not None or hourly_load_max is not None:
            if hourly_load_min is None or hourly_load_max is None:
                raise ValueError("Pass both hourly_load_min and hourly_load_max.")
            Lmin14 = self._coerce_forecast14(hourly_load_min)
            Lmax14 = self._coerce_forecast14(hourly_load_max)
            if Lmin14.shape != forecast14.shape or Lmax14.shape != forecast14.shape:
                raise ValueError("hourly_load_min/max shape must match forecast shape after coercion.")

            # hourly implied omega bounds
            om_min_h = Lmin14 - forecast14
            om_max_h = Lmax14 - forecast14
            om_min_h, om_max_h = np.minimum(om_min_h, om_max_h), np.maximum(om_min_h, om_max_h)

            # sample implied omega bounds
            om_min_s = Omega14.min(axis=0)
            om_max_s = Omega14.max(axis=0)

            # final omega bounds cover both hourly bounds and sample extremes
            om_min = np.minimum(om_min_h, om_min_s)
            om_max = np.maximum(om_max_h, om_max_s)

        elif fixed_om_min is not None or fixed_om_max is not None:
            if fixed_om_min is None or fixed_om_max is None:
                raise ValueError("Pass both fixed_om_min and fixed_om_max.")
            # allow (11,T) or (14,T)
            om_min = self._coerce_forecast14(fixed_om_min)
            om_max = self._coerce_forecast14(fixed_om_max)
            if om_min.shape != forecast14.shape or om_max.shape != forecast14.shape:
                raise ValueError("fixed_om_min/max must match forecast shape after coercion.")
            om_min, om_max = np.minimum(om_min, om_max), np.maximum(om_min, om_max)

        else:
            om_max = Omega14.max(axis=0) * float(ub_lb_scale)  # (14,T)
            om_min = Omega14.min(axis=0) * float(ub_lb_scale)  # (14,T)

        # --------- PRECOMPUTE L1 DISTANCES (CONSTANTS) ----------
        dist_max = np.sum(np.abs(om_max[None, :, :] - Omega14), axis=(1, 2))  # (N,)
        dist_min = np.sum(np.abs(om_min[None, :, :] - Omega14), axis=(1, 2))  # (N,)

        m = gp.Model("IEEE14_MultiNode_Reserve_DRO_NoNetwork_UBLB")
        m.setParam("OutputFlag", int(output_flag))

        # ---------- Stage-1 ----------
        P_DA = m.addVars(self.gen_bus_list, range(T), lb=0, name="P_DA")
        R_up = m.addVars(self.gen_bus_list, range(T), lb=0, name="R_up")
        R_dn = m.addVars(self.gen_bus_list, range(T), lb=0, name="R_dn")
        Cost_DA = m.addVars(self.gen_bus_list, range(T), lb=0, name="Cost_DA")
        S_pool = m.addVars(self.all_buses, range(T), lb=-GRB.INFINITY, name="S_pool")

        # ---------- Stage-2 (per scenario) ----------
        d_up = m.addVars(self.gen_bus_list, range(N), range(T), lb=0, name="d_up")
        d_dn = m.addVars(self.gen_bus_list, range(N), range(T), lb=0, name="d_dn")
        LS = m.addVars(range(N), self.all_buses, range(T), lb=0, name="LS")
        SP = m.addVars(range(N), self.all_buses, range(T), lb=0, name="SP")
        S_pool_s = m.addVars(range(N), self.all_buses, range(T), lb=-GRB.INFINITY, name="S_pool_s")

        # ---------- Stage-2 extremes (max/min) ----------
        du_max = m.addVars(self.gen_bus_list, range(T), lb=0, name="du_max")
        dd_max = m.addVars(self.gen_bus_list, range(T), lb=0, name="dd_max")
        LS_max = m.addVars(self.all_buses, range(T), lb=0, name="LS_max")
        SP_max = m.addVars(self.all_buses, range(T), lb=0, name="SP_max")
        S_pool_max = m.addVars(self.all_buses, range(T), lb=-GRB.INFINITY, name="S_pool_max")

        du_min = m.addVars(self.gen_bus_list, range(T), lb=0, name="du_min")
        dd_min = m.addVars(self.gen_bus_list, range(T), lb=0, name="dd_min")
        LS_min = m.addVars(self.all_buses, range(T), lb=0, name="LS_min")
        SP_min = m.addVars(self.all_buses, range(T), lb=0, name="SP_min")
        S_pool_min = m.addVars(self.all_buses, range(T), lb=-GRB.INFINITY, name="S_pool_min")

        # ---------- DRO ----------
        phi = m.addVars(range(N), lb=-GRB.INFINITY, name="phi")
        gamma = m.addVar(lb=0, name="gamma")

        # ---------- parameters/prices ----------
        voll = float(getattr(self.args, "voll", 1000.0))
        vosp = float(getattr(self.args, "vosp", 50.0))
        reserve_up_ratio = float(getattr(self.args, "reserve_up_ratio", 0.05))
        reserve_dn_ratio = float(getattr(self.args, "reserve_dn_ratio", 0.02))
        rt_up_ratio = float(getattr(self.args, "rt_up_ratio", 2.0))
        rt_dn_ratio = float(getattr(self.args, "rt_dn_ratio", 0.5))
        ramp_factor = float(self.ramp_factor)

        b_g = {g: float(self.gen_info[g]["b"]) for g in self.gen_bus_list}
        c_g = {g: float(self.gen_info[g]["c"]) for g in self.gen_bus_list}

        price_res_up = {g: reserve_up_ratio * b_g[g] for g in self.gen_bus_list}
        price_res_dn = {g: reserve_dn_ratio * b_g[g] for g in self.gen_bus_list}
        price_rt_up = {g: rt_up_ratio * b_g[g] for g in self.gen_bus_list}
        price_rt_dn = {g: rt_dn_ratio * b_g[g] for g in self.gen_bus_list}

        # ---------- objective ----------
        obj_stage1_energy = gp.quicksum(Cost_DA[g, t] for g in self.gen_bus_list for t in range(T))
        obj_stage1_reserve = gp.quicksum(
            price_res_up[g] * R_up[g, t] + price_res_dn[g] * R_dn[g, t]
            for g in self.gen_bus_list for t in range(T)
        )
        obj_dro = (1.0 / N) * gp.quicksum(phi[n] for n in range(N)) + float(eps) * gamma
        m.setObjective(obj_stage1_energy + obj_stage1_reserve + obj_dro, GRB.MINIMIZE)

        # ---------- helper: recourse costs ----------
        def recourse_cost_sample(n):
            deploy = gp.quicksum(
                price_rt_up[g] * d_up[g, n, t] + price_rt_dn[g] * d_dn[g, n, t]
                for g in self.gen_bus_list for t in range(T)
            )
            penalty = gp.quicksum(
                voll * LS[n, b, t] + vosp * SP[n, b, t]
                for b in self.all_buses for t in range(T)
            )
            return deploy + penalty

        cost_max = (
            gp.quicksum(price_rt_up[g] * du_max[g, t] + price_rt_dn[g] * dd_max[g, t]
                        for g in self.gen_bus_list for t in range(T))
            + gp.quicksum(voll * LS_max[b, t] + vosp * SP_max[b, t]
                          for b in self.all_buses for t in range(T))
        )
        cost_min = (
            gp.quicksum(price_rt_up[g] * du_min[g, t] + price_rt_dn[g] * dd_min[g, t]
                        for g in self.gen_bus_list for t in range(T))
            + gp.quicksum(voll * LS_min[b, t] + vosp * SP_min[b, t]
                          for b in self.all_buses for t in range(T))
        )

        # ---------- constraints ----------
        for g in self.gen_bus_list:
            pmin, pmax = float(self.gen_info[g]["min"]), float(self.gen_info[g]["max"])
            b_lin = float(b_g[g])
            c0 = float(c_g[g])

            for t in range(T):
                m.addConstr(P_DA[g, t] >= pmin)
                m.addConstr(P_DA[g, t] <= pmax)

                m.addConstr(P_DA[g, t] + R_up[g, t] <= pmax, name=f"cap_up_{g}_{t}")
                m.addConstr(P_DA[g, t] - R_dn[g, t] >= pmin, name=f"cap_dn_{g}_{t}")

                m.addConstr(Cost_DA[g, t] == b_lin * P_DA[g, t] + c0, name=f"lin_cost_{g}_{t}")

            for t in range(1, T):
                m.addConstr(P_DA[g, t] - P_DA[g, t - 1] <= pmax * ramp_factor)
                m.addConstr(P_DA[g, t - 1] - P_DA[g, t] <= pmax * ramp_factor)

            for n in range(N):
                for t in range(T):
                    m.addConstr(d_up[g, n, t] <= R_up[g, t])
                    m.addConstr(d_dn[g, n, t] <= R_dn[g, t])

            for t in range(T):
                m.addConstr(du_max[g, t] <= R_up[g, t])
                m.addConstr(dd_max[g, t] <= R_dn[g, t])
                m.addConstr(du_min[g, t] <= R_up[g, t])
                m.addConstr(dd_min[g, t] <= R_dn[g, t])

        # ---- DA nodal balance with pool ----
        for t in range(T):
            m.addConstr(gp.quicksum(S_pool[b, t] for b in self.all_buses) == 0.0, name=f"pool_DA_{t}")
            for b in self.all_buses:
                gen_inj = gp.quicksum(P_DA[g, t] for g in self.gen_bus_list if self.gen_bus_of[g] == b)
                load = float(forecast14[b - 1, t])
                m.addConstr(gen_inj - load == S_pool[b, t], name=f"nodalBal_DA_{b}_{t}")

        # ---- RT nodal balance: each scenario ----
        for n in range(N):
            for t in range(T):
                m.addConstr(gp.quicksum(S_pool_s[n, b, t] for b in self.all_buses) == 0.0,
                            name=f"pool_RT_{n}_{t}")
                for b in self.all_buses:
                    gen_inj_rt = gp.quicksum(
                        (P_DA[g, t] + d_up[g, n, t] - d_dn[g, n, t])
                        for g in self.gen_bus_list if self.gen_bus_of[g] == b
                    )
                    load_rt = float(scen14[n, b - 1, t])
                    m.addConstr(gen_inj_rt - SP[n, b, t] - (load_rt - LS[n, b, t]) == S_pool_s[n, b, t],
                                name=f"nodalBal_RT_{n}_{b}_{t}")

        # ---- RT nodal balance: extremes (forecast + omega_ext) ----
        for t in range(T):
            m.addConstr(gp.quicksum(S_pool_max[b, t] for b in self.all_buses) == 0.0, name=f"pool_MAX_{t}")
            m.addConstr(gp.quicksum(S_pool_min[b, t] for b in self.all_buses) == 0.0, name=f"pool_MIN_{t}")

            for b in self.all_buses:
                gen_rt_max = gp.quicksum(
                    (P_DA[g, t] + du_max[g, t] - dd_max[g, t])
                    for g in self.gen_bus_list if self.gen_bus_of[g] == b
                )
                load_max = float(forecast14[b - 1, t] + om_max[b - 1, t])
                m.addConstr(gen_rt_max - SP_max[b, t] - (load_max - LS_max[b, t]) == S_pool_max[b, t],
                            name=f"nodalMax_{b}_{t}")

                gen_rt_min = gp.quicksum(
                    (P_DA[g, t] + du_min[g, t] - dd_min[g, t])
                    for g in self.gen_bus_list if self.gen_bus_of[g] == b
                )
                load_min = float(forecast14[b - 1, t] + om_min[b - 1, t])
                m.addConstr(gen_rt_min - SP_min[b, t] - (load_min - LS_min[b, t]) == S_pool_min[b, t],
                            name=f"nodalMin_{b}_{t}")

        # ---- DRO constraints ----
        for n in range(N):
            dmax_n = float(dist_max[n])
            dmin_n = float(dist_min[n])
            m.addConstr(phi[n] >= recourse_cost_sample(n), name=f"phi_sample_{n}")
            m.addConstr(phi[n] >= cost_max - gamma * dmax_n, name=f"phi_ub_{n}")
            m.addConstr(phi[n] >= cost_min - gamma * dmin_n, name=f"phi_lb_{n}")

        m.update()

        self.model = m
        self.vars = {
            "P_DA": P_DA,
            "R_up": R_up,
            "R_dn": R_dn,
            "Cost_DA": Cost_DA,
            "S_pool": S_pool,

            "d_up": d_up,
            "d_dn": d_dn,
            "LS": LS,
            "SP": SP,
            "S_pool_s": S_pool_s,

            "du_max": du_max,
            "dd_max": dd_max,
            "LS_max": LS_max,
            "SP_max": SP_max,
            "S_pool_max": S_pool_max,

            "du_min": du_min,
            "dd_min": dd_min,
            "LS_min": LS_min,
            "SP_min": SP_min,
            "S_pool_min": S_pool_min,

            "phi": phi,
            "gamma": gamma,
        }

        self._forecast14 = forecast14.copy()
        self._scen14 = scen14.copy()
        self._Omega14 = Omega14.copy()
        self._Omega_min = om_min.copy()
        self._Omega_max = om_max.copy()
        self._dist_max = dist_max.copy()
        self._dist_min = dist_min.copy()
        self._dro_eps = float(eps)
        self._dro_ub_lb_scale = float(ub_lb_scale)

    def solve(self, warm_start=False, threads=None, method=None, output_flag=None):
        if self.model is None:
            raise RuntimeError("Call build_model() first.")
        if output_flag is not None:
            self.model.setParam("OutputFlag", int(output_flag))
        if warm_start:
            self.model.setParam("LPWarmStart", 1)
        if threads is not None:
            self.model.setParam("Threads", int(threads))
        if method is not None:
            self.model.setParam("Method", int(method))

        self.model.optimize()
        if self.model.status != GRB.OPTIMAL:
            raise RuntimeError(f"Optimization not optimal. status={self.model.status}")

    def get_stage1_results(self):
        if self.model is None or self.model.status != GRB.OPTIMAL:
            return None

        T = self.T
        G_list = self.gen_bus_list
        P = np.zeros((T, len(G_list)))
        Rup = np.zeros((T, len(G_list)))
        Rdn = np.zeros((T, len(G_list)))

        for i, g in enumerate(G_list):
            for t in range(T):
                P[t, i] = float(self.vars["P_DA"][g, t].X)
                Rup[t, i] = float(self.vars["R_up"][g, t].X)
                Rdn[t, i] = float(self.vars["R_dn"][g, t].X)

        return (
            pd.DataFrame(P, columns=[f"Gen_{g}" for g in G_list]),
            pd.DataFrame(Rup, columns=[f"Rup_{g}" for g in G_list]),
            pd.DataFrame(Rdn, columns=[f"Rdn_{g}" for g in G_list]),
            {
                "Objective": float(self.model.ObjVal),
                "gamma": float(self.vars["gamma"].X),
                "eps": float(self._dro_eps) if self._dro_eps is not None else None,
                "ub_lb_scale": float(self._dro_ub_lb_scale) if self._dro_ub_lb_scale is not None else None,
            },
        )

    def compute_true_cost(self, actual, mu=None, output_flag=0, warm_start=False, threads=None, method=None):
        if self.model is None or self.model.status != GRB.OPTIMAL:
            raise RuntimeError("Model not solved to OPTIMAL. Call solve() first.")

        if mu is None:
            if self._forecast14 is None:
                raise RuntimeError("No stored forecast found. Pass mu explicitly.")
            forecast14 = np.asarray(self._forecast14, dtype=float)
        else:
            forecast14 = self._coerce_forecast14(mu)

        actual14 = self._coerce_forecast14(actual)
        if actual14.shape != forecast14.shape:
            raise ValueError(f"actual shape {actual14.shape} != forecast shape {forecast14.shape}")

        T = forecast14.shape[1]
        G_list = self.gen_bus_list
        Buses = self.all_buses

        reserve_dn_ratio = float(self.args.reserve_dn_ratio)
        reserve_up_ratio = float(self.args.reserve_up_ratio)
        rt_dn_ratio = float(self.args.rt_dn_ratio)
        rt_up_ratio = float(self.args.rt_up_ratio)
        voll = float(self.args.voll)
        vosp = float(self.args.vosp)

        b_g = {g: float(self.gen_info[g]["b"]) for g in G_list}
        price_res_up = {g: reserve_up_ratio * b_g[g] for g in G_list}
        price_res_dn = {g: reserve_dn_ratio * b_g[g] for g in G_list}
        price_rt_up = {g: rt_up_ratio * b_g[g] for g in G_list}
        price_rt_dn = {g: rt_dn_ratio * b_g[g] for g in G_list}

        P_DA = np.array([[self.vars["P_DA"][g, t].X for g in G_list] for t in range(T)], dtype=float)
        R_up = np.array([[self.vars["R_up"][g, t].X for g in G_list] for t in range(T)], dtype=float)
        R_dn = np.array([[self.vars["R_dn"][g, t].X for g in G_list] for t in range(T)], dtype=float)
        Cost_DA = np.array([[self.vars["Cost_DA"][g, t].X for g in G_list] for t in range(T)], dtype=float)

        stage1_energy_cost = float(Cost_DA.sum())
        stage1_reserve_cost = 0.0
        for gi, g in enumerate(G_list):
            stage1_reserve_cost += float(np.sum(price_res_up[g] * R_up[:, gi] + price_res_dn[g] * R_dn[:, gi]))
        stage1_reserve_cost = float(stage1_reserve_cost)

        Omega14 = actual14 - forecast14  # (14,T)

        gen_to_idx = {g: i for i, g in enumerate(G_list)}

        sm = gp.Model("RT_NoNetwork_AllT_eval_DRO")
        sm.setParam("OutputFlag", int(output_flag))
        if warm_start:
            sm.setParam("LPWarmStart", 1)
        if threads is not None:
            sm.setParam("Threads", int(threads))
        if method is not None:
            sm.setParam("Method", int(method))

        du = sm.addVars(G_list, range(T), lb=0, name="du")
        dd = sm.addVars(G_list, range(T), lb=0, name="dd")
        ls = sm.addVars(Buses, range(T), lb=0, name="ls")
        sp = sm.addVars(Buses, range(T), lb=0, name="sp")
        s_pool = sm.addVars(Buses, range(T), lb=-GRB.INFINITY, name="s_pool")

        for t in range(T):
            for g in G_list:
                gi = gen_to_idx[g]
                sm.addConstr(du[g, t] <= float(R_up[t, gi]), name=f"du_le_Rup_{g}_{t}")
                sm.addConstr(dd[g, t] <= float(R_dn[t, gi]), name=f"dd_le_Rdn_{g}_{t}")

            sm.addConstr(gp.quicksum(s_pool[b, t] for b in Buses) == 0.0, name=f"pool_{t}")

            for b in Buses:
                gen_inj = gp.quicksum(
                    float(P_DA[t, gen_to_idx[g]]) + du[g, t] - dd[g, t]
                    for g in G_list if self.gen_bus_of[g] == b
                )
                load = float(forecast14[b - 1, t] + Omega14[b - 1, t])
                sm.addConstr(gen_inj - sp[b, t] - (load - ls[b, t]) == s_pool[b, t],
                             name=f"nodal_{b}_{t}")

        sm.setObjective(
            gp.quicksum(price_rt_up[g] * du[g, t] + price_rt_dn[g] * dd[g, t]
                        for g in G_list for t in range(T))
            + voll * gp.quicksum(ls[b, t] for b in Buses for t in range(T))
            + vosp * gp.quicksum(sp[b, t] for b in Buses for t in range(T)),
            GRB.MINIMIZE
        )

        sm.optimize()
        if sm.status != GRB.OPTIMAL:
            raise RuntimeError(f"RT all-T subproblem not optimal, status={sm.status}")

        rt_deploy_cost = float(sum(price_rt_up[g] * du[g, t].X + price_rt_dn[g] * dd[g, t].X
                                   for g in G_list for t in range(T)))
        rt_penalty_cost = float(
            voll * sum(ls[b, t].X for b in Buses for t in range(T))
            + vosp * sum(sp[b, t].X for b in Buses for t in range(T))
        )
        sm.dispose()

        total_cost = stage1_energy_cost + stage1_reserve_cost + rt_deploy_cost + rt_penalty_cost

        return {
            "stage1_energy_cost": float(stage1_energy_cost),
            "stage1_reserve_cost": float(stage1_reserve_cost),
            "rt_deploy_cost": float(rt_deploy_cost),
            "rt_penalty_cost": float(rt_penalty_cost),
            "total_cost": float(total_cost),
            "Omega14": Omega14,
            "P_DA": P_DA,
            "R_up": R_up,
            "R_dn": R_dn,
            "Cost_DA": Cost_DA,
        }


def _solve_one_day_multi(day, args, data, pred_mean_all, mode=None):
    T = args.T
    sl = slice(T*day, T*day+T)

    scenarios = data["Y_pred"][0:args.N_scen, :, sl]#.sum(axis=1)          # (S,T)
    forecast_load = pred_mean_all[:, sl]#.sum(axis=0)          # (T,)
    Y_true = data["Y_true"][:, sl]#.sum(axis=0)                # (T,)

    if mode == "ideal":
        forecast_load = Y_true.copy()
        scenarios = Y_true.reshape(-1,11,T)

    mgr = IEEE14_Reserve_SO_Manager_MultiNode(args)
    mgr.build_model(forecast_load, scenarios, output_flag=0)

    mgr.solve(threads=1, method=1, warm_start=False, output_flag=0)
    true_eval = mgr.compute_true_cost(Y_true, output_flag=0, threads=1, method=1)
    return true_eval["total_cost"]

def Average_cost_Reserve_SO_MultiNode(args, data, mode=None, n_jobs=None):
    T = args.T
    Y_pred = data["Y_pred"]
    S, N, L = Y_pred.shape
    days = L // T

    pred_mean_all = Y_pred.mean(axis=0)  # (N,L)

    if n_jobs is None:
        n_jobs = min(os.cpu_count() or 1, days)

    costs = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_solve_one_day_multi)(day, args, data, pred_mean_all, mode)
        for day in range(days)
    )
    return costs


def build_dc_matrices_torch(mgr, dtype=torch.float64, device="cpu"):
    """
    Build DC power flow matrices for CVXPY/CvxpyLayer usage.

    Returns dict with:
      nB: number of buses
      nE: number of directed edges (= 2 * #undirected lines)
      A:  (nB, nE) node-arc incidence for directed edges
      Bvec: (nE, ) susceptance per directed edge
      E_dir: list of directed edges [(u,v), ...] length nE
      bus_to_i: mapping {bus_id: row_index}
    """
    buses = list(mgr.all_buses)
    nB = len(buses)
    bus_to_i = {int(b): i for i, b in enumerate(buses)}

    # directed edges: for each undirected (u,v), create (u,v) and (v,u)
    E_dir = []
    for (u, v) in mgr.lines_undirected:
        u = int(u); v = int(v)
        E_dir.append((u, v))
        E_dir.append((v, u))
    nE = len(E_dir)

    A = np.zeros((nB, nE), dtype=float)
    Bvec = np.zeros((nE,), dtype=float)

    for e, (u, v) in enumerate(E_dir):
        ui = bus_to_i[u]
        vi = bus_to_i[v]
        # incidence for directed edge u->v
        A[ui, e] = 1.0
        A[vi, e] = -1.0
        # susceptance for this (u,v)
        Bvec[e] = float(mgr.b_susceptance[(u, v)])

    out = {
        "nB": nB,
        "nE": nE,
        "A": torch.tensor(A, dtype=dtype, device=device),
        "Bvec": torch.tensor(Bvec, dtype=dtype, device=device),
        "E_dir": E_dir,
        "bus_to_i": bus_to_i,
    }
    return out

class MultiNode_Reserve_Deterministic_DA_OptNet(torch.nn.Module):
    """
    Multi-bus deterministic day-ahead reserve OptNet.

    Input:
        forecast14: [B, nB, T]

    Output:
        P_DA:    [B, G, T]
        R_up:    [B, G, T]
        R_dn:    [B, G, T]
        Cost_DA: [B, G, T] (optional)
        obj:     [B]
    """
    def __init__(self, mgr, T=24, dtype=torch.float64):
        super().__init__()
        self.T = int(T)
        self.dtype = dtype

        self.gen_ids = list(mgr.gen_bus_list)
        self.gen_info = mgr.gen_info
        self.G = len(self.gen_ids)

        self.nB = 14
        bus_to_i = {b: b - 1 for b in range(1, 15)}

        reserve_up_ratio = float(getattr(mgr.args, "reserve_up_ratio", 0.05))
        reserve_dn_ratio = float(getattr(mgr.args, "reserve_dn_ratio", 0.02))
        reserve_req_up_ratio = float(getattr(mgr.args, "system_reserve_up_ratio", 0.02))
        reserve_req_dn_ratio = float(getattr(mgr.args, "system_reserve_dn_ratio", 0.02))
        ramp_factor = float(getattr(mgr.args, "ramp_rate", 1.0))

        pmin = np.array([self.gen_info[g]["min"] for g in self.gen_ids], dtype=float)
        pmax = np.array([self.gen_info[g]["max"] for g in self.gen_ids], dtype=float)
        b = np.array([self.gen_info[g]["b"] for g in self.gen_ids], dtype=float)
        c = np.array([self.gen_info[g]["c"] for g in self.gen_ids], dtype=float)
        ramp = pmax * ramp_factor

        self.register_buffer("b_G", torch.tensor(b, dtype=self.dtype))
        self.register_buffer("c_G", torch.tensor(c, dtype=self.dtype))
        self.register_buffer("res_up_ratio", torch.tensor(reserve_up_ratio, dtype=self.dtype))
        self.register_buffer("res_dn_ratio", torch.tensor(reserve_dn_ratio, dtype=self.dtype))
        self.register_buffer("req_up_ratio", torch.tensor(reserve_req_up_ratio, dtype=self.dtype))
        self.register_buffer("req_dn_ratio", torch.tensor(reserve_req_dn_ratio, dtype=self.dtype))

        # generator-to-bus incidence C: (nB,G)
        C = np.zeros((self.nB, self.G), dtype=float)
        for gi, gbus in enumerate(self.gen_ids):
            C[bus_to_i[int(gbus)], gi] = 1.0

        # ---------- CVXPY params ----------
        forecast = cp.Parameter((self.nB, self.T), nonneg=True, name="forecast14")

        # ---------- Variables ----------
        P = cp.Variable((self.G, self.T), nonneg=True, name="P_DA")
        R_up = cp.Variable((self.G, self.T), nonneg=True, name="R_up")
        R_dn = cp.Variable((self.G, self.T), nonneg=True, name="R_dn")
        Cost = cp.Variable((self.G, self.T), nonneg=True, name="Cost_DA")
        s_pool = cp.Variable((self.nB, self.T), name="s_pool_DA")

        cons = []
        cons += [P >= pmin.reshape(-1, 1)]
        cons += [P <= pmax.reshape(-1, 1)]
        cons += [P + R_up <= pmax.reshape(-1, 1)]
        cons += [P - R_dn >= pmin.reshape(-1, 1)]

        # linear DA cost
        cons += [Cost == cp.multiply(b.reshape(-1, 1), P) + c.reshape(-1, 1)]

        # DA nodal balance without network
        cons += [C @ P - forecast == s_pool]
        cons += [cp.sum(s_pool, axis=0) == 0]

        # system reserve requirement
        total_load = cp.sum(forecast, axis=0)
        cons += [cp.sum(R_up, axis=0) >= reserve_req_up_ratio * total_load]
        cons += [cp.sum(R_dn, axis=0) >= reserve_req_dn_ratio * total_load]

        # DA ramp constraints
        if self.T >= 2:
            cons += [P[:, 1:] - P[:, :-1] <= ramp.reshape(-1, 1)]
            cons += [P[:, :-1] - P[:, 1:] <= ramp.reshape(-1, 1)]

        res_up_price = reserve_up_ratio * b
        res_dn_price = reserve_dn_ratio * b

        obj = (
            cp.sum(Cost)
            + cp.sum(cp.multiply(res_up_price.reshape(-1, 1), R_up))
            + cp.sum(cp.multiply(res_dn_price.reshape(-1, 1), R_dn))
        )

        prob = cp.Problem(cp.Minimize(obj), cons)
        self.layer = CvxpyLayer(prob, parameters=[forecast], variables=[P, R_up, R_dn, Cost])

    def forward(self, forecast14, solver="ECOS", return_cost=True):
        orig_device = forecast14.device
        forecast_in = forecast14.to(device="cpu", dtype=self.dtype).contiguous()

        if solver == "ECOS":
            solver_args = {"solve_method": "ECOS"}
        elif solver == "SCS":
            solver_args = {"solve_method": "SCS", "max_iters": 200000, "eps": 1e-6}
        else:
            raise ValueError("solver must be 'ECOS' or 'SCS'")

        P, R_up, R_dn, Cost = self.layer(forecast_in, solver_args=solver_args)

        solve_device = P.device
        bG = self.b_G.to(device=solve_device, dtype=self.dtype)
        res_up_price = self.res_up_ratio.to(device=solve_device, dtype=self.dtype) * bG
        res_dn_price = self.res_dn_ratio.to(device=solve_device, dtype=self.dtype) * bG

        stage1_energy = Cost.sum(dim=(1, 2))
        stage1_reserve = (
            (res_up_price[None, :, None] * R_up).sum(dim=(1, 2))
            + (res_dn_price[None, :, None] * R_dn).sum(dim=(1, 2))
        )
        obj = stage1_energy + stage1_reserve

        outs = {
            "P_DA": P.to(orig_device),
            "R_up": R_up.to(orig_device),
            "R_dn": R_dn.to(orig_device),
            "obj": obj.to(orig_device),
        }
        if return_cost:
            outs["Cost_DA"] = Cost.to(orig_device)
        return outs

class MultiNode_Reserve_SAA_DA_OptNet(torch.nn.Module):
    """
    Multi-bus DA + SAA-RT OptNet.

    Input:
        forecast14: [B, nB, T]
        omega14:    [B, N, nB, T]   # scenario deviation

    Output:
        P_DA:    [B, G, T]
        R_up:    [B, G, T]
        R_dn:    [B, G, T]
        Cost_DA: [B, G, T] (optional)
        d_up:    [B, N, G, T] (optional)
        d_dn:    [B, N, G, T] (optional)
        LS:      [B, N, nB, T] (optional)
        SP:      [B, N, nB, T] (optional)
        obj:     [B]
    """
    def __init__(self, mgr, N_scen, T=24, dtype=torch.float64):
        super().__init__()
        self.N = int(N_scen)
        self.T = int(T)
        self.dtype = dtype

        self.gen_ids = list(mgr.gen_bus_list)
        self.gen_info = mgr.gen_info
        self.G = len(self.gen_ids)

        self.nB = 14
        bus_to_i = {b: b - 1 for b in range(1, 15)}

        voll = float(getattr(mgr.args, "voll", 1000.0))
        vosp = float(getattr(mgr.args, "vosp", 50.0))
        reserve_up_ratio = float(getattr(mgr.args, "reserve_up_ratio", 0.05))
        reserve_dn_ratio = float(getattr(mgr.args, "reserve_dn_ratio", 0.02))
        rt_up_ratio = float(getattr(mgr.args, "rt_up_ratio", 2.0))
        rt_dn_ratio = float(getattr(mgr.args, "rt_dn_ratio", 0.5))
        ramp_factor = float(getattr(mgr.args, "ramp_rate", 1.0))

        pmin = np.array([self.gen_info[g]["min"] for g in self.gen_ids], dtype=float)
        pmax = np.array([self.gen_info[g]["max"] for g in self.gen_ids], dtype=float)
        b = np.array([self.gen_info[g]["b"] for g in self.gen_ids], dtype=float)
        c = np.array([self.gen_info[g]["c"] for g in self.gen_ids], dtype=float)
        ramp = pmax * ramp_factor

        self.register_buffer("b_G", torch.tensor(b, dtype=self.dtype))
        self.register_buffer("voll", torch.tensor(voll, dtype=self.dtype))
        self.register_buffer("vosp", torch.tensor(vosp, dtype=self.dtype))
        self.register_buffer("res_up_ratio", torch.tensor(reserve_up_ratio, dtype=self.dtype))
        self.register_buffer("res_dn_ratio", torch.tensor(reserve_dn_ratio, dtype=self.dtype))
        self.register_buffer("rt_up_ratio", torch.tensor(rt_up_ratio, dtype=self.dtype))
        self.register_buffer("rt_dn_ratio", torch.tensor(rt_dn_ratio, dtype=self.dtype))

        # generator-to-bus incidence C: (nB,G)
        C = np.zeros((self.nB, self.G), dtype=float)
        for gi, gbus in enumerate(self.gen_ids):
            C[bus_to_i[int(gbus)], gi] = 1.0

        # ---------- CVXPY params ----------
        forecast = cp.Parameter((self.nB, self.T), nonneg=True, name="forecast14")
        omega_list = [cp.Parameter((self.nB, self.T), name=f"omega_{n}") for n in range(self.N)]

        # ---------- Stage-1 vars ----------
        P = cp.Variable((self.G, self.T), nonneg=True, name="P_DA")
        R_up = cp.Variable((self.G, self.T), nonneg=True, name="R_up")
        R_dn = cp.Variable((self.G, self.T), nonneg=True, name="R_dn")
        Cost = cp.Variable((self.G, self.T), nonneg=True, name="Cost_DA")
        s_pool = cp.Variable((self.nB, self.T), name="s_pool_DA")

        # ---------- Stage-2 vars ----------
        d_up_list = [cp.Variable((self.G, self.T), nonneg=True, name=f"d_up_{n}") for n in range(self.N)]
        d_dn_list = [cp.Variable((self.G, self.T), nonneg=True, name=f"d_dn_{n}") for n in range(self.N)]
        LS_list = [cp.Variable((self.nB, self.T), nonneg=True, name=f"LS_{n}") for n in range(self.N)]
        SP_list = [cp.Variable((self.nB, self.T), nonneg=True, name=f"SP_{n}") for n in range(self.N)]
        s_pool_s_list = [cp.Variable((self.nB, self.T), name=f"s_pool_RT_{n}") for n in range(self.N)]

        cons = []
        cons += [P >= pmin.reshape(-1, 1)]
        cons += [P <= pmax.reshape(-1, 1)]
        cons += [P + R_up <= pmax.reshape(-1, 1)]
        cons += [P - R_dn >= pmin.reshape(-1, 1)]

        # linear DA cost
        cons += [Cost == cp.multiply(b.reshape(-1, 1), P) + c.reshape(-1, 1)]

        # DA nodal balance
        cons += [C @ P - forecast == s_pool]
        cons += [cp.sum(s_pool, axis=0) == 0]

        # DA ramp constraints
        if self.T >= 2:
            cons += [P[:, 1:] - P[:, :-1] <= ramp.reshape(-1, 1)]
            cons += [P[:, :-1] - P[:, 1:] <= ramp.reshape(-1, 1)]

        # per-scenario RT constraints
        for n in range(self.N):
            d_up = d_up_list[n]
            d_dn = d_dn_list[n]
            LS = LS_list[n]
            SP = SP_list[n]
            sps = s_pool_s_list[n]

            cons += [d_up <= R_up]
            cons += [d_dn <= R_dn]
            cons += [
                C @ (P + d_up - d_dn) - (forecast + omega_list[n]) + LS - SP == sps
            ]
            cons += [cp.sum(sps, axis=0) == 0]

        res_up_price = reserve_up_ratio * b
        res_dn_price = reserve_dn_ratio * b
        rt_up_price = rt_up_ratio * b
        rt_dn_price = rt_dn_ratio * b

        obj_stage1 = cp.sum(Cost)
        obj_reserve = (
            cp.sum(cp.multiply(res_up_price.reshape(-1, 1), R_up))
            + cp.sum(cp.multiply(res_dn_price.reshape(-1, 1), R_dn))
        )

        obj_deploy = 0
        obj_pen = 0
        for n in range(self.N):
            obj_deploy += (
                cp.sum(cp.multiply(rt_up_price.reshape(-1, 1), d_up_list[n]))
                + cp.sum(cp.multiply(rt_dn_price.reshape(-1, 1), d_dn_list[n]))
            )
            obj_pen += voll * cp.sum(LS_list[n]) + vosp * cp.sum(SP_list[n])

        obj_stage2 = (obj_deploy + obj_pen) / float(self.N)

        prob = cp.Problem(cp.Minimize(obj_stage1 + obj_reserve + obj_stage2), cons)

        variables = (
            [P, R_up, R_dn, Cost]
            + d_up_list + d_dn_list + LS_list + SP_list
        )
        self.layer = CvxpyLayer(prob, parameters=[forecast] + omega_list, variables=variables)

        self._idx = {
            "P": 0,
            "R_up": 1,
            "R_dn": 2,
            "Cost": 3,
            "d_up_start": 4,
            "d_dn_start": 4 + self.N,
            "LS_start": 4 + 2 * self.N,
            "SP_start": 4 + 3 * self.N,
        }

    def forward(self, forecast14, omega14, solver="ECOS", return_rt=True, return_cost=True):
        orig_device = forecast14.device

        forecast_in = forecast14.to(device="cpu", dtype=self.dtype).contiguous()
        omega_in = omega14.to(device="cpu", dtype=self.dtype).contiguous()

        if solver == "ECOS":
            solver_args = {"solve_method": "ECOS"}
        elif solver == "SCS":
            solver_args = {"solve_method": "SCS", "max_iters": 20000, "eps": 1e-6}
        else:
            raise ValueError("solver must be 'ECOS' or 'SCS'")

        omega_split = list(torch.unbind(omega_in, dim=1))  # N x (B,nB,T)
        sol = self.layer(forecast_in, *omega_split, solver_args=solver_args)

        solve_device = sol[0].device
        P = sol[self._idx["P"]]
        R_up = sol[self._idx["R_up"]]
        R_dn = sol[self._idx["R_dn"]]
        Cost = sol[self._idx["Cost"]]

        d_up_list = sol[self._idx["d_up_start"]: self._idx["d_up_start"] + self.N]
        d_dn_list = sol[self._idx["d_dn_start"]: self._idx["d_dn_start"] + self.N]
        LS_list = sol[self._idx["LS_start"]: self._idx["LS_start"] + self.N]
        SP_list = sol[self._idx["SP_start"]: self._idx["SP_start"] + self.N]

        d_up = torch.stack(d_up_list, dim=1)  # (B,N,G,T)
        d_dn = torch.stack(d_dn_list, dim=1)
        LS = torch.stack(LS_list, dim=1)      # (B,N,nB,T)
        SP = torch.stack(SP_list, dim=1)

        bG = self.b_G.to(device=solve_device, dtype=self.dtype)
        voll = self.voll.to(device=solve_device, dtype=self.dtype)
        vosp = self.vosp.to(device=solve_device, dtype=self.dtype)
        res_up_ratio = self.res_up_ratio.to(device=solve_device, dtype=self.dtype)
        res_dn_ratio = self.res_dn_ratio.to(device=solve_device, dtype=self.dtype)
        rt_up_ratio = self.rt_up_ratio.to(device=solve_device, dtype=self.dtype)
        rt_dn_ratio = self.rt_dn_ratio.to(device=solve_device, dtype=self.dtype)

        stage1_energy = Cost.sum(dim=(1, 2))
        res_cost = (
            ((res_up_ratio * bG)[None, :, None] * R_up)
            + ((res_dn_ratio * bG)[None, :, None] * R_dn)
        ).sum(dim=(1, 2))

        rt_deploy = (
            ((rt_up_ratio * bG)[None, None, :, None] * d_up)
            + ((rt_dn_ratio * bG)[None, None, :, None] * d_dn)
        ).sum(dim=(2, 3)).sum(dim=1)

        rt_pen = (
            voll * LS.sum(dim=(2, 3))
            + vosp * SP.sum(dim=(2, 3))
        ).sum(dim=1)

        obj = stage1_energy + res_cost + (rt_deploy + rt_pen) / float(self.N)

        outs = {
            "P_DA": P.to(orig_device),
            "R_up": R_up.to(orig_device),
            "R_dn": R_dn.to(orig_device),
            "obj": obj.to(orig_device),
        }
        if return_cost:
            outs["Cost_DA"] = Cost.to(orig_device)
        if return_rt:
            outs["d_up"] = d_up.to(orig_device)
            outs["d_dn"] = d_dn.to(orig_device)
            outs["LS"] = LS.to(orig_device)
            outs["SP"] = SP.to(orig_device)
        return outs



class MultiNode_Reserve_DRO_DA_OptNet(torch.nn.Module):
    """
    Multi-bus DRO DA OptNet.

    Notes:
    - linear distance terms, no abs()
    - no omega envelope constraints except [om_min, om_max] extreme recourse construction
    - DA ramp constraints added
    """
    def __init__(self, mgr, N_scen, T=24, dtype=torch.float64):
        super().__init__()
        self.N = int(N_scen)
        self.T = int(T)
        self.dtype = dtype

        self.gen_ids = list(mgr.gen_bus_list)
        self.gen_info = mgr.gen_info
        self.G = len(self.gen_ids)

        self.nB = 14
        bus_to_i = {b: b - 1 for b in range(1, 15)}

        voll = float(getattr(mgr.args, "voll", 1000.0))
        vosp = float(getattr(mgr.args, "vosp", 50.0))
        reserve_up_ratio = float(getattr(mgr.args, "reserve_up_ratio", 0.05))
        reserve_dn_ratio = float(getattr(mgr.args, "reserve_dn_ratio", 0.02))
        rt_up_ratio = float(getattr(mgr.args, "rt_up_ratio", 2.0))
        rt_dn_ratio = float(getattr(mgr.args, "rt_dn_ratio", 0.5))
        ramp_factor = float(getattr(mgr.args, "ramp_rate", 1.0))

        pmin = np.array([self.gen_info[g]["min"] for g in self.gen_ids], dtype=float)
        pmax = np.array([self.gen_info[g]["max"] for g in self.gen_ids], dtype=float)
        b = np.array([self.gen_info[g]["b"] for g in self.gen_ids], dtype=float)
        c = np.array([self.gen_info[g]["c"] for g in self.gen_ids], dtype=float)
        ramp = pmax * ramp_factor

        self.register_buffer("b_G", torch.tensor(b, dtype=self.dtype))
        self.register_buffer("voll", torch.tensor(voll, dtype=self.dtype))
        self.register_buffer("vosp", torch.tensor(vosp, dtype=self.dtype))
        self.register_buffer("res_up_ratio", torch.tensor(reserve_up_ratio, dtype=self.dtype))
        self.register_buffer("res_dn_ratio", torch.tensor(reserve_dn_ratio, dtype=self.dtype))
        self.register_buffer("rt_up_ratio", torch.tensor(rt_up_ratio, dtype=self.dtype))
        self.register_buffer("rt_dn_ratio", torch.tensor(rt_dn_ratio, dtype=self.dtype))

        # generator-to-bus incidence C: (nB,G)
        C = np.zeros((self.nB, self.G), dtype=float)
        for gi, gbus in enumerate(self.gen_ids):
            C[bus_to_i[int(gbus)], gi] = 1.0

        # ---------- CVXPY params ----------
        forecast = cp.Parameter((self.nB, self.T), nonneg=True, name="forecast14")
        omega_list = [cp.Parameter((self.nB, self.T), name=f"omega_{n}") for n in range(self.N)]
        om_min = cp.Parameter((self.nB, self.T), name="om_min14")
        om_max = cp.Parameter((self.nB, self.T), name="om_max14")
        eps = cp.Parameter(nonneg=True, name="eps")

        # ---------- Stage-1 vars ----------
        P = cp.Variable((self.G, self.T), nonneg=True, name="P_DA")
        R_up = cp.Variable((self.G, self.T), nonneg=True, name="R_up")
        R_dn = cp.Variable((self.G, self.T), nonneg=True, name="R_dn")
        Cost = cp.Variable((self.G, self.T), nonneg=True, name="Cost_DA")
        s_pool = cp.Variable((self.nB, self.T), name="s_pool_DA")

        # ---------- Stage-2 vars ----------
        d_up_list = [cp.Variable((self.G, self.T), nonneg=True, name=f"d_up_{n}") for n in range(self.N)]
        d_dn_list = [cp.Variable((self.G, self.T), nonneg=True, name=f"d_dn_{n}") for n in range(self.N)]
        LS_list = [cp.Variable((self.nB, self.T), nonneg=True, name=f"LS_{n}") for n in range(self.N)]
        SP_list = [cp.Variable((self.nB, self.T), nonneg=True, name=f"SP_{n}") for n in range(self.N)]
        s_pool_s_list = [cp.Variable((self.nB, self.T), name=f"s_pool_RT_{n}") for n in range(self.N)]

        # ---------- extremes ----------
        du_max = cp.Variable((self.G, self.T), nonneg=True, name="du_max")
        dd_max = cp.Variable((self.G, self.T), nonneg=True, name="dd_max")
        LS_max = cp.Variable((self.nB, self.T), nonneg=True, name="LS_max")
        SP_max = cp.Variable((self.nB, self.T), nonneg=True, name="SP_max")
        s_pool_max = cp.Variable((self.nB, self.T), name="s_pool_MAX")

        du_min = cp.Variable((self.G, self.T), nonneg=True, name="du_min")
        dd_min = cp.Variable((self.G, self.T), nonneg=True, name="dd_min")
        LS_min = cp.Variable((self.nB, self.T), nonneg=True, name="LS_min")
        SP_min = cp.Variable((self.nB, self.T), nonneg=True, name="SP_min")
        s_pool_min = cp.Variable((self.nB, self.T), name="s_pool_MIN")

        # ---------- DRO vars ----------
        phi = cp.Variable((self.N,), name="phi")
        gamma = cp.Variable(nonneg=True, name="gamma")

        cons = []
        cons += [P >= pmin.reshape(-1, 1)]
        cons += [P <= pmax.reshape(-1, 1)]
        cons += [P + R_up <= pmax.reshape(-1, 1)]
        cons += [P - R_dn >= pmin.reshape(-1, 1)]

        cons += [Cost == cp.multiply(b.reshape(-1, 1), P) + c.reshape(-1, 1)]

        # DA balance
        cons += [C @ P - forecast == s_pool]
        cons += [cp.sum(s_pool, axis=0) == 0]

        # DA ramp constraints
        if self.T >= 2:
            cons += [P[:, 1:] - P[:, :-1] <= ramp.reshape(-1, 1)]
            cons += [P[:, :-1] - P[:, 1:] <= ramp.reshape(-1, 1)]

        # RT per scenario
        for n in range(self.N):
            cons += [d_up_list[n] <= R_up]
            cons += [d_dn_list[n] <= R_dn]
            cons += [
                C @ (P + d_up_list[n] - d_dn_list[n])
                - (forecast + omega_list[n])
                + LS_list[n] - SP_list[n]
                == s_pool_s_list[n]
            ]
            cons += [cp.sum(s_pool_s_list[n], axis=0) == 0]

        # extreme feasibility
        cons += [du_max <= R_up, dd_max <= R_dn]
        cons += [du_min <= R_up, dd_min <= R_dn]

        cons += [C @ (P + du_max - dd_max) - (forecast + om_max) + LS_max - SP_max == s_pool_max]
        cons += [cp.sum(s_pool_max, axis=0) == 0]

        cons += [C @ (P + du_min - dd_min) - (forecast + om_min) + LS_min - SP_min == s_pool_min]
        cons += [cp.sum(s_pool_min, axis=0) == 0]

        # costs
        res_up_price = reserve_up_ratio * b
        res_dn_price = reserve_dn_ratio * b
        rt_up_price = rt_up_ratio * b
        rt_dn_price = rt_dn_ratio * b

        stage1_energy = cp.sum(Cost)
        stage1_reserve = (
            cp.sum(cp.multiply(res_up_price.reshape(-1, 1), R_up))
            + cp.sum(cp.multiply(res_dn_price.reshape(-1, 1), R_dn))
        )

        Qn = []
        for n in range(self.N):
            Qn.append(
                cp.sum(cp.multiply(rt_up_price.reshape(-1, 1), d_up_list[n]))
                + cp.sum(cp.multiply(rt_dn_price.reshape(-1, 1), d_dn_list[n]))
                + voll * cp.sum(LS_list[n])
                + vosp * cp.sum(SP_list[n])
            )

        Q_max = (
            cp.sum(cp.multiply(rt_up_price.reshape(-1, 1), du_max))
            + cp.sum(cp.multiply(rt_dn_price.reshape(-1, 1), dd_max))
            + voll * cp.sum(LS_max)
            + vosp * cp.sum(SP_max)
        )
        Q_min = (
            cp.sum(cp.multiply(rt_up_price.reshape(-1, 1), du_min))
            + cp.sum(cp.multiply(rt_dn_price.reshape(-1, 1), dd_min))
            + voll * cp.sum(LS_min)
            + vosp * cp.sum(SP_min)
        )

        for n in range(self.N):
            dist_max_n = cp.sum(om_max - omega_list[n])
            dist_min_n = cp.sum(omega_list[n] - om_min)

            cons += [phi[n] >= Qn[n]]
            cons += [phi[n] >= Q_max - gamma * dist_max_n]
            cons += [phi[n] >= Q_min - gamma * dist_min_n]

        obj = stage1_energy + stage1_reserve + (1.0 / self.N) * cp.sum(phi) + eps * gamma
        prob = cp.Problem(cp.Minimize(obj), cons)

        variables = (
            [P, R_up, R_dn, Cost, phi, gamma]
            + d_up_list + d_dn_list + LS_list + SP_list
            + [du_max, dd_max, LS_max, SP_max, du_min, dd_min, LS_min, SP_min]
        )

        self.layer = CvxpyLayer(
            prob,
            parameters=[forecast] + omega_list + [om_min, om_max, eps],
            variables=variables
        )

        self._idx = {
            "P": 0,
            "R_up": 1,
            "R_dn": 2,
            "Cost": 3,
            "phi": 4,
            "gamma": 5,
        }

    def forward(self, forecast_t, omega_scen_t, om_min_t, om_max_t, eps_t, solver="ECOS"):
        B = forecast_t.shape[0]

        if solver == "ECOS":
            solver_args = {"solve_method": "ECOS", "max_iters": 10000}
        elif solver == "SCS":
            solver_args = {"solve_method": "SCS", "max_iters": 200000, "eps": 1e-6}
        else:
            raise ValueError("solver must be 'ECOS' or 'SCS'")

        outs = {
            "P_DA": [],
            "R_up": [],
            "R_dn": [],
            "Cost_DA": [],
            "phi": [],
            "gamma": [],
            "obj": [],
        }

        bG = self.b_G
        res_up_price = self.res_up_ratio * bG
        res_dn_price = self.res_dn_ratio * bG

        for b in range(B):
            fc = forecast_t[b].to(dtype=self.dtype)
            omega_list_b = [omega_scen_t[b, n].to(dtype=self.dtype) for n in range(self.N)]
            omin = (om_min_t[b] if om_min_t.ndim == 3 else om_min_t).to(dtype=self.dtype)
            omax = (om_max_t[b] if om_max_t.ndim == 3 else om_max_t).to(dtype=self.dtype)
            eps = (eps_t[b] if getattr(eps_t, "ndim", 0) > 0 else eps_t).to(dtype=self.dtype)

            sol = self.layer(fc, *omega_list_b, omin, omax, eps, solver_args=solver_args)

            P = sol[self._idx["P"]]
            R_up = sol[self._idx["R_up"]]
            R_dn = sol[self._idx["R_dn"]]
            Cost = sol[self._idx["Cost"]]
            phi = sol[self._idx["phi"]]
            gamma = sol[self._idx["gamma"]]

            stage1_energy = Cost.sum()
            stage1_reserve = (
                (res_up_price.unsqueeze(1) * R_up).sum()
                + (res_dn_price.unsqueeze(1) * R_dn).sum()
            )
            obj = stage1_energy + stage1_reserve + phi.mean() + eps * gamma

            outs["P_DA"].append(P)
            outs["R_up"].append(R_up)
            outs["R_dn"].append(R_dn)
            outs["Cost_DA"].append(Cost)
            outs["phi"].append(phi)
            outs["gamma"].append(gamma)
            outs["obj"].append(obj)

        return {k: torch.stack(v, dim=0) for k, v in outs.items()}



class MultiNode_Reserve_RT_OptNet(torch.nn.Module):
    """
    Multi-bus real-time recourse OptNet.

    Input:
        R_up:    [B, G, T]
        R_dn:    [B, G, T]
        omega14: [B, nB, T]

    Output:
        d_up:   [B, G, T]
        d_dn:   [B, G, T]
        LS:     [B, nB, T]
        SP:     [B, nB, T]
        rt_obj: [B]
    """
    def __init__(self, mgr, T=24, dtype=torch.float64):
        super().__init__()
        self.T = int(T)
        self.dtype = dtype

        self.gen_ids = list(mgr.gen_bus_list)
        self.gen_info = mgr.gen_info
        self.G = len(self.gen_ids)

        self.nB = 14
        bus_to_i = {b: b - 1 for b in range(1, 15)}

        rt_up_ratio = float(getattr(mgr.args, "rt_up_ratio", 2.0))
        rt_dn_ratio = float(getattr(mgr.args, "rt_dn_ratio", 0.5))
        voll = float(getattr(mgr.args, "voll", 1000.0))
        vosp = float(getattr(mgr.args, "vosp", 50.0))

        b = np.array([self.gen_info[g]["b"] for g in self.gen_ids], dtype=float)

        self.register_buffer("b_G", torch.tensor(b, dtype=self.dtype))
        self.register_buffer("voll", torch.tensor(voll, dtype=self.dtype))
        self.register_buffer("vosp", torch.tensor(vosp, dtype=self.dtype))
        self.register_buffer("rt_up_ratio", torch.tensor(rt_up_ratio, dtype=self.dtype))
        self.register_buffer("rt_dn_ratio", torch.tensor(rt_dn_ratio, dtype=self.dtype))

        C = np.zeros((self.nB, self.G), dtype=float)
        for gi, gbus in enumerate(self.gen_ids):
            C[bus_to_i[int(gbus)], gi] = 1.0

        # CVXPY params
        Rup = cp.Parameter((self.G, self.T), nonneg=True, name="R_up")
        Rdn = cp.Parameter((self.G, self.T), nonneg=True, name="R_dn")
        omega = cp.Parameter((self.nB, self.T), name="omega14_true")

        # vars
        du = cp.Variable((self.G, self.T), nonneg=True, name="d_up")
        dd = cp.Variable((self.G, self.T), nonneg=True, name="d_dn")
        LS = cp.Variable((self.nB, self.T), nonneg=True, name="LS")
        SP = cp.Variable((self.nB, self.T), nonneg=True, name="SP")
        s_pool = cp.Variable((self.nB, self.T), name="s_pool_RT")

        cons = []
        cons += [du <= Rup]
        cons += [dd <= Rdn]
        cons += [C @ (du - dd) - omega + LS - SP == s_pool]
        cons += [cp.sum(s_pool, axis=0) == 0]

        rt_up_price = rt_up_ratio * b
        rt_dn_price = rt_dn_ratio * b

        obj = (
            cp.sum(cp.multiply(rt_up_price.reshape(-1, 1), du))
            + cp.sum(cp.multiply(rt_dn_price.reshape(-1, 1), dd))
            + voll * cp.sum(LS)
            + vosp * cp.sum(SP)
        )

        prob = cp.Problem(cp.Minimize(obj), cons)
        self.layer = CvxpyLayer(prob, parameters=[Rup, Rdn, omega], variables=[du, dd, LS, SP])

    def forward(self, R_up, R_dn, omega14_true, solver="ECOS"):
        R_up = R_up.to(dtype=self.dtype)
        R_dn = R_dn.to(dtype=self.dtype)
        omega14_true = omega14_true.to(dtype=self.dtype)

        if solver == "ECOS":
            solver_args = {"solve_method": "ECOS"}
        elif solver == "SCS":
            solver_args = {"solve_method": "SCS", "max_iters": 10000, "eps": 1e-6}
        else:
            raise ValueError("solver must be 'ECOS' or 'SCS'")

        B = omega14_true.shape[0]
        du_out, dd_out, ls_out, sp_out, obj_out = [], [], [], [], []

        rt_up_price = self.rt_up_ratio * self.b_G
        rt_dn_price = self.rt_dn_ratio * self.b_G

        for b in range(B):
            du, dd, LS, SP = self.layer(R_up[b], R_dn[b], omega14_true[b], solver_args=solver_args)

            rt_cost = (
                (rt_up_price.unsqueeze(1) * du).sum()
                + (rt_dn_price.unsqueeze(1) * dd).sum()
                + self.voll * LS.sum()
                + self.vosp * SP.sum()
            )

            du_out.append(du)
            dd_out.append(dd)
            ls_out.append(LS)
            sp_out.append(SP)
            obj_out.append(rt_cost)

        return {
            "d_up": torch.stack(du_out, dim=0),
            "d_dn": torch.stack(dd_out, dim=0),
            "LS": torch.stack(ls_out, dim=0),
            "SP": torch.stack(sp_out, dim=0),
            "rt_obj": torch.stack(obj_out, dim=0),
        }

def compute_true_cost_optnet_multinode(da_out, rt_layer, actual_load14, forecast14, da_layer):
    """
    da_out: output of MultiNode_Reserve_DRO_DA_OptNet
    actual_load14: (B,14,T)
    forecast14: (B,14,T) 这里只是占位，实际上 RT 用 actual_load14 即可
    """
    P_DA = da_out["P_DA"]
    R_up = da_out["R_up"]
    R_dn = da_out["R_dn"]
    Cost = da_out.get("Cost_DA", None)

    # stage-1 costs
    bG = da_layer.b_G.to(P_DA.device)
    stage1_energy = Cost.sum(dim=(1,2)) if Cost is not None else (bG[None,:,None] * P_DA).sum(dim=(1,2))
    stage1_res = (
        ((da_layer.res_up_ratio * bG)[None,:,None] * R_up).sum(dim=(1,2))
        + ((da_layer.res_dn_ratio * bG)[None,:,None] * R_dn).sum(dim=(1,2))
    )

    # RT true-cost (solve RT recourse once)
    rt_out = rt_layer(P_DA, R_up, R_dn, actual_load14, solver="ECOS")
    rt_cost = rt_out["rt_obj"]  # (B,)

    total = stage1_energy + stage1_res + rt_cost
    return {"total_cost": total, "rt_out": rt_out}


def _solve_one_day_multi_det(day, args, data, pred_mean_all):
    T = args.T
    sl = slice(T * day, T * day + T)

    forecast_load = pred_mean_all[:, sl]   # (11,T) or (Nbus,T) depending on your raw data
    Y_true = data["Y_true"][:, sl]         # same shape

    mgr = IEEE14_Reserve_Deterministic_DA_Manager_MultiNode(args)
    mgr.build_model(forecast_load, output_flag=0)
    mgr.solve(threads=1, method=1, warm_start=False, output_flag=0)

    true_eval = mgr.compute_true_cost(Y_true, output_flag=0, threads=1, method=1)
    return true_eval["total_cost"]

def Average_cost_Reserve_DET_MultiNode(args, data, n_jobs=None):
    T = args.T
    Y_pred = data["Y_pred"]
    S, N, L = Y_pred.shape
    days = L // T

    pred_mean_all = Y_pred.mean(axis=0)  # (N,L)

    if n_jobs is None:
        n_jobs = min(os.cpu_count() or 1, days)

    costs = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_solve_one_day_multi_det)(day, args, data, pred_mean_all)
        for day in range(days)
    )
    return costs


def _solve_one_day_multi_ideal(day, args, data):
    T = args.T
    sl = slice(T * day, T * day + T)

    Y_true = data["Y_true"][:, sl]   # (11,T) or (14,T)

    mgr = IEEE14_Ideal_Manager_MultiNode(args)
    mgr.build_model(Y_true, output_flag=0)
    mgr.solve(threads=1, method=1, warm_start=False, output_flag=0)

    true_eval = mgr.compute_true_cost()
    return true_eval["total_cost"]

def Average_cost_Reserve_IDEAL_MultiNode(args, data, n_jobs=None):
    T = args.T
    Y_pred = data["Y_pred"]
    _, _, L = Y_pred.shape
    days = L // T

    if n_jobs is None:
        n_jobs = min(os.cpu_count() or 1, days)

    costs = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_solve_one_day_multi_ideal)(day, args, data)
        for day in range(days)
    )
    return costs




def _solve_one_day_multi(day, args, data, pred_mean_all, mode=None):
    T = args.T
    sl = slice(T*day, T*day+T)

    scenarios = data["Y_pred"][0:args.N_scen, :, sl]#.sum(axis=1)          # (S,T)
    forecast_load = pred_mean_all[:, sl]#.sum(axis=0)          # (T,)
    Y_true = data["Y_true"][:, sl]#.sum(axis=0)                # (T,)

    if mode == "ideal":
        forecast_load = Y_true.copy()
        scenarios = Y_true.reshape(-1,11,T)

    mgr = IEEE14_Reserve_SO_Manager_MultiNode(args)
    mgr.build_model(forecast_load, scenarios, output_flag=0)

    mgr.solve(threads=1, method=1, warm_start=False, output_flag=0)
    true_eval = mgr.compute_true_cost(Y_true, output_flag=0, threads=1, method=1)
    return true_eval["total_cost"]

def Average_cost_Reserve_SO_MultiNode(args, data, mode=None, n_jobs=None):
    T = args.T
    Y_pred = data["Y_pred"]
    S, N, L = Y_pred.shape
    days = L // T

    pred_mean_all = Y_pred.mean(axis=0)  # (N,L)

    if n_jobs is None:
        n_jobs = min(os.cpu_count() or 1, days)

    costs = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_solve_one_day_multi)(day, args, data, pred_mean_all, mode)
        for day in range(days)
    )
    return costs



def _solve_one_day_multi_dro(day, args, data, pred_mean_all, Lmin_11, Lmax_11, eps, mode=None):
    T = args.T
    sl = slice(T * day, T * day + T)

    # data["Y_pred"]: (S, N, L) => scen: (S, N, T)
    scen_11 = np.asarray(data["Y_pred"][0:args.N_scen, :, sl], dtype=float)
    forecast_11 = np.asarray(pred_mean_all[:, sl], dtype=float)  # (N,T)
    actual_11 = np.asarray(data["Y_true"][:, sl], dtype=float)   # (N,T)

    if mode == "ideal":
        # 用真实负荷当 forecast，且场景也用真实负荷（按你的 SO ideal 写法对齐）
        forecast_11 = actual_11.copy()
        scen_11 = np.repeat(actual_11[None, :, :], scen_11.shape[0], axis=0)  # (S,N,T)

    mgr = IEEE14_Reserve_DRO_Manager_MultiNode(args)
    mgr.build_model(
        forecast=forecast_11,           # (11,24)
        scenarios=scen_11,              # (S,11,24)
        eps=float(eps),
        hourly_load_min=Lmin_11,        # (11,24)
        hourly_load_max=Lmax_11,        # (11,24)
        output_flag=0,
    )
    mgr.solve(output_flag=0)

    res = mgr.compute_true_cost(actual=actual_11, mu=forecast_11, output_flag=0)

    # 你如果还想返回训练目标/ gamma 也可以一起返回
    out = {
        "total_cost": float(res["total_cost"]),
        "train_obj": float(mgr.model.ObjVal),
        "gamma": float(mgr.vars["gamma"].X) if "gamma" in getattr(mgr, "vars", {}) else None,
    }
    return out



def Average_cost_Reserve_DRO_MultiNode(args, data, Lmin_11, Lmax_11, eps, mode=None, n_jobs=None, return_full=False):
    T = args.T
    Y_pred = data["Y_pred"]
    S, N, L = Y_pred.shape
    days = L // T

    pred_mean_all = Y_pred.mean(axis=0)  # (N,L)

    if n_jobs is None:
        n_jobs = min(os.cpu_count() or 1, days)

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_solve_one_day_multi_dro)(day, args, data, pred_mean_all, Lmin_11, Lmax_11, eps, mode)
        for day in range(days)
    )

    if return_full:
        return results
    else:
        return [r["total_cost"] for r in results]