import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import torch
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
from joblib import Parallel, delayed

class IEEE14_Ideal_Manager_SingleNode:
    """
    Ideal single-node day-ahead dispatch with perfect forecast.

    Interpretation:
      - Forecast is assumed to be 100% accurate
      - The model is built directly on the true load
      - No stochastic optimization
      - No exogenous reserve requirement
      - Linear DA energy cost: Cost_DA = b*P + c

    This serves as a perfect-information lower-bound baseline.
    """

    def __init__(self, args):
        self.args = args

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

        self.model = None
        self.vars = {}
        self.T = None
        self._true_load = None

    def build_model(self, y_true, output_flag=0):
        """
        y_true: (T,) true system load

        Ideal case:
          - use true load directly in DA optimization
          - no reserve requirement
        """
        y_true = np.asarray(y_true, dtype=float).reshape(-1)
        T_steps = y_true.shape[0]

        m = gp.Model("SingleNode_Ideal_PerfectForecast")
        m.setParam("OutputFlag", int(output_flag))

        self.T = int(T_steps)

        # ---------- variables ----------
        P_DA = m.addVars(self.gen_bus_list, range(T_steps), lb=0, name="P_DA")
        Cost_DA = m.addVars(self.gen_bus_list, range(T_steps), lb=0, name="Cost_DA")

        # optional reserve variables kept for output compatibility
        R_up = m.addVars(self.gen_bus_list, range(T_steps), lb=0, ub=0, name="R_up")
        R_dn = m.addVars(self.gen_bus_list, range(T_steps), lb=0, ub=0, name="R_dn")

        # ---------- parameters ----------
        ramp_factor = float(getattr(self.args, "ramp_rate", 1.0))

        b_g = {g: float(self.gen_info[g]["b"]) for g in self.gen_bus_list}
        c_g = {g: float(self.gen_info[g]["c"]) for g in self.gen_bus_list}

        # ---------- objective ----------
        obj_stage1_energy = gp.quicksum(
            Cost_DA[g, t] for g in self.gen_bus_list for t in range(T_steps)
        )

        m.setObjective(obj_stage1_energy, GRB.MINIMIZE)

        # ---------- constraints ----------
        for g in self.gen_bus_list:
            pmin = float(self.gen_info[g]["min"])
            pmax = float(self.gen_info[g]["max"])
            b_lin = float(b_g[g])
            c0 = float(c_g[g])

            for t in range(T_steps):
                m.addConstr(P_DA[g, t] >= pmin, name=f"pmin_{g}_{t}")
                m.addConstr(P_DA[g, t] <= pmax, name=f"pmax_{g}_{t}")
                m.addConstr(Cost_DA[g, t] == b_lin * P_DA[g, t] + c0, name=f"lin_cost_{g}_{t}")

            for t in range(1, T_steps):
                m.addConstr(P_DA[g, t] - P_DA[g, t - 1] <= pmax * ramp_factor, name=f"ramp_up_{g}_{t}")
                m.addConstr(P_DA[g, t - 1] - P_DA[g, t] <= pmax * ramp_factor, name=f"ramp_dn_{g}_{t}")

        for t in range(T_steps):
            m.addConstr(
                gp.quicksum(P_DA[g, t] for g in self.gen_bus_list) == float(y_true[t]),
                name=f"Balance_{t}"
            )

        m.update()
        self.model = m
        self.vars = {
            "P_DA": P_DA,
            "R_up": R_up,
            "R_dn": R_dn,
            "Cost_DA": Cost_DA,
        }
        self._true_load = y_true.copy()

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

    def compute_true_cost(self, y_true=None, **kwargs):
        """
        In the ideal model, forecast is perfect, so realized cost equals solved DA cost.
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
        deploy_cost = 0.0
        penalty_cost = 0.0

        omega = np.zeros(T)
        deploy_up = np.zeros_like(P_DA)
        deploy_dn = np.zeros_like(P_DA)
        LS = np.zeros(T)
        SP = np.zeros(T)

        total_cost = stage1_energy_cost_pwl

        return {
            "stage1_energy_cost_pwl": float(stage1_energy_cost_pwl),
            "stage1_reserve_cost": float(stage1_reserve_cost),
            "rt_deploy_cost": float(deploy_cost),
            "rt_penalty_cost": float(penalty_cost),
            "total_cost": float(total_cost),
            "omega": omega,
            "P_DA": P_DA,
            "R_up": R_up,
            "R_dn": R_dn,
            "Cost_DA": Cost_DA,
            "deploy_up": deploy_up,
            "deploy_dn": deploy_dn,
            "LS": LS,
            "SP": SP,
        }


class IEEE14_Reserve_Deterministic_DA_Manager_SingleNode:
    """
    Deterministic single-node day-ahead reserve scheduling.

    Features:
      - No stochastic optimization / no SAA
      - Day-ahead only
      - Linear DA energy cost: Cost_DA = b*P + c
      - System reserve requirement:
            sum_g R_up[g,t] >= 2% * forecast[t]
            sum_g R_dn[g,t] >= 2% * forecast[t]
      - True realized cost is evaluated by a deterministic RT recourse model
    """

    def __init__(self, args):
        self.args = args

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

        self.model = None
        self.vars = {}
        self.T = None
        self._forecast_sys = None

    def build_model(self, forecasts, output_flag=0):
        """
        forecasts: (T,) system forecast load

        Deterministic DA reserve scheduling with fixed system reserve requirement:
            upward reserve  = 2% of forecast
            downward reserve = 2% of forecast
        """
        forecasts = np.asarray(forecasts, dtype=float).reshape(-1)
        T_steps = forecasts.shape[0]

        m = gp.Model("SingleNode_Reserve_DA")
        m.setParam("OutputFlag", int(output_flag))

        self.T = int(T_steps)

        # ---------- Stage-1 ----------
        P_DA = m.addVars(self.gen_bus_list, range(T_steps), lb=0, name="P_DA")
        R_up = m.addVars(self.gen_bus_list, range(T_steps), lb=0, name="R_up")
        R_dn = m.addVars(self.gen_bus_list, range(T_steps), lb=0, name="R_dn")
        Cost_DA = m.addVars(self.gen_bus_list, range(T_steps), lb=0, name="Cost_DA")

        # ---------- parameters ----------
        ramp_factor = float(getattr(self.args, "ramp_rate", 1.0))
        reserve_req_up_ratio = float(getattr(self.args, "system_reserve_up_ratio", 0.02))
        reserve_req_dn_ratio = float(getattr(self.args, "system_reserve_dn_ratio", 0.02))

        # reserve holding prices
        reserve_up_ratio = float(getattr(self.args, "reserve_up_ratio", 0.05))
        reserve_dn_ratio = float(getattr(self.args, "reserve_dn_ratio", 0.02))

        b_g = {g: float(self.gen_info[g]["b"]) for g in self.gen_bus_list}
        c_g = {g: float(self.gen_info[g]["c"]) for g in self.gen_bus_list}

        price_res_up = {g: reserve_up_ratio * b_g[g] for g in self.gen_bus_list}
        price_res_dn = {g: reserve_dn_ratio * b_g[g] for g in self.gen_bus_list}

        # ---------- objective ----------
        obj_stage1_energy = gp.quicksum(
            Cost_DA[g, t] for g in self.gen_bus_list for t in range(T_steps)
        )
        obj_stage1_reserve = gp.quicksum(
            price_res_up[g] * R_up[g, t] + price_res_dn[g] * R_dn[g, t]
            for g in self.gen_bus_list for t in range(T_steps)
        )

        m.setObjective(obj_stage1_energy + obj_stage1_reserve, GRB.MINIMIZE)

        # ---------- constraints ----------
        for g in self.gen_bus_list:
            pmin = float(self.gen_info[g]["min"])
            pmax = float(self.gen_info[g]["max"])
            b_lin = float(b_g[g])
            c0 = float(c_g[g])

            for t in range(T_steps):
                m.addConstr(P_DA[g, t] >= pmin, name=f"pmin_{g}_{t}")
                m.addConstr(P_DA[g, t] <= pmax, name=f"pmax_{g}_{t}")

                m.addConstr(P_DA[g, t] + R_up[g, t] <= pmax, name=f"cap_up_{g}_{t}")
                m.addConstr(P_DA[g, t] - R_dn[g, t] >= pmin, name=f"cap_dn_{g}_{t}")

                # linear DA energy cost
                m.addConstr(Cost_DA[g, t] == b_lin * P_DA[g, t] + c0, name=f"lin_cost_{g}_{t}")

            # ramp constraints on DA schedule
            for t in range(1, T_steps):
                m.addConstr(P_DA[g, t] - P_DA[g, t - 1] <= pmax * ramp_factor, name=f"ramp_up_{g}_{t}")
                m.addConstr(P_DA[g, t - 1] - P_DA[g, t] <= pmax * ramp_factor, name=f"ramp_dn_{g}_{t}")

        # DA balance + system reserve requirements
        for t in range(T_steps):
            m.addConstr(
                gp.quicksum(P_DA[g, t] for g in self.gen_bus_list) == float(forecasts[t]),
                name=f"DA_Balance_{t}"
            )

            m.addConstr(
                gp.quicksum(R_up[g, t] for g in self.gen_bus_list) >= reserve_req_up_ratio * float(forecasts[t]),
                name=f"Sys_Reserve_Up_{t}"
            )

            m.addConstr(
                gp.quicksum(R_dn[g, t] for g in self.gen_bus_list) >= reserve_req_dn_ratio * float(forecasts[t]),
                name=f"Sys_Reserve_Dn_{t}"
            )

        m.update()
        self.model = m
        self.vars = {
            "P_DA": P_DA,
            "R_up": R_up,
            "R_dn": R_dn,
            "Cost_DA": Cost_DA,
        }
        self._forecast_sys = forecasts.copy()

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

    def compute_true_cost(self, y_true, mu=None, output_flag=0, warm_start=False, threads=None, method=None):
        """
        Evaluate realized cost under true load y_true using fixed DA schedule and reserve decisions.
        RT recourse is solved deterministically for all t in one model.
        """
        if self.model is None or self.model.status != GRB.OPTIMAL:
            raise RuntimeError("Model not solved to OPTIMAL. Call solve() first.")

        if mu is None:
            if self._forecast_sys is None:
                raise RuntimeError("No stored forecast found. Pass mu explicitly.")
            sys_fcst = np.asarray(self._forecast_sys, dtype=float).reshape(-1)
        else:
            sys_fcst = np.asarray(mu, dtype=float).reshape(-1)

        sys_true = np.asarray(y_true, dtype=float).reshape(-1)
        if sys_true.shape[0] != sys_fcst.shape[0]:
            raise ValueError("y_true length != forecast length")

        T = sys_fcst.shape[0]
        G_list = self.gen_bus_list
        G = len(G_list)

        reserve_up_ratio = float(getattr(self.args, "reserve_up_ratio", 0.05))
        reserve_dn_ratio = float(getattr(self.args, "reserve_dn_ratio", 0.02))
        rt_up_ratio = float(getattr(self.args, "rt_up_ratio", 2.0))
        rt_dn_ratio = float(getattr(self.args, "rt_dn_ratio", 0.5))
        voll = float(getattr(self.args, "voll", 1000.0))
        vosp = float(getattr(self.args, "vosp", 50.0))

        b = np.array([float(self.gen_info[g]["b"]) for g in G_list], dtype=float)
        price_res_up = reserve_up_ratio * b
        price_res_dn = reserve_dn_ratio * b
        price_rt_up = rt_up_ratio * b
        price_rt_dn = rt_dn_ratio * b

        # extract stage-1 decisions
        P_DA = np.array([[self.vars["P_DA"][g, t].X for g in G_list] for t in range(T)], dtype=float)
        R_up = np.array([[self.vars["R_up"][g, t].X for g in G_list] for t in range(T)], dtype=float)
        R_dn = np.array([[self.vars["R_dn"][g, t].X for g in G_list] for t in range(T)], dtype=float)
        Cost_DA = np.array([[self.vars["Cost_DA"][g, t].X for g in G_list] for t in range(T)], dtype=float)

        stage1_energy_cost_pwl = float(Cost_DA.sum())  # keep old key name for compatibility
        stage1_reserve_cost = float(
            np.sum(R_up * price_res_up.reshape(1, -1) + R_dn * price_res_dn.reshape(1, -1))
        )

        omega = (sys_true - sys_fcst).astype(float)

        # ---- one RT model for all t ----
        sm = gp.Model("RT_AllT_SingleNode")
        sm.setParam("OutputFlag", int(output_flag))
        if warm_start:
            sm.setParam("LPWarmStart", 1)
        if threads is not None:
            sm.setParam("Threads", int(threads))
        if method is not None:
            sm.setParam("Method", int(method))

        du = sm.addVars(range(G), range(T), lb=0, name="du")
        dd = sm.addVars(range(G), range(T), lb=0, name="dd")
        ls = sm.addVars(range(T), lb=0, name="ls")
        sp = sm.addVars(range(T), lb=0, name="sp")

        for t in range(T):
            for i in range(G):
                sm.addConstr(du[i, t] <= float(R_up[t, i]), name=f"du_le_Rup_{i}_{t}")
                sm.addConstr(dd[i, t] <= float(R_dn[t, i]), name=f"dd_le_Rdn_{i}_{t}")

            sm.addConstr(
                gp.quicksum(du[i, t] - dd[i, t] for i in range(G)) + ls[t] - sp[t] == float(omega[t]),
                name=f"balance_{t}"
            )

        sm.setObjective(
            gp.quicksum(
                float(price_rt_up[i]) * du[i, t] + float(price_rt_dn[i]) * dd[i, t]
                for i in range(G) for t in range(T)
            )
            + voll * gp.quicksum(ls[t] for t in range(T))
            + vosp * gp.quicksum(sp[t] for t in range(T)),
            GRB.MINIMIZE
        )

        sm.optimize()
        if sm.status != GRB.OPTIMAL:
            raise RuntimeError(f"RT all-T subproblem not optimal, status={sm.status}")

        deploy_up = np.zeros((T, G))
        deploy_dn = np.zeros((T, G))
        LS = np.zeros(T)
        SP = np.zeros(T)

        for t in range(T):
            for i in range(G):
                deploy_up[t, i] = float(du[i, t].X)
                deploy_dn[t, i] = float(dd[i, t].X)
            LS[t] = float(ls[t].X)
            SP[t] = float(sp[t].X)

        deploy_cost = float(sum(
            price_rt_up[i] * du[i, t].X + price_rt_dn[i] * dd[i, t].X
            for i in range(G) for t in range(T)
        ))
        penalty_cost = float(
            voll * sum(ls[t].X for t in range(T)) + vosp * sum(sp[t].X for t in range(T))
        )

        sm.dispose()

        total_cost = stage1_energy_cost_pwl + stage1_reserve_cost + deploy_cost + penalty_cost

        return {
            "stage1_energy_cost_pwl": float(stage1_energy_cost_pwl),
            "stage1_reserve_cost": float(stage1_reserve_cost),
            "rt_deploy_cost": float(deploy_cost),
            "rt_penalty_cost": float(penalty_cost),
            "total_cost": float(total_cost),
            "omega": omega,
            "P_DA": P_DA,
            "R_up": R_up,
            "R_dn": R_dn,
            "Cost_DA": Cost_DA,
            "deploy_up": deploy_up,
            "deploy_dn": deploy_dn,
            "LS": LS,
            "SP": SP,
        }


class IEEE14_Reserve_SO_Manager_SingleNode:
    """
    Single-node reserve-based RT recourse (d_up/d_dn + LS/SP) two-stage SAA.

    MODIFIED:
      - remove PWL epigraph constraints
      - stage-1 energy cost only uses linear term: Cost_DA = b*P (+c)
      - objective uses sum(Cost_DA)

    SPEEDUP (修改 compute_true_cost):
      - 原来 compute_true_cost: 每个t建一个RT模型并求解一次
      - 现在: 建一个RT模型覆盖所有t，一次 optimize() 解完（显著加速）
      - 不保留 slow 版本
    """

    def __init__(self, args):
        self.args = args

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

        self.model = None
        self.vars = {}
        self.T = None
        self.N_scen = None
        self._forecast_sys = None

    def build_model(self, forecasts, scenarios, output_flag=0):
        """
        forecasts: (T,) system forecast load
        scenarios: (N,T) system load scenarios

        MODIFIED:
          - remove PWL epigraph constraints
          - stage-1 energy cost only uses linear term: Cost_DA = b*P (+c)
          - objective still uses sum(Cost_DA) (linear)
        """
        forecasts = np.asarray(forecasts, dtype=float).reshape(-1)
        scenarios = np.asarray(scenarios, dtype=float)
        if scenarios.ndim != 2:
            raise ValueError(f"scenarios must be (N,T), got {scenarios.shape}")

        N_scen, T_steps = scenarios.shape
        if forecasts.shape[0] != T_steps:
            raise ValueError(f"forecasts length {forecasts.shape[0]} != T {T_steps}")

        Omega = scenarios - forecasts[None, :]  # (N,T)

        m = gp.Model("SingleNode_Reserve_SO")
        m.setParam("OutputFlag", int(output_flag))

        self.T = int(T_steps)
        self.N_scen = int(N_scen)

        # ---------- Stage-1 ----------
        P_DA = m.addVars(self.gen_bus_list, range(T_steps), lb=0, name="P_DA")
        R_up = m.addVars(self.gen_bus_list, range(T_steps), lb=0, name="R_up")
        R_dn = m.addVars(self.gen_bus_list, range(T_steps), lb=0, name="R_dn")
        Cost_DA = m.addVars(self.gen_bus_list, range(T_steps), lb=0, name="Cost_DA")

        # ---------- Stage-2 (SAA) ----------
        d_up = m.addVars(self.gen_bus_list, range(N_scen), range(T_steps), lb=0, name="d_up")
        d_dn = m.addVars(self.gen_bus_list, range(N_scen), range(T_steps), lb=0, name="d_dn")
        LS = m.addVars(range(N_scen), range(T_steps), lb=0, name="LS")
        SP = m.addVars(range(N_scen), range(T_steps), lb=0, name="SP")

        # ---------- parameters ----------
        ramp_factor = float(getattr(self.args, "ramp_rate", 1.0))
        voll = float(getattr(self.args, "voll", 1000.0))
        vosp = float(getattr(self.args, "vosp", 50.0))

        reserve_up_ratio = float(getattr(self.args, "reserve_up_ratio", 0.05))
        reserve_dn_ratio = float(getattr(self.args, "reserve_dn_ratio", 0.02))
        rt_up_ratio = float(getattr(self.args, "rt_up_ratio", 2.0))
        rt_dn_ratio = float(getattr(self.args, "rt_dn_ratio", 0.5))

        b_g = {g: float(self.gen_info[g]["b"]) for g in self.gen_bus_list}
        c_g = {g: float(self.gen_info[g]["c"]) for g in self.gen_bus_list}

        price_res_up = {g: reserve_up_ratio * b_g[g] for g in self.gen_bus_list}
        price_res_dn = {g: reserve_dn_ratio * b_g[g] for g in self.gen_bus_list}
        price_rt_up = {g: rt_up_ratio * b_g[g] for g in self.gen_bus_list}
        price_rt_dn = {g: rt_dn_ratio * b_g[g] for g in self.gen_bus_list}

        # ---------- objective ----------
        obj_stage1_energy = gp.quicksum(Cost_DA[g, t] for g in self.gen_bus_list for t in range(T_steps))
        obj_stage1_reserve = gp.quicksum(
            price_res_up[g] * R_up[g, t] + price_res_dn[g] * R_dn[g, t]
            for g in self.gen_bus_list for t in range(T_steps)
        )
        obj_stage2_deploy = (1.0 / N_scen) * gp.quicksum(
            price_rt_up[g] * d_up[g, n, t] + price_rt_dn[g] * d_dn[g, n, t]
            for g in self.gen_bus_list for n in range(N_scen) for t in range(T_steps)
        )
        obj_stage2_penalty = (1.0 / N_scen) * gp.quicksum(
            voll * LS[n, t] + vosp * SP[n, t]
            for n in range(N_scen) for t in range(T_steps)
        )

        m.setObjective(obj_stage1_energy + obj_stage1_reserve + obj_stage2_deploy + obj_stage2_penalty,
                       GRB.MINIMIZE)

        # ---------- constraints ----------
        for g in self.gen_bus_list:
            pmin, pmax = float(self.gen_info[g]["min"]), float(self.gen_info[g]["max"])
            b_lin = float(b_g[g])
            c0 = float(c_g[g])

            for t in range(T_steps):
                m.addConstr(P_DA[g, t] >= pmin)
                m.addConstr(P_DA[g, t] <= pmax)

                m.addConstr(P_DA[g, t] + R_up[g, t] <= pmax, name=f"cap_up_{g}_{t}")
                m.addConstr(P_DA[g, t] - R_dn[g, t] >= pmin, name=f"cap_dn_{g}_{t}")

                # linear DA energy cost only
                m.addConstr(Cost_DA[g, t] == b_lin * P_DA[g, t] + c0, name=f"lin_cost_{g}_{t}")

            # ramp on DA schedule
            for t in range(1, T_steps):
                m.addConstr(P_DA[g, t] - P_DA[g, t - 1] <= pmax * ramp_factor)
                m.addConstr(P_DA[g, t - 1] - P_DA[g, t] <= pmax * ramp_factor)

            # deployment bounded by reserved bands
            for n in range(N_scen):
                for t in range(T_steps):
                    m.addConstr(d_up[g, n, t] <= R_up[g, t])
                    m.addConstr(d_dn[g, n, t] <= R_dn[g, t])

        # DA balance
        for t in range(T_steps):
            m.addConstr(
                gp.quicksum(P_DA[g, t] for g in self.gen_bus_list) == float(forecasts[t]),
                name=f"DA_Balance_{t}"
            )

        # RT balance per scenario: reserves first, then LS/SP
        for n in range(N_scen):
            for t in range(T_steps):
                sys_mismatch = float(Omega[n, t])  # scenario - forecast
                m.addConstr(
                    gp.quicksum(d_up[g, n, t] - d_dn[g, n, t] for g in self.gen_bus_list)
                    + LS[n, t] - SP[n, t]
                    == sys_mismatch,
                    name=f"RT_Balance_{n}_{t}"
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
        }
        self._forecast_sys = forecasts.copy()

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
        if self.model.status == GRB.OPTIMAL:
             pass
        #    print(f"Solved. Obj: {self.model.ObjVal:.2f}")
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

    def compute_true_cost(self, y_true, mu=None, output_flag=0, warm_start=False, threads=None, method=None):
        if self.model is None or self.model.status != GRB.OPTIMAL:
            raise RuntimeError("Model not solved to OPTIMAL. Call solve() first.")

        if mu is None:
            if self._forecast_sys is None:
                raise RuntimeError("No stored forecast found. Pass mu explicitly.")
            sys_fcst = np.asarray(self._forecast_sys, dtype=float).reshape(-1)
        else:
            sys_fcst = np.asarray(mu, dtype=float).reshape(-1)

        sys_true = np.asarray(y_true, dtype=float).reshape(-1)
        if sys_true.shape[0] != sys_fcst.shape[0]:
            raise ValueError("y_true length != forecast length")

        T = sys_fcst.shape[0]
        G_list = self.gen_bus_list
        G = len(G_list)

        reserve_up_ratio = float(getattr(self.args, "reserve_up_ratio", 0.05))
        reserve_dn_ratio = float(getattr(self.args, "reserve_dn_ratio", 0.02))
        rt_up_ratio = float(getattr(self.args, "rt_up_ratio", 2.0))
        rt_dn_ratio = float(getattr(self.args, "rt_dn_ratio", 0.5))
        voll = float(getattr(self.args, "voll", 1000.0))
        vosp = float(getattr(self.args, "vosp", 50.0))

        b = np.array([float(self.gen_info[g]["b"]) for g in G_list], dtype=float)
        price_res_up = reserve_up_ratio * b
        price_res_dn = reserve_dn_ratio * b
        price_rt_up = rt_up_ratio * b
        price_rt_dn = rt_dn_ratio * b

        # extract stage-1 decisions
        P_DA = np.array([[self.vars["P_DA"][g, t].X for g in G_list] for t in range(T)], dtype=float)
        R_up = np.array([[self.vars["R_up"][g, t].X for g in G_list] for t in range(T)], dtype=float)
        R_dn = np.array([[self.vars["R_dn"][g, t].X for g in G_list] for t in range(T)], dtype=float)

        # STRICT stage-1 energy cost: use solved Cost_DA (linear b*P+c)
        Cost_DA = np.array([[self.vars["Cost_DA"][g, t].X for g in G_list] for t in range(T)], dtype=float)
        stage1_energy_cost_pwl = float(Cost_DA.sum())  # name kept for backward compatibility

        stage1_reserve_cost = float(
            np.sum(R_up * price_res_up.reshape(1, -1) + R_dn * price_res_dn.reshape(1, -1))
        )

        omega = (sys_true - sys_fcst).astype(float)

        # ---- ONE RT model for all t ----
        sm = gp.Model("RT_AllT_SingleNode")
        sm.setParam("OutputFlag", int(output_flag))
        if warm_start:
            sm.setParam("LPWarmStart", 1)
        if threads is not None:
            sm.setParam("Threads", int(threads))
        if method is not None:
            sm.setParam("Method", int(method))

        du = sm.addVars(range(G), range(T), lb=0, name="du")
        dd = sm.addVars(range(G), range(T), lb=0, name="dd")
        ls = sm.addVars(range(T), lb=0, name="ls")
        sp = sm.addVars(range(T), lb=0, name="sp")

        for t in range(T):
            for i in range(G):
                sm.addConstr(du[i, t] <= float(R_up[t, i]), name=f"du_le_Rup_{i}_{t}")
                sm.addConstr(dd[i, t] <= float(R_dn[t, i]), name=f"dd_le_Rdn_{i}_{t}")

            sm.addConstr(
                gp.quicksum(du[i, t] - dd[i, t] for i in range(G)) + ls[t] - sp[t] == float(omega[t]),
                name=f"balance_{t}"
            )

        sm.setObjective(
            gp.quicksum(float(price_rt_up[i]) * du[i, t] + float(price_rt_dn[i]) * dd[i, t]
                        for i in range(G) for t in range(T))
            + voll * gp.quicksum(ls[t] for t in range(T))
            + vosp * gp.quicksum(sp[t] for t in range(T)),
            GRB.MINIMIZE
        )

        sm.optimize()
        if sm.status != GRB.OPTIMAL:
            raise RuntimeError(f"RT all-T subproblem not optimal, status={sm.status}")

        deploy_up = np.zeros((T, G))
        deploy_dn = np.zeros((T, G))
        LS = np.zeros(T)
        SP = np.zeros(T)

        for t in range(T):
            for i in range(G):
                deploy_up[t, i] = float(du[i, t].X)
                deploy_dn[t, i] = float(dd[i, t].X)
            LS[t] = float(ls[t].X)
            SP[t] = float(sp[t].X)

        deploy_cost = float(sum(price_rt_up[i] * du[i, t].X + price_rt_dn[i] * dd[i, t].X
                                for i in range(G) for t in range(T)))
        penalty_cost = float(voll * sum(ls[t].X for t in range(T)) + vosp * sum(sp[t].X for t in range(T)))

        sm.dispose()

        total_cost = stage1_energy_cost_pwl + stage1_reserve_cost + deploy_cost + penalty_cost

        return {
            "stage1_energy_cost_pwl": float(stage1_energy_cost_pwl),
            "stage1_reserve_cost": float(stage1_reserve_cost),
            "rt_deploy_cost": float(deploy_cost),
            "rt_penalty_cost": float(penalty_cost),
            "total_cost": float(total_cost),
            "omega": omega,
            "P_DA": P_DA,
            "R_up": R_up,
            "R_dn": R_dn,
            "Cost_DA": Cost_DA,
            "deploy_up": deploy_up,
            "deploy_dn": deploy_dn,
            "LS": LS,
            "SP": SP,
        }


class IEEE14_Reserve_DRO_Manager_SingleNode:
    """
    LP Wasserstein-DRO with UB/LB extreme points, implemented as a linear program.

      min  stage1_cost + (1/N) * sum_n phi[n] + eps * gamma

      s.t. (per sample n)
           phi[n] >= Q(omega_n)                                            (sample point)
           phi[n] >= Q(omega_max) - gamma * ||omega_max - omega_n||_1      (UB extreme)
           phi[n] >= Q(omega_min) - gamma * ||omega_min - omega_n||_1      (LB extreme)

    IMPORTANT:
      ||omega_ext - omega_n||_1 is PRECOMPUTED as a CONSTANT (numpy),
      so gamma * distance stays LINEAR.
    """

    def __init__(self, args):
        self.args = args

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

        self.model = None
        self.vars = {}
        self.T = None
        self.N_scen = None
        self._forecast_sys = None

        # debug/saved
        self._Omega = None
        self._Omega_min = None
        self._Omega_max = None
        self._dro_eps = None
        self._dro_ub_lb_scale = None
        self._dist_max = None
        self._dist_min = None

    def build_model(
        self,
        forecasts,
        scenarios,
        eps,
        ub_lb_scale=1.0,
        fixed_lb=None,
        fixed_ub=None,
        fixed_om_min=None,
        fixed_om_max=None,
        hourly_load_min=None,   # time-varying load lower bound, shape (T,)
        hourly_load_max=None,   # time-varying load upper bound, shape (T,)
        output_flag=0,
    ):
        forecasts = np.asarray(forecasts, dtype=float).reshape(-1)
        scenarios = np.asarray(scenarios, dtype=float)
        if scenarios.ndim != 2:
            raise ValueError(f"scenarios must be (N,T), got {scenarios.shape}")
        N_scen, T_steps = scenarios.shape
        if forecasts.shape[0] != T_steps:
            raise ValueError(f"forecasts length {forecasts.shape[0]} != T {T_steps}")

        Omega = scenarios - forecasts[None, :]  # (N,T)

        # -------- choose omega_min/max (TIME-VARYING) --------
        # Priority order:
        #   1) hourly_load_min/max  -> omega bounds implied by load bounds,
        #      BUT if these are tighter than sample extrema, relax to sample extrema (cover both).
        #   2) fixed_om_min/max     -> directly use omega bounds
        #   3) fixed_lb/ub          -> constant omega bounds
        #   4) from samples         -> sample min/max scaled
        if hourly_load_min is not None or hourly_load_max is not None:
            if hourly_load_min is None or hourly_load_max is None:
                raise ValueError("Pass both hourly_load_min and hourly_load_max (length T).")
            Lmin = np.asarray(hourly_load_min, dtype=float).reshape(-1)
            Lmax = np.asarray(hourly_load_max, dtype=float).reshape(-1)
            if Lmin.shape[0] != T_steps or Lmax.shape[0] != T_steps:
                raise ValueError("hourly_load_min/max must have length T.")

            # hourly implied omega bounds
            om_min_h = Lmin - forecasts
            om_max_h = Lmax - forecasts
            om_min_h, om_max_h = np.minimum(om_min_h, om_max_h), np.maximum(om_min_h, om_max_h)

            # sample implied omega bounds
            om_min_s = Omega.min(axis=0)
            om_max_s = Omega.max(axis=0)

            # optnet-style: final bounds cover BOTH hourly bounds and sample extrema
            om_min = np.minimum(om_min_h, om_min_s)
            om_max = np.maximum(om_max_h, om_max_s)

        elif fixed_om_min is not None or fixed_om_max is not None:
            if fixed_om_min is None or fixed_om_max is None:
                raise ValueError("Pass both fixed_om_min and fixed_om_max.")
            om_min = np.asarray(fixed_om_min, dtype=float).reshape(-1)
            om_max = np.asarray(fixed_om_max, dtype=float).reshape(-1)
            if om_min.shape[0] != T_steps or om_max.shape[0] != T_steps:
                raise ValueError("fixed_om_min/max must have length T.")
            om_min, om_max = np.minimum(om_min, om_max), np.maximum(om_min, om_max)

        elif fixed_lb is not None or fixed_ub is not None:
            if fixed_lb is None or fixed_ub is None:
                raise ValueError("Pass both fixed_lb and fixed_ub.")
            om_min = np.full(T_steps, float(fixed_lb), dtype=float)
            om_max = np.full(T_steps, float(fixed_ub), dtype=float)
            om_min, om_max = np.minimum(om_min, om_max), np.maximum(om_min, om_max)

        else:
            om_max = Omega.max(axis=0) * float(ub_lb_scale)
            om_min = Omega.min(axis=0) * float(ub_lb_scale)

        # --------- PRECOMPUTE L1 DISTANCES (CONSTANTS) ----------
        dist_max = np.sum(np.abs(om_max[None, :] - Omega), axis=1)  # (N,)
        dist_min = np.sum(np.abs(om_min[None, :] - Omega), axis=1)  # (N,)

        m = gp.Model("IEEE14_SingleNode_Reserve_DRO_UBLB")
        m.setParam("OutputFlag", int(output_flag))

        self.T = int(T_steps)
        self.N_scen = int(N_scen)

        # ---------- Stage-1 ----------
        P_DA = m.addVars(self.gen_bus_list, range(T_steps), lb=0, name="P_DA")
        R_up = m.addVars(self.gen_bus_list, range(T_steps), lb=0, name="R_up")
        R_dn = m.addVars(self.gen_bus_list, range(T_steps), lb=0, name="R_dn")
        Cost_DA = m.addVars(self.gen_bus_list, range(T_steps), lb=0, name="Cost_DA")

        # ---------- Stage-2 recourse for each sample ----------
        d_up = m.addVars(self.gen_bus_list, range(N_scen), range(T_steps), lb=0, name="d_up")
        d_dn = m.addVars(self.gen_bus_list, range(N_scen), range(T_steps), lb=0, name="d_dn")
        LS = m.addVars(range(N_scen), range(T_steps), lb=0, name="LS")
        SP = m.addVars(range(N_scen), range(T_steps), lb=0, name="SP")

        # ---------- Stage-2 recourse for UB/LB extreme points ----------
        du_max = m.addVars(self.gen_bus_list, range(T_steps), lb=0, name="du_max")
        dd_max = m.addVars(self.gen_bus_list, range(T_steps), lb=0, name="dd_max")
        LS_max = m.addVars(range(T_steps), lb=0, name="LS_max")
        SP_max = m.addVars(range(T_steps), lb=0, name="SP_max")

        du_min = m.addVars(self.gen_bus_list, range(T_steps), lb=0, name="du_min")
        dd_min = m.addVars(self.gen_bus_list, range(T_steps), lb=0, name="dd_min")
        LS_min = m.addVars(range(T_steps), lb=0, name="LS_min")
        SP_min = m.addVars(range(T_steps), lb=0, name="SP_min")

        # ---------- DRO variables ----------
        phi = m.addVars(range(N_scen), lb=-GRB.INFINITY, name="phi")
        gamma = m.addVar(lb=0, name="gamma")

        # ---------- cost parameters ----------
        ramp_factor = float(getattr(self.args, "ramp_rate", 1.0))
        voll = float(getattr(self.args, "voll", 1000.0))
        vosp = float(getattr(self.args, "vosp", 50.0))

        reserve_up_ratio = float(getattr(self.args, "reserve_up_ratio", 0.05))
        reserve_dn_ratio = float(getattr(self.args, "reserve_dn_ratio", 0.02))
        rt_up_ratio = float(getattr(self.args, "rt_up_ratio", 2.0))
        rt_dn_ratio = float(getattr(self.args, "rt_dn_ratio", 0.5))

        b_g = {g: float(self.gen_info[g]["b"]) for g in self.gen_bus_list}
        c_g = {g: float(self.gen_info[g]["c"]) for g in self.gen_bus_list}

        price_res_up = {g: reserve_up_ratio * b_g[g] for g in self.gen_bus_list}
        price_res_dn = {g: reserve_dn_ratio * b_g[g] for g in self.gen_bus_list}
        price_rt_up = {g: rt_up_ratio * b_g[g] for g in self.gen_bus_list}
        price_rt_dn = {g: rt_dn_ratio * b_g[g] for g in self.gen_bus_list}

        # ---------- recourse cost expressions ----------
        def recourse_cost_sample(n):
            return (
                gp.quicksum(
                    price_rt_up[g] * d_up[g, n, t] + price_rt_dn[g] * d_dn[g, n, t]
                    for g in self.gen_bus_list
                    for t in range(T_steps)
                )
                + gp.quicksum(voll * LS[n, t] + vosp * SP[n, t] for t in range(T_steps))
            )

        cost_max = (
            gp.quicksum(
                price_rt_up[g] * du_max[g, t] + price_rt_dn[g] * dd_max[g, t]
                for g in self.gen_bus_list
                for t in range(T_steps)
            )
            + gp.quicksum(voll * LS_max[t] + vosp * SP_max[t] for t in range(T_steps))
        )

        cost_min = (
            gp.quicksum(
                price_rt_up[g] * du_min[g, t] + price_rt_dn[g] * dd_min[g, t]
                for g in self.gen_bus_list
                for t in range(T_steps)
            )
            + gp.quicksum(voll * LS_min[t] + vosp * SP_min[t] for t in range(T_steps))
        )

        # ---------- objective ----------
        obj_stage1_energy = gp.quicksum(Cost_DA[g, t] for g in self.gen_bus_list for t in range(T_steps))
        obj_stage1_reserve = gp.quicksum(
            price_res_up[g] * R_up[g, t] + price_res_dn[g] * R_dn[g, t]
            for g in self.gen_bus_list
            for t in range(T_steps)
        )
        obj_dro = (1.0 / N_scen) * gp.quicksum(phi[n] for n in range(N_scen)) + float(eps) * gamma
        m.setObjective(obj_stage1_energy + obj_stage1_reserve + obj_dro, GRB.MINIMIZE)

        # ---------- constraints ----------
        for g in self.gen_bus_list:
            pmin, pmax = float(self.gen_info[g]["min"]), float(self.gen_info[g]["max"])
            b_lin = float(b_g[g])
            c0 = float(c_g[g])

            for t in range(T_steps):
                m.addConstr(P_DA[g, t] >= pmin, name=f"Pmin_{g}_{t}")
                m.addConstr(P_DA[g, t] <= pmax, name=f"Pmax_{g}_{t}")
                m.addConstr(P_DA[g, t] + R_up[g, t] <= pmax, name=f"cap_up_{g}_{t}")
                m.addConstr(P_DA[g, t] - R_dn[g, t] >= pmin, name=f"cap_dn_{g}_{t}")
                m.addConstr(Cost_DA[g, t] == b_lin * P_DA[g, t] + c0, name=f"lin_cost_{g}_{t}")

            for t in range(1, T_steps):
                m.addConstr(P_DA[g, t] - P_DA[g, t - 1] <= pmax * ramp_factor, name=f"ramp_up_{g}_{t}")
                m.addConstr(P_DA[g, t - 1] - P_DA[g, t] <= pmax * ramp_factor, name=f"ramp_dn_{g}_{t}")

            # sample deployment bounded by reserved bands
            for n in range(N_scen):
                for t in range(T_steps):
                    m.addConstr(d_up[g, n, t] <= R_up[g, t], name=f"du_le_Rup_{g}_{n}_{t}")
                    m.addConstr(d_dn[g, n, t] <= R_dn[g, t], name=f"dd_le_Rdn_{g}_{n}_{t}")

            # extreme deployment bounded by reserved bands
            for t in range(T_steps):
                m.addConstr(du_max[g, t] <= R_up[g, t], name=f"duMax_le_Rup_{g}_{t}")
                m.addConstr(dd_max[g, t] <= R_dn[g, t], name=f"ddMax_le_Rdn_{g}_{t}")
                m.addConstr(du_min[g, t] <= R_up[g, t], name=f"duMin_le_Rup_{g}_{t}")
                m.addConstr(dd_min[g, t] <= R_dn[g, t], name=f"ddMin_le_Rdn_{g}_{t}")

        # DA balance
        for t in range(T_steps):
            m.addConstr(
                gp.quicksum(P_DA[g, t] for g in self.gen_bus_list) == float(forecasts[t]),
                name=f"DA_Balance_{t}",
            )

        # RT balance: per sample
        for n in range(N_scen):
            for t in range(T_steps):
                sys_mismatch = float(Omega[n, t])
                m.addConstr(
                    gp.quicksum(d_up[g, n, t] - d_dn[g, n, t] for g in self.gen_bus_list)
                    + LS[n, t] - SP[n, t]
                    == sys_mismatch,
                    name=f"RT_Balance_{n}_{t}",
                )

        # RT balance: extremes omega_max/min
        for t in range(T_steps):
            m.addConstr(
                gp.quicksum(du_max[g, t] - dd_max[g, t] for g in self.gen_bus_list)
                + LS_max[t] - SP_max[t]
                == float(om_max[t]),
                name=f"RT_Balance_max_{t}",
            )
            m.addConstr(
                gp.quicksum(du_min[g, t] - dd_min[g, t] for g in self.gen_bus_list)
                + LS_min[t] - SP_min[t]
                == float(om_min[t]),
                name=f"RT_Balance_min_{t}",
            )

        # DRO UB/LB constraints: 3 per sample n (ALL LINEAR, distances are constants)
        for n in range(N_scen):
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
            "d_up": d_up,
            "d_dn": d_dn,
            "LS": LS,
            "SP": SP,
            "du_max": du_max,
            "dd_max": dd_max,
            "LS_max": LS_max,
            "SP_max": SP_max,
            "du_min": du_min,
            "dd_min": dd_min,
            "LS_min": LS_min,
            "SP_min": SP_min,
            "phi": phi,
            "gamma": gamma,
        }

        self._forecast_sys = forecasts.copy()
        self._Omega = Omega.copy()
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

    def compute_true_cost(self, y_true, mu=None, output_flag=0, warm_start=False, threads=None, method=None):
        if self.model is None or self.model.status != GRB.OPTIMAL:
            raise RuntimeError("Model not solved to OPTIMAL. Call solve() first.")

        # forecast
        if mu is None:
            if self._forecast_sys is None:
                raise RuntimeError("No stored forecast found. Pass mu explicitly.")
            sys_fcst = np.asarray(self._forecast_sys, dtype=float).reshape(-1)
        else:
            sys_fcst = np.asarray(mu, dtype=float).reshape(-1)

        # true
        sys_true = np.asarray(y_true, dtype=float).reshape(-1)
        if sys_true.shape[0] != sys_fcst.shape[0]:
            raise ValueError("y_true length != forecast length")

        T = sys_fcst.shape[0]
        G_list = self.gen_bus_list
        G = len(G_list)

        # prices
        reserve_up_ratio = float(getattr(self.args, "reserve_up_ratio", 0.05))
        reserve_dn_ratio = float(getattr(self.args, "reserve_dn_ratio", 0.02))
        rt_up_ratio = float(getattr(self.args, "rt_up_ratio", 2.0))
        rt_dn_ratio = float(getattr(self.args, "rt_dn_ratio", 0.5))
        voll = float(getattr(self.args, "voll", 1000.0))
        vosp = float(getattr(self.args, "vosp", 50.0))

        b = np.array([float(self.gen_info[g]["b"]) for g in G_list], dtype=float)
        price_res_up = reserve_up_ratio * b
        price_res_dn = reserve_dn_ratio * b
        price_rt_up = rt_up_ratio * b
        price_rt_dn = rt_dn_ratio * b

        # stage-1 decisions (from solved DRO)
        P_DA = np.array([[self.vars["P_DA"][g, t].X for g in G_list] for t in range(T)], dtype=float)
        R_up = np.array([[self.vars["R_up"][g, t].X for g in G_list] for t in range(T)], dtype=float)
        R_dn = np.array([[self.vars["R_dn"][g, t].X for g in G_list] for t in range(T)], dtype=float)
        Cost_DA = np.array([[self.vars["Cost_DA"][g, t].X for g in G_list] for t in range(T)], dtype=float)

        stage1_energy_cost = float(Cost_DA.sum())
        stage1_reserve_cost = float(np.sum(R_up * price_res_up.reshape(1, -1) + R_dn * price_res_dn.reshape(1, -1)))

        omega = (sys_true - sys_fcst).astype(float)

        # ---- ONE RT model for all t ----
        sm = gp.Model("RT_AllT_SingleNode_eval_DRO")
        sm.setParam("OutputFlag", int(output_flag))
        if warm_start:
            sm.setParam("LPWarmStart", 1)
        if threads is not None:
            sm.setParam("Threads", int(threads))
        if method is not None:
            sm.setParam("Method", int(method))

        du = sm.addVars(range(G), range(T), lb=0, name="du")
        dd = sm.addVars(range(G), range(T), lb=0, name="dd")
        ls = sm.addVars(range(T), lb=0, name="ls")
        sp = sm.addVars(range(T), lb=0, name="sp")

        for t in range(T):
            for i in range(G):
                sm.addConstr(du[i, t] <= float(R_up[t, i]), name=f"du_le_Rup_{i}_{t}")
                sm.addConstr(dd[i, t] <= float(R_dn[t, i]), name=f"dd_le_Rdn_{i}_{t}")
            sm.addConstr(
                gp.quicksum(du[i, t] - dd[i, t] for i in range(G)) + ls[t] - sp[t] == float(omega[t]),
                name=f"balance_{t}",
            )

        sm.setObjective(
            gp.quicksum(
                float(price_rt_up[i]) * du[i, t] + float(price_rt_dn[i]) * dd[i, t]
                for i in range(G)
                for t in range(T)
            )
            + voll * gp.quicksum(ls[t] for t in range(T))
            + vosp * gp.quicksum(sp[t] for t in range(T)),
            GRB.MINIMIZE,
        )

        sm.optimize()
        if sm.status != GRB.OPTIMAL:
            raise RuntimeError(f"RT all-T subproblem not optimal, status={sm.status}")

        deploy_cost = float(sum(price_rt_up[i] * du[i, t].X + price_rt_dn[i] * dd[i, t].X
                                for i in range(G) for t in range(T)))
        penalty_cost = float(voll * sum(ls[t].X for t in range(T)) + vosp * sum(sp[t].X for t in range(T)))

        sm.dispose()

        total_cost = stage1_energy_cost + stage1_reserve_cost + deploy_cost + penalty_cost

        return {
            "stage1_energy_cost": float(stage1_energy_cost),
            "stage1_reserve_cost": float(stage1_reserve_cost),
            "rt_deploy_cost": float(deploy_cost),
            "rt_penalty_cost": float(penalty_cost),
            "total_cost": float(total_cost),
            "omega": omega,
            "P_DA": P_DA,
            "R_up": R_up,
            "R_dn": R_dn,
            "Cost_DA": Cost_DA,
        }


def _solve_one_day_single(day, args, data, pred_mean_all, mode=None):
    T = args.T
    sl = slice(T*day, T*day+T)

    scenarios = data["Y_pred"][0:args.N_scen, :, sl].sum(axis=1)          # (S,T)
    forecast_load = pred_mean_all[:, sl].sum(axis=0)          # (T,)
    Y_true = data["Y_true"][:, sl].sum(axis=0)                # (T,)

    if mode == "ideal":
        forecast_load = Y_true.copy()
        scenarios = Y_true.reshape(1, T)

    mgr = IEEE14_Reserve_SO_Manager_SingleNode(args)
    mgr.build_model(forecast_load, scenarios, output_flag=0)

    mgr.solve(threads=1, method=1, warm_start=False, output_flag=0)
    true_eval = mgr.compute_true_cost(Y_true, output_flag=0, threads=1, method=1)
    return true_eval["total_cost"]


def Average_cost_Reserve_SO_SingleNode(args, data, mode=None, n_jobs=None):
    print(args.N_scen)
    T = args.T
    Y_pred = data["Y_pred"]
    S, N, L = Y_pred.shape
    days = L // T

    pred_mean_all = Y_pred.mean(axis=0)  # (N,L)

    if n_jobs is None:
        n_jobs = min(os.cpu_count() or 1, days)

    costs = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_solve_one_day_single)(day, args, data, pred_mean_all, mode)
        for day in range(days)
    )
    return costs


class SingleNode_Reserve_Deterministic_DA_OptNet(torch.nn.Module):
    def __init__(self, mgr, T=24, dtype=torch.float64):
        super().__init__()
        self.T = int(T)
        self.dtype = dtype

        self.gen_ids = list(mgr.gen_bus_list)
        self.gen_info = mgr.gen_info
        self.G = len(self.gen_ids)

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

        forecast = cp.Parameter((self.T,), nonneg=True, name="forecast")

        P = cp.Variable((self.G, self.T), nonneg=True, name="P_DA")
        R_up = cp.Variable((self.G, self.T), nonneg=True, name="R_up")
        R_dn = cp.Variable((self.G, self.T), nonneg=True, name="R_dn")
        Cost = cp.Variable((self.G, self.T), nonneg=True, name="Cost_DA")

        cons = []
        cons += [P >= pmin.reshape(-1, 1)]
        cons += [P <= pmax.reshape(-1, 1)]
        cons += [P + R_up <= pmax.reshape(-1, 1)]
        cons += [P - R_dn >= pmin.reshape(-1, 1)]
        cons += [cp.sum(P, axis=0) == forecast]
        cons += [cp.sum(R_up, axis=0) >= reserve_req_up_ratio * forecast]
        cons += [cp.sum(R_dn, axis=0) >= reserve_req_dn_ratio * forecast]
        cons += [Cost == cp.multiply(b.reshape(-1, 1), P) + c.reshape(-1, 1)]

        # ramp constraints
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

    def forward(self, forecast, omega=None, solver="ECOS", return_rt=False, return_cost=True):
        forecast = forecast.to(dtype=self.dtype)

        if solver == "ECOS":
            solver_args = {"solve_method": "ECOS"}
        elif solver == "SCS":
            solver_args = {"solve_method": "SCS", "max_iters": 200000, "eps": 1e-6}
        else:
            raise ValueError("solver must be 'ECOS' or 'SCS'")

        B = forecast.shape[0]
        outs = {"P_DA": [], "R_up": [], "R_dn": [], "obj": []}
        if return_cost:
            outs["Cost_DA"] = []

        bG = self.b_G
        res_up_price = self.res_up_ratio * bG
        res_dn_price = self.res_dn_ratio * bG

        for bi in range(B):
            P, R_up, R_dn, Cost = self.layer(forecast[bi], solver_args=solver_args)

            stage1_energy = Cost.sum()
            stage1_reserve = (
                (res_up_price.unsqueeze(1) * R_up).sum()
                + (res_dn_price.unsqueeze(1) * R_dn).sum()
            )
            obj = stage1_energy + stage1_reserve

            outs["P_DA"].append(P)
            outs["R_up"].append(R_up)
            outs["R_dn"].append(R_dn)
            outs["obj"].append(obj)
            if return_cost:
                outs["Cost_DA"].append(Cost)

        outs["P_DA"] = torch.stack(outs["P_DA"], dim=0)
        outs["R_up"] = torch.stack(outs["R_up"], dim=0)
        outs["R_dn"] = torch.stack(outs["R_dn"], dim=0)
        outs["obj"] = torch.stack(outs["obj"], dim=0)

        if return_cost:
            outs["Cost_DA"] = torch.stack(outs["Cost_DA"], dim=0)

        return outs


class SingleNode_Reserve_SAA_DA_OptNet(torch.nn.Module):
    """
    Day-ahead SAA OptNet (single node).

    MODIFIED:
      - remove PWL epigraph linearization
      - stage-1 energy cost uses linear term only: Cost = b*P (+c)
      - add DA ramp constraints to match Gurobi manager
    """
    def __init__(self, mgr, N_scen, T=24, dtype=torch.float64):
        super().__init__()
        self.N = int(N_scen)
        self.T = int(T)
        self.dtype = dtype

        self.gen_ids = list(mgr.gen_bus_list)
        self.gen_info = mgr.gen_info
        self.G = len(self.gen_ids)

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

        # torch buffers for objective recompute
        self.register_buffer("b_G", torch.tensor(b, dtype=self.dtype))
        self.register_buffer("voll", torch.tensor(voll, dtype=self.dtype))
        self.register_buffer("vosp", torch.tensor(vosp, dtype=self.dtype))
        self.register_buffer("res_up_ratio", torch.tensor(reserve_up_ratio, dtype=self.dtype))
        self.register_buffer("res_dn_ratio", torch.tensor(reserve_dn_ratio, dtype=self.dtype))
        self.register_buffer("rt_up_ratio", torch.tensor(rt_up_ratio, dtype=self.dtype))
        self.register_buffer("rt_dn_ratio", torch.tensor(rt_dn_ratio, dtype=self.dtype))

        # ---------- CVXPY params ----------
        forecast = cp.Parameter((self.T,), nonneg=True, name="forecast")  # (T,)
        omega = cp.Parameter((self.N, self.T), name="omega")              # (N,T)

        # ---------- Stage-1 vars ----------
        P = cp.Variable((self.G, self.T), nonneg=True, name="P_DA")
        R_up = cp.Variable((self.G, self.T), nonneg=True, name="R_up")
        R_dn = cp.Variable((self.G, self.T), nonneg=True, name="R_dn")
        Cost = cp.Variable((self.G, self.T), nonneg=True, name="Cost_DA")

        # ---------- Stage-2 vars ----------
        d_up_list = [cp.Variable((self.G, self.T), nonneg=True, name=f"d_up_{n}") for n in range(self.N)]
        d_dn_list = [cp.Variable((self.G, self.T), nonneg=True, name=f"d_dn_{n}") for n in range(self.N)]
        LS_list = [cp.Variable((self.T,), nonneg=True, name=f"LS_{n}") for n in range(self.N)]
        SP_list = [cp.Variable((self.T,), nonneg=True, name=f"SP_{n}") for n in range(self.N)]

        cons = []
        cons += [P >= pmin.reshape(-1, 1)]
        cons += [P <= pmax.reshape(-1, 1)]
        cons += [P + R_up <= pmax.reshape(-1, 1)]
        cons += [P - R_dn >= pmin.reshape(-1, 1)]
        cons += [cp.sum(P, axis=0) == forecast]

        # linear DA energy cost only
        cons += [Cost == cp.multiply(b.reshape(-1, 1), P) + c.reshape(-1, 1)]

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

            cons += [d_up <= R_up]
            cons += [d_dn <= R_dn]
            cons += [cp.sum(d_up - d_dn, axis=0) + LS - SP == omega[n, :]]  # (T,)

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

        variables = [P, R_up, R_dn, Cost] + d_up_list + d_dn_list + LS_list + SP_list
        self.layer = CvxpyLayer(prob, parameters=[forecast, omega], variables=variables)

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

    def forward(self, forecast, omega, solver="ECOS", return_rt=True, return_cost=True):
        forecast = forecast.to(dtype=self.dtype)
        omega = omega.to(dtype=self.dtype)

        if solver == "ECOS":
            solver_args = {"solve_method": "ECOS"}
        elif solver == "SCS":
            solver_args = {"solve_method": "SCS", "max_iters": 200000, "eps": 1e-6}
        else:
            raise ValueError("solver must be 'ECOS' or 'SCS'")

        B = forecast.shape[0]
        outs = {"P_DA": [], "R_up": [], "R_dn": [], "obj": []}
        if return_cost:
            outs["Cost_DA"] = []
        if return_rt:
            outs.update({"d_up": [], "d_dn": [], "LS": [], "SP": []})

        bG = self.b_G
        res_up_price = self.res_up_ratio * bG
        res_dn_price = self.res_dn_ratio * bG
        rt_up_price = self.rt_up_ratio * bG
        rt_dn_price = self.rt_dn_ratio * bG

        for bi in range(B):
            sol = self.layer(forecast[bi], omega[bi], solver_args=solver_args)

            P = sol[self._idx["P"]]         # (G,T)
            R_up = sol[self._idx["R_up"]]   # (G,T)
            R_dn = sol[self._idx["R_dn"]]   # (G,T)
            Cost = sol[self._idx["Cost"]]   # (G,T)

            d_up_list = sol[self._idx["d_up_start"]: self._idx["d_up_start"] + self.N]
            d_dn_list = sol[self._idx["d_dn_start"]: self._idx["d_dn_start"] + self.N]
            LS_list = sol[self._idx["LS_start"]: self._idx["LS_start"] + self.N]
            SP_list = sol[self._idx["SP_start"]: self._idx["SP_start"] + self.N]

            d_up = torch.stack(d_up_list, dim=0)  # (N,G,T)
            d_dn = torch.stack(d_dn_list, dim=0)  # (N,G,T)
            LS = torch.stack(LS_list, dim=0)      # (N,T)
            SP = torch.stack(SP_list, dim=0)      # (N,T)

            stage1_energy = Cost.sum()
            stage1_reserve = (
                (res_up_price.unsqueeze(1) * R_up).sum()
                + (res_dn_price.unsqueeze(1) * R_dn).sum()
            )
            rt_deploy = (
                (rt_up_price.unsqueeze(0).unsqueeze(-1) * d_up).sum()
                + (rt_dn_price.unsqueeze(0).unsqueeze(-1) * d_dn).sum()
            )
            rt_pen = self.voll * LS.sum() + self.vosp * SP.sum()
            obj = stage1_energy + stage1_reserve + (rt_deploy + rt_pen) / float(self.N)

            outs["P_DA"].append(P)
            outs["R_up"].append(R_up)
            outs["R_dn"].append(R_dn)
            outs["obj"].append(obj)

            if return_cost:
                outs["Cost_DA"].append(Cost)

            if return_rt:
                outs["d_up"].append(d_up)
                outs["d_dn"].append(d_dn)
                outs["LS"].append(LS)
                outs["SP"].append(SP)

        outs["P_DA"] = torch.stack(outs["P_DA"], dim=0)  # (B,G,T)
        outs["R_up"] = torch.stack(outs["R_up"], dim=0)
        outs["R_dn"] = torch.stack(outs["R_dn"], dim=0)
        outs["obj"] = torch.stack(outs["obj"], dim=0)    # (B,)

        if return_cost:
            outs["Cost_DA"] = torch.stack(outs["Cost_DA"], dim=0)  # (B,G,T)

        if return_rt:
            outs["d_up"] = torch.stack(outs["d_up"], dim=0)  # (B,N,G,T)
            outs["d_dn"] = torch.stack(outs["d_dn"], dim=0)
            outs["LS"] = torch.stack(outs["LS"], dim=0)      # (B,N,T)
            outs["SP"] = torch.stack(outs["SP"], dim=0)

        return outs

class SingleNode_Reserve_DRO_DA_OptNet(torch.nn.Module):
    """
    Single-node DRO DA OptNet.

    MODIFIED:
      - add DA ramp constraints to match Gurobi manager
      - reserve requirement remains unchanged (not added on purpose)
    """
    def __init__(self, mgr, N_scen, T=24, dtype=torch.float64):
        super().__init__()
        self.N = int(N_scen)
        self.T = int(T)
        self.dtype = dtype

        self.gen_ids = list(mgr.gen_bus_list)
        self.gen_info = mgr.gen_info
        self.G = len(self.gen_ids)

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

        # torch buffers
        self.register_buffer("b_G", torch.tensor(b, dtype=self.dtype))
        self.register_buffer("voll", torch.tensor(voll, dtype=self.dtype))
        self.register_buffer("vosp", torch.tensor(vosp, dtype=self.dtype))
        self.register_buffer("res_up_ratio", torch.tensor(reserve_up_ratio, dtype=self.dtype))
        self.register_buffer("res_dn_ratio", torch.tensor(reserve_dn_ratio, dtype=self.dtype))
        self.register_buffer("rt_up_ratio", torch.tensor(rt_up_ratio, dtype=self.dtype))
        self.register_buffer("rt_dn_ratio", torch.tensor(rt_dn_ratio, dtype=self.dtype))

        # -------- CVXPY params --------
        forecast = cp.Parameter((self.T,), nonneg=True, name="forecast")
        omega = cp.Parameter((self.N, self.T), name="omega")
        om_min = cp.Parameter((self.T,), name="om_min")
        om_max = cp.Parameter((self.T,), name="om_max")
        eps = cp.Parameter(nonneg=True, name="eps")

        # -------- Stage-1 vars --------
        P = cp.Variable((self.G, self.T), nonneg=True, name="P_DA")
        R_up = cp.Variable((self.G, self.T), nonneg=True, name="R_up")
        R_dn = cp.Variable((self.G, self.T), nonneg=True, name="R_dn")
        Cost = cp.Variable((self.G, self.T), nonneg=True, name="Cost_DA")

        # -------- Stage-2 per-sample recourse --------
        d_up_list = [cp.Variable((self.G, self.T), nonneg=True, name=f"d_up_{n}") for n in range(self.N)]
        d_dn_list = [cp.Variable((self.G, self.T), nonneg=True, name=f"d_dn_{n}") for n in range(self.N)]
        LS_list = [cp.Variable((self.T,), nonneg=True, name=f"LS_{n}") for n in range(self.N)]
        SP_list = [cp.Variable((self.T,), nonneg=True, name=f"SP_{n}") for n in range(self.N)]

        # -------- Stage-2 extreme recourse --------
        du_max = cp.Variable((self.G, self.T), nonneg=True, name="du_max")
        dd_max = cp.Variable((self.G, self.T), nonneg=True, name="dd_max")
        LS_max = cp.Variable((self.T,), nonneg=True, name="LS_max")
        SP_max = cp.Variable((self.T,), nonneg=True, name="SP_max")

        du_min = cp.Variable((self.G, self.T), nonneg=True, name="du_min")
        dd_min = cp.Variable((self.G, self.T), nonneg=True, name="dd_min")
        LS_min = cp.Variable((self.T,), nonneg=True, name="LS_min")
        SP_min = cp.Variable((self.T,), nonneg=True, name="SP_min")

        # -------- DRO vars --------
        phi = cp.Variable((self.N,), name="phi")   # free
        gamma = cp.Variable(nonneg=True, name="gamma")

        # -------- constraints --------
        cons = []
        cons += [P >= pmin.reshape(-1, 1)]
        cons += [P <= pmax.reshape(-1, 1)]
        cons += [P + R_up <= pmax.reshape(-1, 1)]
        cons += [P - R_dn >= pmin.reshape(-1, 1)]
        cons += [cp.sum(P, axis=0) == forecast]
        cons += [Cost == cp.multiply(b.reshape(-1, 1), P) + c.reshape(-1, 1)]

        # DA ramp constraints
        if self.T >= 2:
            cons += [P[:, 1:] - P[:, :-1] <= ramp.reshape(-1, 1)]
            cons += [P[:, :-1] - P[:, 1:] <= ramp.reshape(-1, 1)]

        # sample feasibility
        for n in range(self.N):
            cons += [d_up_list[n] <= R_up]
            cons += [d_dn_list[n] <= R_dn]
            cons += [cp.sum(d_up_list[n] - d_dn_list[n], axis=0) + LS_list[n] - SP_list[n] == omega[n, :]]

        # extreme feasibility
        cons += [du_max <= R_up, dd_max <= R_dn]
        cons += [du_min <= R_up, dd_min <= R_dn]
        cons += [cp.sum(du_max - dd_max, axis=0) + LS_max - SP_max == om_max]
        cons += [cp.sum(du_min - dd_min, axis=0) + LS_min - SP_min == om_min]

        # -------- costs --------
        res_up_price = reserve_up_ratio * b
        res_dn_price = reserve_dn_ratio * b
        rt_up_price = rt_up_ratio * b
        rt_dn_price = rt_dn_ratio * b

        stage1_energy = cp.sum(Cost)
        stage1_reserve = (
            cp.sum(cp.multiply(res_up_price.reshape(-1, 1), R_up)) +
            cp.sum(cp.multiply(res_dn_price.reshape(-1, 1), R_dn))
        )

        # Q(omega_n)
        Qn = []
        for n in range(self.N):
            Qn.append(
                cp.sum(cp.multiply(rt_up_price.reshape(-1, 1), d_up_list[n])) +
                cp.sum(cp.multiply(rt_dn_price.reshape(-1, 1), d_dn_list[n])) +
                voll * cp.sum(LS_list[n]) +
                vosp * cp.sum(SP_list[n])
            )

        Q_max = (
            cp.sum(cp.multiply(rt_up_price.reshape(-1, 1), du_max)) +
            cp.sum(cp.multiply(rt_dn_price.reshape(-1, 1), dd_max)) +
            voll * cp.sum(LS_max) +
            vosp * cp.sum(SP_max)
        )
        Q_min = (
            cp.sum(cp.multiply(rt_up_price.reshape(-1, 1), du_min)) +
            cp.sum(cp.multiply(rt_dn_price.reshape(-1, 1), dd_min)) +
            voll * cp.sum(LS_min) +
            vosp * cp.sum(SP_min)
        )

        # -------- DRO constraints --------
        ones_T = np.ones(self.T)
        for n in range(self.N):
            dist_max_n = ones_T @ (om_max - omega[n, :])
            dist_min_n = ones_T @ (omega[n, :] - om_min)

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
            parameters=[forecast, omega, om_min, om_max, eps],
            variables=variables
        )

        self._idx = {
            "P": 0,
            "R_up": 1,
            "R_dn": 2,
            "Cost": 3,
            "phi": 4,
            "gamma": 5,
            "d_up_start": 6,
            "d_dn_start": 6 + self.N,
            "LS_start": 6 + 2 * self.N,
            "SP_start": 6 + 3 * self.N,
        }

    def forward(self, forecast, omega_scen, om_min, om_max, eps, solver="ECOS"):
        forecast = forecast.to(dtype=self.dtype)
        omega_scen = omega_scen.to(dtype=self.dtype)
        om_min = om_min.to(dtype=self.dtype)
        om_max = om_max.to(dtype=self.dtype)
        eps = eps.to(dtype=self.dtype)

        if solver == "ECOS":
            solver_args = {"solve_method": "ECOS"}
        elif solver == "SCS":
            solver_args = {"solve_method": "SCS", "max_iters": 200000, "eps": 1e-6}
        else:
            raise ValueError("solver must be 'ECOS' or 'SCS'")

        B = forecast.shape[0]
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

        for bi in range(B):
            fb = forecast[bi]
            ob = omega_scen[bi]
            omin_b = om_min[bi] if om_min.ndim == 2 else om_min
            omax_b = om_max[bi] if om_max.ndim == 2 else om_max
            eps_b = eps[bi] if eps.ndim > 0 else eps

            sol = self.layer(fb, ob, omin_b, omax_b, eps_b, solver_args=solver_args)

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
            obj = stage1_energy + stage1_reserve + phi.mean() + eps_b * gamma

            outs["P_DA"].append(P)
            outs["R_up"].append(R_up)
            outs["R_dn"].append(R_dn)
            outs["Cost_DA"].append(Cost)
            outs["phi"].append(phi)
            outs["gamma"].append(gamma)
            outs["obj"].append(obj)

        return {k: torch.stack(v, dim=0) for k, v in outs.items()}


class SingleNode_Reserve_RT_OptNet(torch.nn.Module):
    """
    Real-time recourse OptNet (true intraday).
    """
    def __init__(self, mgr, T=24, dtype=torch.float64):
        super().__init__()
        self.T = int(T)
        self.dtype = dtype

        self.gen_ids = list(mgr.gen_bus_list)
        self.gen_info = mgr.gen_info
        self.G = len(self.gen_ids)

        rt_up_ratio = mgr.args.rt_up_ratio
        rt_dn_ratio = mgr.args.rt_dn_ratio
        voll = mgr.args.voll
        vosp = mgr.args.vosp

        b = np.array([self.gen_info[g]["b"] for g in self.gen_ids], dtype=float)

        self.register_buffer("b_G", torch.tensor(b, dtype=self.dtype))
        self.register_buffer("voll", torch.tensor(voll, dtype=self.dtype))
        self.register_buffer("vosp", torch.tensor(vosp, dtype=self.dtype))
        self.register_buffer("rt_up_ratio", torch.tensor(rt_up_ratio, dtype=self.dtype))
        self.register_buffer("rt_dn_ratio", torch.tensor(rt_dn_ratio, dtype=self.dtype))

        # CVXPY params
        Rup = cp.Parameter((self.G, self.T), nonneg=True, name="R_up")
        Rdn = cp.Parameter((self.G, self.T), nonneg=True, name="R_dn")
        omega = cp.Parameter((self.T,), name="omega_true")

        # vars
        du = cp.Variable((self.G, self.T), nonneg=True, name="d_up")
        dd = cp.Variable((self.G, self.T), nonneg=True, name="d_dn")
        LS = cp.Variable((self.T,), nonneg=True, name="LS")
        SP = cp.Variable((self.T,), nonneg=True, name="SP")

        cons = []
        cons += [du <= Rup]
        cons += [dd <= Rdn]
        cons += [cp.sum(du - dd, axis=0) + LS - SP == omega]  # (T,)

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

    def forward(self, R_up, R_dn, omega_true, solver="ECOS"):
        R_up = R_up.to(dtype=self.dtype)               # (B,G,T)
        R_dn = R_dn.to(dtype=self.dtype)               # (B,G,T)
        omega_true = omega_true.to(dtype=self.dtype)   # (B,T)

        if solver == "ECOS":
            solver_args = {"solve_method": "ECOS"}
        elif solver == "SCS":
            solver_args = {"solve_method": "SCS", "max_iters": 10000, "eps": 1e-6}
        else:
            raise ValueError("solver must be 'ECOS' or 'SCS'")

        B = omega_true.shape[0]
        du_out, dd_out, ls_out, sp_out, obj_out = [], [], [], [], []

        rt_up_price = self.rt_up_ratio * self.b_G      # (G,)
        rt_dn_price = self.rt_dn_ratio * self.b_G      # (G,)

        for b in range(B):
            du, dd, LS, SP = self.layer(R_up[b], R_dn[b], omega_true[b], solver_args=solver_args)

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
            "d_up": torch.stack(du_out, dim=0),     # (B,G,T)
            "d_dn": torch.stack(dd_out, dim=0),     # (B,G,T)
            "LS": torch.stack(ls_out, dim=0),       # (B,T)
            "SP": torch.stack(sp_out, dim=0),       # (B,T)
            "rt_obj": torch.stack(obj_out, dim=0),  # (B,)
        }


def compute_true_cost_optnet(da_out, rt_layer, y_true, forecast, da_layer):
    """
    对齐 Gurobi compute_true_cost 的口径：
      total = stage1_energy_cost_pwl(Cost_DA) + stage1_reserve + rt_deploy + rt_penalty
    """
    device = forecast.device
    dtype = forecast.dtype

    if isinstance(y_true, np.ndarray):
        y_true = torch.tensor(y_true, device=device, dtype=dtype)
    if y_true.ndim == 1:
        y_true = y_true.unsqueeze(0)  # (B,T)

    omega_true = y_true - forecast  # (B,T)

    if "Cost_DA" not in da_out:
        raise KeyError("da_out must contain 'Cost_DA' (PWL epigraph cost).")
    stage1_energy_cost_pwl = da_out["Cost_DA"].sum(dim=(1, 2))  # (B,)

    bG = da_layer.b_G
    price_res_up = da_layer.res_up_ratio * bG
    price_res_dn = da_layer.res_dn_ratio * bG
    stage1_reserve_cost = (
        (price_res_up[None, :, None] * da_out["R_up"]).sum(dim=(1, 2))
        + (price_res_dn[None, :, None] * da_out["R_dn"]).sum(dim=(1, 2))
    )  # (B,)

    rt_out = rt_layer(da_out["R_up"], da_out["R_dn"], omega_true)

    price_rt_up = da_layer.rt_up_ratio * bG
    price_rt_dn = da_layer.rt_dn_ratio * bG
    rt_deploy_cost = (
        (price_rt_up[None, :, None] * rt_out["d_up"]).sum(dim=(1, 2))
        + (price_rt_dn[None, :, None] * rt_out["d_dn"]).sum(dim=(1, 2))
    )  # (B,)

    rt_penalty_cost = da_layer.voll * rt_out["LS"].sum(dim=1) + da_layer.vosp * rt_out["SP"].sum(dim=1)  # (B,)

    total_cost = stage1_energy_cost_pwl + stage1_reserve_cost + rt_deploy_cost + rt_penalty_cost

    return {
        "stage1_energy_cost_pwl": stage1_energy_cost_pwl,
        "stage1_reserve_cost": stage1_reserve_cost,
        "rt_deploy_cost": rt_deploy_cost,
        "rt_penalty_cost": rt_penalty_cost,
        "total_cost": total_cost,
        "omega": omega_true,
        "deploy_up": rt_out["d_up"],
        "deploy_dn": rt_out["d_dn"],
        "LS": rt_out["LS"],
        "SP": rt_out["SP"],
    }


def _solve_one_day_single_det(day, args, data, pred_mean_all):
    T = args.T
    sl = slice(T * day, T * day + T)

    forecast_load = pred_mean_all[:, sl].sum(axis=0)   # (T,)
    Y_true = data["Y_true"][:, sl].sum(axis=0)         # (T,)

    mgr = IEEE14_Reserve_Deterministic_DA_Manager_SingleNode(args)
    mgr.build_model(forecast_load, output_flag=0)
    mgr.solve(threads=1, method=1, warm_start=False, output_flag=0)

    true_eval = mgr.compute_true_cost(Y_true, output_flag=0, threads=1, method=1)
    return true_eval["total_cost"]

def Average_cost_Reserve_DET_SingleNode(args, data, n_jobs=None):
    T = args.T
    Y_pred = data["Y_pred"]
    S, N, L = Y_pred.shape
    days = L // T

    pred_mean_all = Y_pred.mean(axis=0)  # (N,L)

    if n_jobs is None:
        n_jobs = min(os.cpu_count() or 1, days)

    costs = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_solve_one_day_single_det)(day, args, data, pred_mean_all)
        for day in range(days)
    )
    return costs

def _solve_one_day_single_ideal(day, args, data):
    T = args.T
    sl = slice(T * day, T * day + T)

    Y_true = data["Y_true"][:, sl].sum(axis=0)   # (T,)

    mgr = IEEE14_Ideal_Manager_SingleNode(args)
    mgr.build_model(Y_true, output_flag=0)
    mgr.solve(threads=1, method=1, warm_start=False, output_flag=0)

    true_eval = mgr.compute_true_cost()
    return true_eval["total_cost"]

def Average_cost_Reserve_IDEAL_SingleNode(args, data, n_jobs=None):
    T = args.T
    Y_pred = data["Y_pred"]
    _, _, L = Y_pred.shape
    days = L // T

    if n_jobs is None:
        n_jobs = min(os.cpu_count() or 1, days)

    costs = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_solve_one_day_single_ideal)(day, args, data)
        for day in range(days)
    )
    return costs



def _solve_one_day_single(day, args, data, pred_mean_all, mode=None):
    T = args.T
    sl = slice(T*day, T*day+T)

    scenarios = data["Y_pred"][0:args.N_scen, :, sl].sum(axis=1)          # (S,T)
    forecast_load = pred_mean_all[:, sl].sum(axis=0)          # (T,)
    Y_true = data["Y_true"][:, sl].sum(axis=0)                # (T,)

    if mode == "ideal":
        forecast_load = Y_true.copy()
        scenarios = Y_true.reshape(1, T)

    mgr = IEEE14_Reserve_SO_Manager_SingleNode(args)
    mgr.build_model(forecast_load, scenarios, output_flag=0)

    mgr.solve(threads=1, method=1, warm_start=False, output_flag=0)
    true_eval = mgr.compute_true_cost(Y_true, output_flag=0, threads=1, method=1)
    return true_eval["total_cost"]

def Average_cost_Reserve_SO_SingleNode(args, data, mode=None, n_jobs=None):
    T = args.T
    Y_pred = data["Y_pred"]
    S, N, L = Y_pred.shape
    days = L // T

    pred_mean_all = Y_pred.mean(axis=0)  # (N,L)

    if n_jobs is None:
        n_jobs = min(os.cpu_count() or 1, days)

    costs = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_solve_one_day_single)(day, args, data, pred_mean_all, mode)
        for day in range(days)
    )
    return costs


def _solve_one_day_single_dro(day, args, data, pred_mean_all, Lmin_11, Lmax_11, eps, mode=None):
    T = args.T
    sl = slice(T * day, T * day + T)

    # data["Y_pred"]: (S, N, L)
    scen_sys = np.asarray(data["Y_pred"][0:args.N_scen, :, sl], dtype=float).sum(axis=1)   # (S,T)
    forecast_sys = np.asarray(pred_mean_all[:, sl], dtype=float).sum(axis=0)               # (T,)
    actual_sys = np.asarray(data["Y_true"][:, sl], dtype=float).sum(axis=0)                # (T,)

    # convert nodal min/max to system min/max
    Lmin_sys = np.asarray(Lmin_11, dtype=float).sum(axis=0)   # (T,)
    Lmax_sys = np.asarray(Lmax_11, dtype=float).sum(axis=0)   # (T,)

    if mode == "ideal":
        forecast_sys = actual_sys.copy()
        scen_sys = np.repeat(actual_sys.reshape(1, T), scen_sys.shape[0], axis=0)   # (S,T)

    mgr = IEEE14_Reserve_DRO_Manager_SingleNode(args)
    mgr.build_model(
        forecast=forecast_sys,         # (T,)
        scenarios=scen_sys,            # (S,T)
        eps=float(eps),
        hourly_load_min=Lmin_sys,      # (T,)
        hourly_load_max=Lmax_sys,      # (T,)
        output_flag=0,
    )
    mgr.solve(output_flag=0)

    res = mgr.compute_true_cost(actual_sys, mu=forecast_sys, output_flag=0)

    out = {
        "total_cost": float(res["total_cost"]),
        "train_obj": float(mgr.model.ObjVal),
        "gamma": float(mgr.vars["gamma"].X) if "gamma" in getattr(mgr, "vars", {}) else None,
    }
    return out

def Average_cost_Reserve_DRO_SingleNode(args, data, Lmin_11, Lmax_11, eps, mode=None, n_jobs=None, return_full=False):
    T = args.T
    Y_pred = data["Y_pred"]
    S, N, L = Y_pred.shape
    days = L // T

    pred_mean_all = Y_pred.mean(axis=0)  # (N,L)

    if n_jobs is None:
        n_jobs = min(os.cpu_count() or 1, days)

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_solve_one_day_single_dro)(day, args, data, pred_mean_all, Lmin_11, Lmax_11, eps, mode)
        for day in range(days)
    )

    if return_full:
        return results
    else:
        return [r["total_cost"] for r in results]
