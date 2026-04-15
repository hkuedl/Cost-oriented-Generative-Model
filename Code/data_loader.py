import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler,MinMaxScaler

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, x):
        x = np.asarray(x, dtype=np.float32)
        self.mean_ = x.mean(axis=0, keepdims=True)
        self.std_ = x.std(axis=0, keepdims=True) + 1e-8
        self.scale_ = self.std_ 
        return self

    def transform(self, x):
        x = np.asarray(x, dtype=np.float32)
        return (x - self.mean_) / self.std_

    def inverse_transform(self, x):
        x = np.asarray(x, dtype=np.float32)
        return x * self.std_ + self.mean_


class Dataset_load_single_node_parametric(Dataset):
    """
    From a wide CSV (many node columns), pick one node_col and build day-level samples:
      X: [D, 24, F]
      y: [D, 24]

    Features:
      month, weekday, hour, TEMP, MAX, MIN, load_1_day_before (= node_col shifted by 24)
    Scaling:
      scalers fit on train_pool only (first train_length rows AFTER dropna),
      then applied to all splits.
    Split:
      train_pool -> random train/val by day
      test       -> rows after train_length
    """

    def __init__(
        self,
        data_path,
        node_col,
        flag="train",
        train_length=4296,      # hour-rows split point
        val_ratio=0.2,
        scale=True,
        seed=42,
        features= [
            "month", "weekday", "hour",
            "TEMP", "MAX", "MIN",
            "load_1_day_before",
        ]
    ):
        self.data_path = data_path
        self.node_col = node_col
        self.flag = flag
        self.train_length = int(train_length)
        self.val_ratio = float(val_ratio)
        self.scale = bool(scale)
        self.seed = int(seed)

        # self.time_col = time_col
        # self.temp_col = temp_col
        # self.max_col = max_col
        # self.min_col = min_col

        self.month_col = "month"
        self.weekday_col = "weekday"
        self.hour_col = "hour"

        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.features = features
        self.X, self.y = self._process()

    def _process(self):
        df = pd.read_csv(self.data_path)
        df = df.interpolate(method="cubic", limit_direction="both")

        if self.node_col not in df.columns:
            raise ValueError(f"node_col={self.node_col} not found in columns")

        dt = pd.to_datetime(df["DATETIME"])
        df["month"] = dt.dt.month
        df[ "weekday"] = dt.dt.weekday
        df["hour"] = dt.dt.hour

        df["load_1_day_before"] = df[self.node_col].shift(24)

        X_cols = self.features
        y_cols = [self.node_col]

        data = df[X_cols + y_cols].dropna().copy()

        train_pool = data.iloc[: self.train_length].copy()
        test_data = data.iloc[self.train_length:].copy()

        X_train_pool = train_pool[X_cols].values
        y_train_pool = train_pool[y_cols].values  # [rows,1]
        X_test = test_data[X_cols].values
        y_test = test_data[y_cols].values

        if self.scale:
            self.scaler_x.fit(X_train_pool)
            self.scaler_y.fit(y_train_pool)
            X_train_pool = self.scaler_x.transform(X_train_pool)
            y_train_pool = self.scaler_y.transform(y_train_pool)
            X_test = self.scaler_x.transform(X_test)
            y_test = self.scaler_y.transform(y_test)

        F = X_train_pool.shape[1]
        X_train_pool = X_train_pool.reshape(-1, 24, F)
        y_train_pool = y_train_pool.reshape(-1, 24)
        X_test = X_test.reshape(-1, 24, F)
        y_test = y_test.reshape(-1, 24)

        n_days = X_train_pool.shape[0]
        rng = np.random.RandomState(self.seed)
        perm = rng.permutation(n_days)
        n_val = int(round(n_days * self.val_ratio))
        val_idx = perm[:n_val]
        tr_idx = perm[n_val:]

        X_train, y_train = X_train_pool[tr_idx], y_train_pool[tr_idx]
        X_val, y_val = X_train_pool[val_idx], y_train_pool[val_idx]

        if self.flag == "train":
            X, y = X_train, y_train
        elif self.flag == "val":
            X, y = X_val, y_val
        elif self.flag == "test":
            X, y = X_test, y_test
        else:
            raise ValueError("flag must be one of: 'train','val','test'")

        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return self.X.shape[0]

    def inverse_transform(self, y_nor):
        if torch.is_tensor(y_nor):
            y_np = y_nor.detach().cpu().numpy()
        else:
            y_np = np.asarray(y_nor)
        if y_np.ndim == 1:
            return self.scaler_y.inverse_transform(y_np.reshape(-1, 1)).reshape(-1)
        return self.scaler_y.inverse_transform(y_np.reshape(-1, 1)).reshape(y_np.shape[0], y_np.shape[1])



class Dataset_load_single_node_non_parametric(Dataset):
    """
    Hour-level samples (one timestamp per item):
      X: [F]
      y: scalar

    Features (default):
      month, weekday, hour, TEMP, MAX, MIN, load_1_day_before (= node_col shifted by 24)

    Split logic stays consistent with your parametric version:
      - build 'data' after dropna
      - first train_length rows => train_pool (fit scalers here)
      - remaining rows => test
      - within train_pool: random split by DAY (not by hour) to avoid leakage
    """
    def __init__(
        self,
        data_path,
        node_col,
        flag="train",
        train_length=4296,
        val_ratio=0.2,
        scale=True,
        seed=42,
        features=None,
    ):
        self.data_path = data_path
        self.node_col = node_col
        self.flag = flag
        self.train_length = int(train_length)
        self.val_ratio = float(val_ratio)
        self.scale = bool(scale)
        self.seed = int(seed)

        self.features = features or [
            "month", "weekday", "hour",
            "TEMP", "MAX", "MIN",
            "load_1_day_before",
        ]

        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()

        # tensors:
        self.X, self.y = self._process()

    def _process(self):
        df = pd.read_csv(self.data_path)
        df = df.interpolate(method="cubic", limit_direction="both")

        if self.node_col not in df.columns:
            raise ValueError(f"node_col={self.node_col} not found in columns")

        dt = pd.to_datetime(df["DATETIME"])
        df["month"] = dt.dt.month
        df["weekday"] = dt.dt.weekday
        df["hour"] = dt.dt.hour

        df["load_1_day_before"] = df[self.node_col].shift(24)

        X_cols = list(self.features)
        y_cols = [self.node_col]

        data = df[X_cols + y_cols].dropna().copy()

        # hour-row split
        train_pool = data.iloc[: self.train_length].copy()
        test_data  = data.iloc[self.train_length :].copy()

        # DAY-wise split indices within train_pool (avoid hour leakage)
        n_pool_rows = len(train_pool)
        n_pool_days = n_pool_rows // 24
        if n_pool_days <= 1:
            raise ValueError("train_pool too short to split by days.")

        rng = np.random.RandomState(self.seed)
        perm_days = rng.permutation(n_pool_days)
        n_val_days = int(round(n_pool_days * self.val_ratio))
        val_days = set(perm_days[:n_val_days].tolist())

        def day_mask_for_rows(n_rows):
            days = np.arange(n_rows) // 24
            return days

        pool_days_arr = day_mask_for_rows(n_pool_rows)
        val_mask = np.array([d in val_days for d in pool_days_arr], dtype=bool)
        tr_mask  = ~val_mask

        train_rows = train_pool.iloc[tr_mask].copy()
        val_rows   = train_pool.iloc[val_mask].copy()

        # fit scalers on TRAIN ONLY (not full train_pool), to be strict
        X_train = train_rows[X_cols].values
        y_train = train_rows[y_cols].values  # [n,1]

        X_val = val_rows[X_cols].values
        y_val = val_rows[y_cols].values
        X_test = test_data[X_cols].values
        y_test = test_data[y_cols].values

        if self.scale:
            self.scaler_x.fit(X_train)
            self.scaler_y.fit(y_train)
            X_train = self.scaler_x.transform(X_train)
            y_train = self.scaler_y.transform(y_train)
            X_val = self.scaler_x.transform(X_val)
            y_val = self.scaler_y.transform(y_val)
            X_test = self.scaler_x.transform(X_test)
            y_test = self.scaler_y.transform(y_test)

        if self.flag == "train":
            X, y = X_train, y_train
        elif self.flag == "val":
            X, y = X_val, y_val
        elif self.flag == "test":
            X, y = X_test, y_test
        else:
            raise ValueError("flag must be one of: 'train','val','test'")

        X = torch.tensor(X, dtype=torch.float32)            # [N,F]
        y = torch.tensor(y.reshape(-1), dtype=torch.float32)  # [N]
        return X, y

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return self.X.shape[0]

    def inverse_transform_y(self, y_nor):
        # y_nor: [N] or scalar
        if torch.is_tensor(y_nor):
            y_np = y_nor.detach().cpu().numpy()
        else:
            y_np = np.asarray(y_nor)
        y_np = y_np.reshape(-1, 1)
        return self.scaler_y.inverse_transform(y_np).reshape(-1)

class Dataset_load_multi_node_vae(Dataset):
    """
    Align with Dataset_load_single_node_parametric:
      X features: month, weekday, hour, TEMP, MAX, MIN,
                  + load_1_day_before_<node> for each node
      Y: [D,24,N]
      scaler_x: one scaler on X (fit on train_pool)
      scaler_y: per-node scalers (fit on train_pool)
    """
    def __init__(
        self,
        data_path,
        node_cols,
        flag="train",
        train_length=4296,
        val_ratio=0.2,
        scale=True,
        seed=42,
        datetime_col="DATETIME",
        base_features=("month", "weekday", "hour", "TEMP", "MAX", "MIN"),
        add_lag1=True,
        interpolate=True,
        interpolate_method="cubic",
    ):
        self.data_path = str(data_path)
        self.node_cols = list(node_cols)
        self.N = len(self.node_cols)

        self.flag = str(flag)
        self.train_length = int(train_length)
        self.val_ratio = float(val_ratio)
        self.scale = bool(scale)
        self.seed = int(seed)

        self.datetime_col = str(datetime_col)
        self.base_features = list(base_features)
        self.add_lag1 = bool(add_lag1)

        self.interpolate = bool(interpolate)
        self.interpolate_method = str(interpolate_method)

        self.scaler_x = StandardScaler()
        self.scalers_y = [StandardScaler() for _ in range(self.N)]

        self.X, self.y = self._process()

    def _process(self):
        df = pd.read_csv(self.data_path)
        if self.interpolate:
            df = df.interpolate(method=self.interpolate_method, limit_direction="both")

        if self.datetime_col not in df.columns:
            raise ValueError(f"{self.datetime_col} not found in CSV")
        for c in self.node_cols:
            if c not in df.columns:
                raise ValueError(f"node col {c} not found in CSV")

        dt = pd.to_datetime(df[self.datetime_col])
        df["month"] = dt.dt.month
        df["weekday"] = dt.dt.weekday
        df["hour"] = dt.dt.hour

        lag_cols = []
        if self.add_lag1:
            for c in self.node_cols:
                lc = f"load_1_day_before_{c}"
                df[lc] = df[c].shift(24)
                lag_cols.append(lc)

        X_cols = self.base_features + lag_cols
        Y_cols = self.node_cols

        data = df[X_cols + Y_cols].dropna().copy()

        # hour-row split
        train_pool = data.iloc[: self.train_length].copy()
        test_data = data.iloc[self.train_length :].copy()

        X_train_pool = train_pool[X_cols].values.astype(np.float32)  # [rows,F]
        Y_train_pool = train_pool[Y_cols].values.astype(np.float32)  # [rows,N]
        X_test = test_data[X_cols].values.astype(np.float32)
        Y_test = test_data[Y_cols].values.astype(np.float32)

        if self.scale:
            self.scaler_x.fit(X_train_pool)
            X_train_pool = self.scaler_x.transform(X_train_pool).astype(np.float32)
            X_test = self.scaler_x.transform(X_test).astype(np.float32)

            Y_train_scaled = np.zeros_like(Y_train_pool, dtype=np.float32)
            Y_test_scaled = np.zeros_like(Y_test, dtype=np.float32)
            for j in range(self.N):
                self.scalers_y[j].fit(Y_train_pool[:, j].reshape(-1, 1))
                Y_train_scaled[:, j] = self.scalers_y[j].transform(Y_train_pool[:, j].reshape(-1, 1)).reshape(-1)
                Y_test_scaled[:, j] = self.scalers_y[j].transform(Y_test[:, j].reshape(-1, 1)).reshape(-1)
            Y_train_pool, Y_test = Y_train_scaled, Y_test_scaled

        # reshape into days
        Fdim = X_train_pool.shape[1]
        X_train_pool = X_train_pool.reshape(-1, 24, Fdim)     # [D,24,F]
        Y_train_pool = Y_train_pool.reshape(-1, 24, self.N)   # [D,24,N]
        X_test = X_test.reshape(-1, 24, Fdim)
        Y_test = Y_test.reshape(-1, 24, self.N)

        # split train/val by day
        n_days = X_train_pool.shape[0]
        rng = np.random.RandomState(self.seed)
        perm = rng.permutation(n_days)
        n_val = int(round(n_days * self.val_ratio))
        val_idx = perm[:n_val]
        tr_idx = perm[n_val:]

        X_train, Y_train = X_train_pool[tr_idx], Y_train_pool[tr_idx]
        X_val, Y_val = X_train_pool[val_idx], Y_train_pool[val_idx]

        if self.flag == "train":
            X, Y = X_train, Y_train
        elif self.flag == "val":
            X, Y = X_val, Y_val
        elif self.flag == "test":
            X, Y = X_test, Y_test
        else:
            raise ValueError("flag must be train/val/test")

        return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]  # [24,F], [24,N]

    def __len__(self):
        return self.X.shape[0]

    def inverse_transform_y(self, y_norm):
        """
        supports: [24,N], [D,24,N], [S,D,24,N], ...
        returns numpy
        """
        if torch.is_tensor(y_norm):
            y_np = y_norm.detach().cpu().numpy()
        else:
            y_np = np.asarray(y_norm)

        if not self.scale:
            return y_np

        orig = y_np.shape
        y2 = y_np.reshape(-1, self.N)
        out = np.zeros_like(y2, dtype=np.float32)
        for j in range(self.N):
            out[:, j] = self.scalers_y[j].inverse_transform(y2[:, j].reshape(-1, 1)).reshape(-1)
        return out.reshape(orig)

    # compatibility alias
    def inverse_transform(self, y_norm):
        return self.inverse_transform_y(y_norm)


class Dataset_load_single_node_diff(Dataset):
    """
    单节点 day-level:
      X: [D,24,F]
      y: [D,24]

    一个类同时持有 train/val/test 切分：
      - scaler 只在 train_pool 上 fit 一次
      - val/test 用同一 scaler transform
      - flag 只负责切换使用哪份数据
    """
    def __init__(
        self,
        data_path,
        node_col,
        flag="train",
        train_length=4296,   # 按“小时行”切分点
        val_ratio=0.2,
        seed=42,
        scale=True,
        scaler_x=None,
        scaler_y=None,
    ):
        super().__init__()
        self.data_path = data_path
        self.node_col = node_col
        self.flag = flag
        self.train_length = int(train_length)
        self.val_ratio = float(val_ratio)
        self.seed = int(seed)
        self.scale = bool(scale)

        self.common_features = ["month", "weekday", "hour", "TEMP", "MAX", "MIN"]

        self.scaler_x = scaler_x if scaler_x is not None else StandardScaler()
        self.scaler_y = scaler_y if scaler_y is not None else StandardScaler()

        # 一次性处理并缓存三份
        self.splits = self._build_splits()

        # 对外暴露当前 flag 对应的数据
        self.set_flag(flag)

    def set_flag(self, flag):
        if flag not in self.splits:
            raise ValueError("flag must be one of: 'train','val','test'")
        self.flag = flag
        self.X, self.y = self.splits[flag]  # torch tensors

    def _build_splits(self):
        df = pd.read_csv(self.data_path)
        df = df.interpolate(method="cubic", limit_direction="both")

        dt = pd.to_datetime(df["DATETIME"])
        df["month"] = dt.dt.month.astype(np.int64)
        df["weekday"] = dt.dt.weekday.astype(np.int64)
        df["hour"] = dt.dt.hour.astype(np.int64)

        # lag24
        df["lag24"] = df[self.node_col].shift(24)

        cols = self.common_features + ["lag24", self.node_col]
        data = df[cols].dropna().copy()

        train_pool = data.iloc[: self.train_length].copy()
        test_data  = data.iloc[self.train_length :].copy()

        X_tr = train_pool[self.common_features + ["lag24"]].values.astype(np.float32)  # [rows,F]
        y_tr = train_pool[[self.node_col]].values.astype(np.float32)                   # [rows,1]
        X_te = test_data[self.common_features + ["lag24"]].values.astype(np.float32)
        y_te = test_data[[self.node_col]].values.astype(np.float32)

        if self.scale:
            # 关键：只在 train_pool fit 一次
            self.scaler_x.fit(X_tr)
            self.scaler_y.fit(y_tr)

            X_tr = self.scaler_x.transform(X_tr)
            X_te = self.scaler_x.transform(X_te)
            y_tr = self.scaler_y.transform(y_tr)
            y_te = self.scaler_y.transform(y_te)

        # reshape 到整天
        def to_days(X, y):
            n = (len(X) // 24) * 24
            X = X[:n].reshape(-1, 24, X.shape[1])          # [D,24,F]
            y = y[:n].reshape(-1, 24)                      # [D,24]
            return X, y

        X_tr, y_tr = to_days(X_tr, y_tr)
        X_te, y_te = to_days(X_te, y_te)

        # train/val split (按天)
        D = X_tr.shape[0]
        rng = np.random.RandomState(self.seed)
        perm = rng.permutation(D)
        n_val = int(round(D * self.val_ratio))
        val_idx = perm[:n_val]
        tr_idx  = perm[n_val:]

        X_train, y_train = X_tr[tr_idx], y_tr[tr_idx]
        X_val,   y_val   = X_tr[val_idx], y_tr[val_idx]

        splits = {
            "train": (torch.tensor(X_train, dtype=torch.float32),
                      torch.tensor(y_train, dtype=torch.float32)),
            "val":   (torch.tensor(X_val, dtype=torch.float32),
                      torch.tensor(y_val, dtype=torch.float32)),
            "test":  (torch.tensor(X_te, dtype=torch.float32),
                      torch.tensor(y_te, dtype=torch.float32)),
        }
        return splits

    def inverse_transform(self, y_nor):
        # y_nor: torch/np, shape [...,24] or [...,1] etc.
        if torch.is_tensor(y_nor):
            y_np = y_nor.detach().cpu().numpy()
        else:
            y_np = np.asarray(y_nor)

        orig_shape = y_np.shape
        y2 = y_np.reshape(-1, 1)
        y_real = self.scaler_y.inverse_transform(y2).reshape(orig_shape)
        return y_real

    def __len__(self):
        return int(self.X.shape[0])

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]



class Dataset_load_multi_node_diff(Dataset):
    def __init__(
        self,
        data_path,
        node_cols,
        flag="train",
        train_length=4296,  # hour-rows split point
        val_ratio=0.2,
        scale=True,
        seed=42,
    ):
        super().__init__()
        self.data_path = data_path
        self.node_cols = list(node_cols)
        self.flag = flag
        self.train_length = int(train_length)
        self.val_ratio = float(val_ratio)
        self.scale = bool(scale)
        self.seed = int(seed)

        self.common_features = [
            "month", "weekday", "hour",
            "TEMP", "MAX", "MIN",
        ]
        self.scaler_common = StandardScaler()
        self.scaler_y = StandardScaler()

        self.cond, self.y = self._process()  # cond: [D,24,Fc+N], y: [D,24,N]

    def _process(self):
        df = pd.read_csv(self.data_path)
        df = df.interpolate(method="cubic", limit_direction="both")

        if "DATETIME" not in df.columns:
            raise ValueError("CSV must contain DATETIME column")

        for c in self.node_cols:
            if c not in df.columns:
                raise ValueError(f"node_col={c} not found in columns")

        dt = pd.to_datetime(df["DATETIME"])
        df["month"] = dt.dt.month.astype(np.int64)
        df["weekday"] = dt.dt.weekday.astype(np.int64)
        df["hour"] = dt.dt.hour.astype(np.int64)

        # lag-24 for all nodes
        lag_cols = [f"{c}__lag24" for c in self.node_cols]
        for c, lc in zip(self.node_cols, lag_cols):
            df[lc] = df[c].shift(24)

        # build data frame with needed columns
        common_cols = list(self.common_features)
        y_cols = list(self.node_cols)

        data = df[common_cols + lag_cols + y_cols].dropna().copy()

        train_pool = data.iloc[: self.train_length].copy()
        test_data = data.iloc[self.train_length:].copy()

        Xc_tr = train_pool[common_cols].values.astype(np.float32)  # [rows, Fc]
        y_tr = train_pool[y_cols].values.astype(np.float32)        # [rows, N]
        lag_tr = train_pool[lag_cols].values.astype(np.float32)    # [rows, N]

        Xc_te = test_data[common_cols].values.astype(np.float32)
        y_te = test_data[y_cols].values.astype(np.float32)
        lag_te = test_data[lag_cols].values.astype(np.float32)

        if self.scale:
            self.scaler_common.fit(Xc_tr)
            self.scaler_y.fit(y_tr)

            Xc_tr = self.scaler_common.transform(Xc_tr)
            Xc_te = self.scaler_common.transform(Xc_te)

            y_tr = self.scaler_y.transform(y_tr)
            y_te = self.scaler_y.transform(y_te)

            # lag uses same scaler as y
            lag_tr = self.scaler_y.transform(lag_tr)
            lag_te = self.scaler_y.transform(lag_te)

        Fc = Xc_tr.shape[1]
        N = y_tr.shape[1]

        # make full days
        n_tr_rows = (Xc_tr.shape[0] // 24) * 24
        n_te_rows = (Xc_te.shape[0] // 24) * 24

        Xc_tr = Xc_tr[:n_tr_rows].reshape(-1, 24, Fc)
        y_tr = y_tr[:n_tr_rows].reshape(-1, 24, N)
        lag_tr = lag_tr[:n_tr_rows].reshape(-1, 24, N)

        Xc_te = Xc_te[:n_te_rows].reshape(-1, 24, Fc)
        y_te = y_te[:n_te_rows].reshape(-1, 24, N)
        lag_te = lag_te[:n_te_rows].reshape(-1, 24, N)

        # cond = concat(common, lag_all) => [D,24,Fc+N]
        cond_tr = np.concatenate([Xc_tr, lag_tr], axis=-1)
        cond_te = np.concatenate([Xc_te, lag_te], axis=-1)

        # split train/val by day index
        n_days = cond_tr.shape[0]
        rng = np.random.RandomState(self.seed)
        perm = rng.permutation(n_days)
        n_val = int(round(n_days * self.val_ratio))
        val_idx = perm[:n_val]
        tr_idx = perm[n_val:]

        cond_train, y_train = cond_tr[tr_idx], y_tr[tr_idx]
        cond_val, y_val = cond_tr[val_idx], y_tr[val_idx]

        if self.flag == "train":
            cond, y = cond_train, y_train
        elif self.flag == "val":
            cond, y = cond_val, y_val
        elif self.flag == "test":
            cond, y = cond_te, y_te
        else:
            raise ValueError("flag must be one of: 'train','val','test'")

        return torch.tensor(cond, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def __getitem__(self, idx):
        return self.cond[idx], self.y[idx]

    def __len__(self):
        return int(self.cond.shape[0])

    def inverse_transform_y(self, y_nor):
        """
        y_nor: [.., N] or [..,24,N] etc.
        Returns same shape in real scale.
        """
        if torch.is_tensor(y_nor):
            y_np = y_nor.detach().cpu().numpy()
        else:
            y_np = np.asarray(y_nor)

        orig_shape = y_np.shape
        y2 = y_np.reshape(-1, orig_shape[-1])  # [-1, N]
        yr = self.scaler_y.inverse_transform(y2).reshape(orig_shape)
        return yr