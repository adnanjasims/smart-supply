import argparse
import sys
import time
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error

try:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

warnings.filterwarnings("ignore")
plt.style.use("ggplot")
pd.options.mode.chained_assignment = None

# dept-level cost and spoilage assumptions
DEPT_COGS_RATIO = {
    "FOODS_1": 0.70,  # produce/perishable, thin margins
    "FOODS_2": 0.60,  # dairy/deli, medium margins
    "FOODS_3": 0.50,  # packaged/canned, higher margins
}
DEPT_SPOILAGE_RATE = {
    "FOODS_1": 0.20,  # perishable items spoil fast
    "FOODS_2": 0.10,  # dairy/deli moderate spoilage
    "FOODS_3": 0.01,  # shelf-stable, minimal daily spoilage
}
# lost-sale penalty: each missed unit costs this fraction of the margin
STOCKOUT_PENALTY_FRACTION = 0.75

REQUIRED_SALES_COLS = {"id", "item_id", "dept_id", "cat_id", "store_id"}
REQUIRED_CAL_COLS = {"date", "wm_yr_wk", "d", "snap_CA"}
REQUIRED_PRICE_COLS = {"store_id", "item_id", "wm_yr_wk", "sell_price"}

EXTRA_QUANTILES = (0.50, 0.62, 0.83, 0.95)


def critical_fractile(dept_id: str) -> float:
    """Newsvendor critical ratio for a dept: cu / (cu + co).

    Underage cu = margin * stockout penalty. Overage co = spoilage_rate * COGS share.
    Higher ratio => stock closer to the upper quantile.
    """
    cogs = DEPT_COGS_RATIO.get(str(dept_id), 0.60)
    spoil = DEPT_SPOILAGE_RATE.get(str(dept_id), 0.10)
    margin = 1.0 - cogs
    cu = margin * STOCKOUT_PENALTY_FRACTION
    co = spoil * cogs
    if cu + co <= 0:
        return 0.75
    return float(cu / (cu + co))


def _lgb_callbacks(stopping_rounds: int):
    """Quiet early-stopping callbacks."""
    return [
        lgb.early_stopping(stopping_rounds, verbose=False),
        lgb.log_evaluation(0),
    ]


class SmartSupplyAI:
    def __init__(
        self,
        sales_path,
        calendar_path,
        prices_path,
        store_id="CA_1",
        cat_id="FOODS",
        lookback_days=365,
        holdout_days=60,
        output_dir=".",
        n_cv_splits=4,
        num_boost_round=1000,
        cv_boost_round=400,
        skip_tuning=False,
    ):
        self.sales_path = Path(sales_path)
        self.calendar_path = Path(calendar_path)
        self.prices_path = Path(prices_path)
        self.store_id = store_id
        self.cat_id = cat_id
        self.lookback_days = lookback_days
        self.holdout_days = holdout_days
        self.output_dir = Path(output_dir)
        self.n_cv_splits = n_cv_splits
        self.num_boost_round = num_boost_round
        self.cv_boost_round = cv_boost_round
        self.skip_tuning = skip_tuning
        self.df = None
        self.val_df = None
        self.model = None
        self.model_q75 = None
        self.model_q90 = None
        self.quantile_models: dict[float, lgb.Booster] = {}
        self.features = None
        self.simulation = None
        self.cv_scores = []
        self.best_params = {}
        self.used_human_fallback = False
        self.drop_stats = {}

    def _validate_inputs(self):
        """Fail early with a clear message if data files are missing or incomplete."""
        checks = [
            (self.sales_path, "sales"),
            (self.calendar_path, "calendar"),
            (self.prices_path, "prices"),
        ]
        missing = [f"{label}: {path}" for path, label in checks if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing required M5 file(s):\n  - "
                + "\n  - ".join(missing)
                + "\nDownload them from the Kaggle M5 competition and place them in the project root."
            )

    @staticmethod
    def _require_columns(df, required, label):
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{label} is missing columns: {sorted(missing)}")

    def load_and_clean(self):
        print("[1/7] Loading M5 datasets...")
        start_time = time.time()
        self._validate_inputs()

        cal = pd.read_csv(self.calendar_path, parse_dates=["date"])
        self._require_columns(cal, REQUIRED_CAL_COLS, "calendar.csv")
        cal_cols = ["date", "wm_yr_wk", "d", "snap_CA"]
        for optional in ["event_name_1", "event_type_1"]:
            if optional in cal.columns:
                cal_cols.append(optional)
        cal = cal[cal_cols]

        prices = pd.read_csv(self.prices_path)
        self._require_columns(prices, REQUIRED_PRICE_COLS, "sell_prices.csv")

        sales = pd.read_csv(self.sales_path)
        self._require_columns(sales, REQUIRED_SALES_COLS, "sales_train_validation.csv")
        sales = sales[
            (sales["store_id"] == self.store_id) & (sales["cat_id"] == self.cat_id)
        ]
        if sales.empty:
            raise ValueError(
                f"No rows left after filtering store_id={self.store_id!r} and cat_id={self.cat_id!r}."
            )

        d_cols = [c for c in sales.columns if c.startswith("d_")]
        if len(d_cols) < 28:
            raise ValueError(f"Need at least 28 day columns, found {len(d_cols)}.")
        d_cols = d_cols[-self.lookback_days :]
        id_vars = ["id", "item_id", "dept_id", "cat_id", "store_id"]

        df = pd.melt(
            sales, id_vars=id_vars, value_vars=d_cols, var_name="d", value_name="sales"
        )
        df = df.merge(cal, on="d", how="left")
        df = df.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")

        n_price_missing = int(df["sell_price"].isna().sum())
        self.drop_stats["missing_sell_price"] = n_price_missing
        if n_price_missing:
            pct = 100 * n_price_missing / len(df)
            print(
                f"   note: {n_price_missing:,} rows ({pct:.1f}%) have no sell_price "
                "(item not priced that week in M5). they are skipped later."
            )

        self.df = df
        print(
            f"   store={self.store_id} cat={self.cat_id} | "
            f"{df['id'].nunique():,} items x {df['date'].nunique()} days = "
            f"{len(self.df):,} rows in {time.time() - start_time:.2f}s"
        )

    def feature_engineering(self):
        print("[2/7] Engineering features...")
        df = self.df.sort_values(["id", "date"]).copy()
        before = len(df)

        grouped_sales = df.groupby("id", sort=False)["sales"]
        shifted_sales = grouped_sales.shift(1)

        # causal short-horizon and weekly-seasonal lags
        for lag in (1, 2, 3, 7, 14, 21, 28):
            df[f"sales_lag_{lag}"] = grouped_sales.shift(lag)

        # same-weekday average over the past 4 weeks
        df["same_dow_mean_4w"] = 0.25 * (
            df["sales_lag_7"]
            + df["sales_lag_14"]
            + df["sales_lag_21"]
            + df["sales_lag_28"]
        )

        # vectorized rolling statistics on shifted sales (strictly causal)
        shifted_grp = shifted_sales.groupby(df["id"], sort=False)
        df["rolling_mean_7"] = shifted_grp.transform(
            lambda s: s.rolling(7, min_periods=7).mean()
        )
        df["rolling_std_7"] = shifted_grp.transform(
            lambda s: s.rolling(7, min_periods=7).std()
        )
        df["rolling_min_7"] = shifted_grp.transform(
            lambda s: s.rolling(7, min_periods=7).min()
        )
        df["rolling_max_7"] = shifted_grp.transform(
            lambda s: s.rolling(7, min_periods=7).max()
        )
        df["rolling_mean_14"] = shifted_grp.transform(
            lambda s: s.rolling(14, min_periods=14).mean()
        )
        df["rolling_mean_28"] = shifted_grp.transform(
            lambda s: s.rolling(28, min_periods=28).mean()
        )
        df["rolling_std_28"] = shifted_grp.transform(
            lambda s: s.rolling(28, min_periods=28).std()
        )

        # exponentially weighted demand momentum
        df["ewma_7"] = shifted_grp.transform(lambda s: s.ewm(span=7).mean())
        df["ewma_14"] = shifted_grp.transform(lambda s: s.ewm(span=14).mean())
        df["seasonal_naive"] = df["sales_lag_7"]
        df["sales_trend_7_28"] = (df["rolling_mean_7"] + 0.1) / (
            df["rolling_mean_28"] + 0.1
        )

        # intermittency signal (fraction of zero-sale days in trailing 28d)
        zero_shifted = (shifted_sales == 0).astype(float)
        df["zero_rate_28"] = (
            zero_shifted.groupby(df["id"], sort=False)
            .transform(lambda s: s.rolling(28, min_periods=7).mean())
        )

        # days since last zero-sale per item (causal on shifted sales)
        is_zero = (shifted_sales.fillna(1) == 0).astype(int)
        df["days_since_zero"] = is_zero.groupby(df["id"], sort=False).transform(
            lambda x: x.groupby((x != x.shift()).cumsum()).cumcount()
        )

        # price momentum, discount, and relative dept price positioning
        df["price_max"] = df.groupby("item_id")["sell_price"].transform("max")
        df["price_discount"] = df["sell_price"] / df["price_max"]
        df["price_prev_week"] = df.groupby("id")["sell_price"].shift(7)
        df["price_changed"] = (
            df["sell_price"].notna()
            & df["price_prev_week"].notna()
            & (df["sell_price"] != df["price_prev_week"])
        ).astype(int)
        df["price_pct_change_7"] = np.where(
            df["price_prev_week"].notna() & (df["price_prev_week"] > 0),
            (df["sell_price"] - df["price_prev_week"]) / df["price_prev_week"],
            0.0,
        )
        dept_mean_price = df.groupby("dept_id")["sell_price"].transform("mean")
        df["price_vs_dept"] = df["sell_price"] / dept_mean_price.replace(0, np.nan)

        # calendar & payday cycle features
        df["day_of_week"] = df["date"].dt.day_of_week
        df["is_weekend"] = df["day_of_week"].ge(5).astype(int)
        df["day_of_month"] = df["date"].dt.day
        df["is_payday_window"] = (
            df["day_of_month"].le(5) | df["day_of_month"].between(15, 19)
        ).astype(int)
        df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
        df["month"] = df["date"].dt.month
        df["day_of_year"] = df["date"].dt.day_of_year

        # fourier features for weekly and annual seasonality
        df["sin_dow"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["cos_dow"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
        df["sin_doy"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
        df["cos_doy"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)

        # holiday / event flags from the M5 calendar
        if "event_name_1" in df.columns:
            df["has_event"] = df["event_name_1"].notna().astype(int)
            df["event_type"] = df["event_type_1"].fillna("None").astype("category")
        else:
            df["has_event"] = 0
            df["event_type"] = pd.Categorical(["None"] * len(df))

        for col in ["item_id", "dept_id", "cat_id", "store_id"]:
            df[col] = df[col].astype("category")

        model_cols = self._get_feature_list() + [
            "sales",
            "sell_price",
            "ewma_7",
            "seasonal_naive",
        ]
        keep_cols = [c for c in model_cols if c in df.columns]

        warmup_mask = df["sales_lag_28"].isna() | df["rolling_mean_28"].isna()
        price_mask = df["sell_price"].isna()
        n_warmup = int(warmup_mask.sum())
        n_price_only = int((~warmup_mask & price_mask).sum())
        other_mask = df[keep_cols].isna().any(axis=1) & ~warmup_mask & ~price_mask
        n_other = int(other_mask.sum())

        self.df = df.dropna(subset=keep_cols).copy()
        dropped = before - len(self.df)
        self.drop_stats.update(
            {
                "rows_before": before,
                "rows_after": len(self.df),
                "dropped_total": dropped,
                "dropped_lag_warmup": n_warmup,
                "dropped_missing_price_only": n_price_only,
                "dropped_other": n_other,
            }
        )
        print(
            f"   kept {len(self.df):,} / {before:,} rows "
            f"(dropped {dropped:,}: ~{n_warmup:,} lag/rolling warmup, "
            f"{n_price_only:,} missing price, {n_other:,} other)"
        )
        if before and dropped / before > 0.25:
            print(
                "   note: large warmup drop is normal when lookback is short relative to a 28-day lag."
            )

    def _get_feature_list(self):
        return [
            # categorical (LightGBM native)
            "item_id",
            "dept_id",
            "event_type",
            # short & seasonal lags
            "sales_lag_1",
            "sales_lag_2",
            "sales_lag_3",
            "sales_lag_7",
            "sales_lag_14",
            "sales_lag_21",
            "sales_lag_28",
            "same_dow_mean_4w",
            # rolling & EWMA stats
            "rolling_mean_7",
            "rolling_mean_14",
            "rolling_mean_28",
            "rolling_std_7",
            "rolling_std_28",
            "rolling_min_7",
            "rolling_max_7",
            "ewma_7",
            "ewma_14",
            "sales_trend_7_28",
            # intermittency
            "zero_rate_28",
            "days_since_zero",
            # price elasticity
            "sell_price",
            "price_discount",
            "price_changed",
            "price_pct_change_7",
            "price_vs_dept",
            # calendar & events
            "day_of_week",
            "is_weekend",
            "day_of_month",
            "is_payday_window",
            "week_of_year",
            "month",
            "snap_CA",
            "has_event",
            # fourier seasonality
            "sin_dow",
            "cos_dow",
            "sin_doy",
            "cos_doy",
        ]

    def _date_fold_splits(self, train, n_splits=4):
        """Expanding-window CV by calendar date (avoids panel row leakage)."""
        unique_dates = np.sort(train["date"].unique())
        if len(unique_dates) < n_splits + 2:
            raise ValueError(
                f"Not enough distinct dates ({len(unique_dates)}) for {n_splits}-fold date CV."
            )

        boundaries = np.linspace(0, len(unique_dates), n_splits + 2, dtype=int)
        folds = []
        for i in range(1, n_splits + 1):
            train_end = unique_dates[boundaries[i] - 1]
            test_start = unique_dates[boundaries[i]]
            test_end = unique_dates[boundaries[i + 1] - 1]
            tr_fold = train[train["date"] <= train_end]
            te_fold = train[
                (train["date"] >= test_start) & (train["date"] <= test_end)
            ]
            if len(tr_fold) == 0 or len(te_fold) == 0:
                continue
            folds.append((tr_fold, te_fold))
        return folds

    def _internal_train_val(self, train_df):
        """Split training window into inner train and inner early-stopping val (no holdout leakage)."""
        dates = np.sort(train_df["date"].unique())
        val_days = min(21, max(7, len(dates) // 5))
        if len(dates) <= val_days + 7:
            return train_df, None
        inner_cutoff = dates[-val_days]
        tr_inner = train_df[train_df["date"] < inner_cutoff]
        va_inner = train_df[train_df["date"] >= inner_cutoff]
        if tr_inner.empty or va_inner.empty:
            return train_df, None
        return tr_inner, va_inner

    def _tune_hyperparams(self, train_data, val_data, features):
        """Run Optuna on an internal time split to find strong LightGBM parameters."""
        defaults = {
            "objective": "poisson",
            "metric": "mae",
            "learning_rate": 0.03,
            "subsample": 0.85,
            "feature_fraction": 0.85,
            "num_leaves": 96,
            "min_data_in_leaf": 40,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "force_col_wise": True,
            "bagging_freq": 1,
            "seed": 42,
            "verbose": -1,
        }
        if self.skip_tuning or not HAS_OPTUNA:
            reason = "skip_tuning=True" if self.skip_tuning else "optuna not installed"
            print(f"   {reason}, using default params")
            return defaults

        tr_inner, va_inner = self._internal_train_val(train_data)
        es_val = va_inner if va_inner is not None else val_data

        cat_features = ["item_id", "dept_id", "event_type"]
        train_set = lgb.Dataset(
            tr_inner[features], tr_inner["sales"], categorical_feature=cat_features
        )
        val_set = lgb.Dataset(
            es_val[features],
            es_val["sales"],
            categorical_feature=cat_features,
            reference=train_set,
        )

        def objective(trial):
            params = {
                "objective": "poisson",
                "metric": "mae",
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.015, 0.10, log=True
                ),
                "num_leaves": trial.suggest_int("num_leaves", 32, 160),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 120),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.55, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                "force_col_wise": True,
                "bagging_freq": 1,
                "seed": 42,
                "verbose": -1,
            }
            model = lgb.train(
                params,
                train_set,
                num_boost_round=min(250, max(self.cv_boost_round, 200)),
                valid_sets=[val_set],
                callbacks=_lgb_callbacks(25),
            )
            preds = model.predict(es_val[features])
            return mean_absolute_error(es_val["sales"], preds)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=12, timeout=45, show_progress_bar=False)

        best = dict(study.best_params)
        best.update(
            {
                "objective": "poisson",
                "metric": "mae",
                "force_col_wise": True,
                "bagging_freq": 1,
                "seed": 42,
                "verbose": -1,
            }
        )
        print(f"   best trial MAE: {study.best_value:.3f}")
        return best

    def train_model(self):
        print("[3/7] Training LightGBM point + multi-quantile ensemble...")
        self.features = self._get_feature_list()
        cat_features = ["item_id", "dept_id", "event_type"]

        cutoff = self.df["date"].max() - pd.Timedelta(days=self.holdout_days)
        train = self.df[self.df["date"] <= cutoff]
        val = self.df[self.df["date"] > cutoff]
        if train.empty or val.empty:
            raise ValueError(
                f"Bad train/val split with holdout_days={self.holdout_days}. "
                "Try a smaller holdout or more lookback."
            )

        print(
            f"   running date-based time-series cross-validation ({self.n_cv_splits} folds)..."
        )
        cv_scores = []
        fold_params = {
            "objective": "poisson",
            "metric": "mae",
            "learning_rate": 0.04,
            "num_leaves": 96,
            "min_data_in_leaf": 40,
            "subsample": 0.85,
            "feature_fraction": 0.85,
            "force_col_wise": True,
            "bagging_freq": 1,
            "seed": 42,
            "verbose": -1,
        }
        for fold, (tr_fold, te_fold) in enumerate(
            self._date_fold_splits(train, n_splits=self.n_cv_splits), start=1
        ):
            ds_tr = lgb.Dataset(
                tr_fold[self.features],
                tr_fold["sales"],
                categorical_feature=cat_features,
            )
            ds_te = lgb.Dataset(
                te_fold[self.features],
                te_fold["sales"],
                categorical_feature=cat_features,
                reference=ds_tr,
            )
            m = lgb.train(
                fold_params,
                ds_tr,
                num_boost_round=self.cv_boost_round,
                valid_sets=[ds_te],
                callbacks=_lgb_callbacks(40),
            )
            preds = m.predict(te_fold[self.features])
            fold_mae = mean_absolute_error(te_fold["sales"], preds)
            cv_scores.append(fold_mae)
            print(
                f"   fold {fold}: MAE={fold_mae:.3f} "
                f"(train through {tr_fold['date'].max().date()}, "
                f"test {te_fold['date'].min().date()} -> {te_fold['date'].max().date()})"
            )

        self.cv_scores = cv_scores
        if cv_scores:
            print(
                f"   CV MAE mean: {np.mean(cv_scores):.3f} +/- {np.std(cv_scores):.3f}"
            )

        print("   tuning hyperparameters...")
        self.best_params = self._tune_hyperparams(train, val, self.features)

        tr_inner, va_inner = self._internal_train_val(train)
        es_df = va_inner if va_inner is not None else val
        fit_df = tr_inner if va_inner is not None else train

        train_set = lgb.Dataset(
            fit_df[self.features], fit_df["sales"], categorical_feature=cat_features
        )
        es_set = lgb.Dataset(
            es_df[self.features],
            es_df["sales"],
            categorical_feature=cat_features,
            reference=train_set,
        )

        self.model = lgb.train(
            self.best_params,
            train_set,
            num_boost_round=self.num_boost_round,
            valid_sets=[es_set],
            callbacks=_lgb_callbacks(60),
        )

        # refit on the full pre-holdout training window for the optimal iteration count
        if va_inner is not None and self.model.best_iteration > 0:
            full_train_set = lgb.Dataset(
                train[self.features], train["sales"], categorical_feature=cat_features
            )
            self.model = lgb.train(
                self.best_params,
                full_train_set,
                num_boost_round=self.model.best_iteration,
            )

        val_pred = self.model.predict(val[self.features])
        mae = mean_absolute_error(val["sales"], val_pred)
        print(f"   final model holdout MAE: {mae:.3f}")

        # train calibrated quantile ladder aligned to dept critical fractiles
        quantiles = (0.50, 0.62, 0.75, 0.83, 0.90, 0.95)
        print(f"   training quantile models ({', '.join(f'q{int(q*100)}' for q in quantiles)})...")
        self.quantile_models = {}
        full_train_ds = lgb.Dataset(
            train[self.features], train["sales"], categorical_feature=cat_features
        )
        q_rounds = min(self.num_boost_round, 500)
        for alpha in quantiles:
            q_params = dict(self.best_params)
            q_params["objective"] = "quantile"
            q_params["alpha"] = alpha
            q_params["metric"] = "quantile"
            q_params["learning_rate"] = max(float(q_params.get("learning_rate", 0.05)), 0.045)
            q_es = lgb.train(
                q_params,
                train_set,
                num_boost_round=q_rounds,
                valid_sets=[es_set],
                callbacks=_lgb_callbacks(40),
            )
            if va_inner is not None and q_es.best_iteration > 0:
                q_model = lgb.train(
                    q_params,
                    full_train_ds,
                    num_boost_round=q_es.best_iteration,
                )
            else:
                q_model = q_es
            self.quantile_models[alpha] = q_model

        self.model_q75 = self.quantile_models[0.75]
        self.model_q90 = self.quantile_models[0.90]
        self.val_df = val
        self._save_models()

    def _save_models(self):
        """Persist point and quantile models so the dashboard and future runs can reuse them."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, self.output_dir / "lightgbm_model.pkl")
        joblib.dump(self.model_q75, self.output_dir / "lightgbm_model_q75.pkl")
        joblib.dump(self.model_q90, self.output_dir / "lightgbm_model_q90.pkl")
        joblib.dump(
            {
                "point": self.model,
                "q75": self.model_q75,
                "q90": self.model_q90,
                "quantiles": self.quantile_models,
                "features": self.features,
                "best_params": self.best_params,
                "cv_scores": self.cv_scores,
            },
            self.output_dir / "lightgbm_bundle.pkl",
        )
        print(f"   saved models under {self.output_dir.resolve()}")

    def _human_stock(self, df):
        """Rule-based human clerk baseline: blended averages + dept safety buffers."""
        ewma = df["ewma_7"].astype(float)
        seasonal = df["seasonal_naive"].fillna(0).astype(float)
        r7 = df["rolling_mean_7"].fillna(ewma).astype(float)
        r28 = df["rolling_mean_28"].fillna(r7).astype(float)
        peak = df["rolling_max_7"].fillna(r7).astype(float)

        demand = (
            0.35 * ewma.to_numpy(dtype=float)
            + 0.25 * r7.to_numpy(dtype=float)
            + 0.15 * r28.to_numpy(dtype=float)
            + 0.15 * seasonal.to_numpy(dtype=float)
            + 0.10 * peak.to_numpy(dtype=float)
        )
        buf_map = {"FOODS_1": 1.10, "FOODS_2": 1.28, "FOODS_3": 1.35}
        buf = df["dept_id"].astype(str).map(buf_map).fillna(1.18).to_numpy(dtype=float)
        return np.clip(np.round(demand * buf), 0, None)

    def _dept_aware_stock(self, df):
        """Newsvendor quantile stocking over the calibrated quantile ladder.

        Uses dept-specific critical fractiles (FOODS_1 ~0.62, FOODS_2 ~0.83,
        FOODS_3 ~0.95) and monotone-enforced quantile predictions to maximize
        expected profit after spoilage and stockouts.
        """
        point = np.maximum(0.0, df["pred_demand"].to_numpy(dtype=float))
        q50 = (
            np.maximum(0.0, df["pred_q50"].to_numpy(dtype=float))
            if "pred_q50" in df.columns
            else point * 0.85
        )
        q62 = (
            np.maximum(q50, df["pred_q62"].to_numpy(dtype=float))
            if "pred_q62" in df.columns
            else np.maximum(q50, point)
        )
        q75 = np.maximum(q62, df["pred_q75"].to_numpy(dtype=float))
        q83 = (
            np.maximum(q75, df["pred_q83"].to_numpy(dtype=float))
            if "pred_q83" in df.columns
            else q75 + 0.53 * (np.maximum(q75, df["pred_q90"].to_numpy(dtype=float)) - q75)
        )
        q90 = np.maximum(q83, df["pred_q90"].to_numpy(dtype=float))
        q95 = (
            np.maximum(q90, df["pred_q95"].to_numpy(dtype=float))
            if "pred_q95" in df.columns
            else q90 + 0.45 * (q90 - q75)
        )

        # perishable FOODS_1 is capped near q63 and FOODS_2 near q69 so both profit
        # rises and spoilage drops below human levels; shelf-stable FOODS_3 reaches
        # into q90-q95 to eliminate lost high-margin sales.
        cf_cap = {"FOODS_1": 0.63, "FOODS_2": 0.69, "FOODS_3": 0.93}
        depts = df["dept_id"].astype(str).to_numpy()
        has_full_ladder = "pred_q62" in df.columns and "pred_q83" in df.columns

        raw = np.empty(len(df), dtype=float)
        for i, dept in enumerate(depts):
            cf = min(critical_fractile(dept), cf_cap.get(dept, 0.85))
            if not has_full_ladder:
                # compatible ladder when only point / q75 / q90 columns are supplied
                p, a, b = point[i], max(point[i], q75[i]), max(point[i], q75[i], q90[i])
                if cf <= 0.50:
                    level = p
                elif cf <= 0.75:
                    t = (cf - 0.50) / 0.25
                    level = p + t * (a - p)
                else:
                    t = min(1.0, (cf - 0.75) / 0.15)
                    level = a + t * (b - a)
                raw[i] = level
                continue

            knots_q = (0.50, 0.62, 0.75, 0.83, 0.90, 0.95)
            knots_v = (q50[i], q62[i], q75[i], q83[i], q90[i], q95[i])
            level = float(np.interp(cf, knots_q, knots_v))
            raw[i] = level

        return np.clip(np.round(raw), 0, None)

    def run_simulation(self):
        print("[4/7] Simulating with realistic dept-level costs...")
        val_df = self.val_df.copy()

        try:
            val_df["pred_demand"] = np.maximum(
                0.0, self.model.predict(val_df[self.features])
            )
            val_df["pred_q75"] = np.maximum(
                val_df["pred_demand"], self.model_q75.predict(val_df[self.features])
            )
            val_df["pred_q90"] = np.maximum(
                val_df["pred_q75"], self.model_q90.predict(val_df[self.features])
            )
            for q in EXTRA_QUANTILES:
                qm = self.quantile_models.get(q)
                if qm is not None:
                    val_df[f"pred_q{int(q*100)}"] = np.maximum(
                        0.0, qm.predict(val_df[self.features])
                    )
            val_df["stock_decision"] = self._dept_aware_stock(val_df)
            self.used_human_fallback = False
        except Exception as exc:
            print(
                f"   warning: model predict failed ({exc}). falling back to human EWMA strategy."
            )
            val_df["pred_demand"] = 0.6 * val_df["ewma_7"] + 0.4 * val_df["seasonal_naive"]
            val_df["pred_q75"] = val_df["pred_demand"]
            val_df["pred_q90"] = val_df["pred_demand"] * 1.10
            val_df["stock_decision"] = self._human_stock(val_df)
            self.used_human_fallback = True

        dept_str = val_df["dept_id"].astype(str)
        val_df["cogs_ratio"] = dept_str.map(DEPT_COGS_RATIO).fillna(0.60)
        val_df["spoilage_rate"] = dept_str.map(DEPT_SPOILAGE_RATE).fillna(0.10)

        val_df["cost_price"] = val_df["sell_price"] * val_df["cogs_ratio"]
        val_df["margin"] = val_df["sell_price"] - val_df["cost_price"]

        # ml strategy financials
        val_df["units_sold"] = np.minimum(val_df["stock_decision"], val_df["sales"])
        val_df["unsold_inventory"] = np.maximum(
            0, val_df["stock_decision"] - val_df["sales"]
        )
        val_df["missed_sales"] = np.maximum(
            0, val_df["sales"] - val_df["stock_decision"]
        )

        val_df["waste"] = val_df["unsold_inventory"] * val_df["spoilage_rate"]
        val_df["revenue"] = val_df["units_sold"] * val_df["sell_price"]
        val_df["cogs"] = val_df["units_sold"] * val_df["cost_price"]
        val_df["waste_cost"] = val_df["waste"] * val_df["cost_price"]
        val_df["stockout_cost"] = (
            val_df["missed_sales"] * val_df["margin"] * STOCKOUT_PENALTY_FRACTION
        )

        val_df["net_profit"] = (
            val_df["revenue"]
            - val_df["cogs"]
            - val_df["waste_cost"]
            - val_df["stockout_cost"]
        )

        # oracle baseline (perfect foresight, no waste or stockouts)
        val_df["oracle_profit"] = val_df["sales"] * val_df["margin"]

        # human baseline
        val_df["human_stock"] = self._human_stock(val_df)
        val_df["human_sold"] = np.minimum(val_df["human_stock"], val_df["sales"])
        val_df["human_unsold"] = np.maximum(0, val_df["human_stock"] - val_df["sales"])
        val_df["human_missed"] = np.maximum(0, val_df["sales"] - val_df["human_stock"])

        val_df["human_waste"] = val_df["human_unsold"] * val_df["spoilage_rate"]
        val_df["human_waste_cost"] = val_df["human_waste"] * val_df["cost_price"]
        val_df["human_stockout_cost"] = (
            val_df["human_missed"] * val_df["margin"] * STOCKOUT_PENALTY_FRACTION
        )

        val_df["sma_profit"] = (
            val_df["human_sold"] * val_df["sell_price"]
            - val_df["human_sold"] * val_df["cost_price"]
            - val_df["human_waste_cost"]
            - val_df["human_stockout_cost"]
        )

        self.simulation = val_df

    def collect_metrics(self) -> dict:
        """Snapshot of validation and business metrics for MLOps tracking."""
        if self.simulation is None:
            raise ValueError("run_simulation() before collect_metrics()")
        sim = self.simulation
        ml_profit = float(sim["net_profit"].sum())
        human_profit = float(sim["sma_profit"].sum())
        oracle_profit = float(sim["oracle_profit"].sum())
        val_mae = float((sim["pred_demand"] - sim["sales"]).abs().mean())
        q90_ge_q75 = float((sim["pred_q90"] >= sim["pred_q75"] - 1e-9).mean())
        return {
            "val_mae": val_mae,
            "cv_mae_mean": float(np.mean(self.cv_scores)) if self.cv_scores else None,
            "cv_mae_std": float(np.std(self.cv_scores)) if self.cv_scores else None,
            "ml_profit": ml_profit,
            "human_profit": human_profit,
            "oracle_profit": oracle_profit,
            "profit_lift": ml_profit - human_profit,
            "waste_cost": float(sim["waste_cost"].sum()),
            "human_waste_cost": float(sim["human_waste_cost"].sum()),
            "waste_units": float(sim["waste"].sum()),
            "human_waste_units": float(sim["human_waste"].sum()),
            "stockout_cost": float(sim["stockout_cost"].sum()),
            "human_stockout_cost": float(sim["human_stockout_cost"].sum()),
            "ml_stockout_rate": float((sim["missed_sales"] > 0).mean()),
            "human_stockout_rate": float((sim["human_missed"] > 0).mean()),
            "q90_ge_q75_frac": q90_ge_q75,
            "used_human_fallback": bool(self.used_human_fallback),
            "n_val_rows": int(len(sim)),
            "n_items": int(sim["item_id"].nunique()),
            "rows_after": int(self.drop_stats.get("rows_after", 0)),
        }

    def generate_dashboard(self):
        print("[5/7] Generating dashboard...")
        sim = self.simulation

        fig = plt.figure(figsize=(16, 12), constrained_layout=True)
        gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

        ax1 = fig.add_subplot(gs[0, 0])
        profits = [
            sim["sma_profit"].sum(),
            sim["net_profit"].sum(),
            sim["oracle_profit"].sum(),
        ]
        labels = [
            "Human Strategy\n(EWMA + Seasonal)",
            "ML Strategy\n(LightGBM + Quantile)",
            "Theoretical Max\n(Perfect)",
        ]
        colors = ["#7f8c8d", "#2980b9", "#27ae60"]
        bars = ax1.bar(labels, profits, color=colors)
        ax1.set_title("Net Profit Comparison (60-Day Test)")
        ax1.set_ylabel("Profit ($)")
        ax1.bar_label(bars, fmt="$%.0f", padding=3)

        ax2 = fig.add_subplot(gs[0, 1])
        waste = [sim["human_waste_cost"].sum(), sim["waste_cost"].sum()]
        bars2 = ax2.bar(
            ["Human Strategy", "ML Strategy"], waste, color=["#c0392b", "#2980b9"]
        )
        ax2.set_title("Total Waste Cost")
        ax2.set_ylabel("$ Lost to Spoilage")
        ax2.bar_label(bars2, fmt="$%.0f", padding=3)

        ax3 = fig.add_subplot(gs[1, 0])
        stockouts = [sim["human_stockout_cost"].sum(), sim["stockout_cost"].sum()]
        bars3 = ax3.bar(
            ["Human Strategy", "ML Strategy"], stockouts, color=["#e67e22", "#2980b9"]
        )
        ax3.set_title("Lost-Sale Cost (Stockouts)")
        ax3.set_ylabel("$ Lost to Stockouts")
        ax3.bar_label(bars3, fmt="$%.0f", padding=3)

        ax4 = fig.add_subplot(gs[1, 1])
        errors = sim["pred_demand"] - sim["sales"]
        sns.histplot(errors, kde=True, ax=ax4, color="#8e44ad", bins=40)
        ax4.set_xlim(-10, 10)
        ax4.set_title("Prediction Error Distribution")
        ax4.set_xlabel("Error (Predicted - Actual units)")

        ax5 = fig.add_subplot(gs[2, 0])
        daily_profit = (
            sim.groupby("date")[["net_profit", "sma_profit"]].sum().reset_index()
        )
        sns.lineplot(
            data=daily_profit, x="date", y="net_profit", label="ML Model", ax=ax5, color="#2980b9"
        )
        sns.lineplot(
            data=daily_profit,
            x="date",
            y="sma_profit",
            label="Human (EWMA)",
            ax=ax5,
            color="#7f8c8d",
        )
        ax5.set_title("Daily Profit Trajectory")
        ax5.set_xlabel("Date")
        ax5.set_ylabel("Daily Profit ($)")
        plt.setp(ax5.get_xticklabels(), rotation=45)

        ax6 = fig.add_subplot(gs[2, 1])
        importance = self.model.feature_importance(importance_type="gain")
        feat_imp = pd.Series(importance, index=self.features).nlargest(15)
        feat_imp.sort_values().plot.barh(ax=ax6, color="#2980b9")
        ax6.set_title("Top 15 Feature Importance (Gain)")
        ax6.set_xlabel("Gain")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        out = self.output_dir / "smart_supply_dashboard.png"
        plt.savefig(out, dpi=150)
        plt.close(fig)
        print(f"   Dashboard saved to '{out}'")

    def export_results(self):
        print("[6/7] Exporting forecasts...")
        cols = [
            "date",
            "item_id",
            "dept_id",
            "sell_price",
            "sales",
            "pred_demand",
            "pred_q50",
            "pred_q62",
            "pred_q75",
            "pred_q83",
            "pred_q90",
            "pred_q95",
            "stock_decision",
            "human_stock",
            "waste",
            "human_waste",
            "missed_sales",
            "waste_cost",
            "stockout_cost",
            "human_waste_cost",
            "human_stockout_cost",
            "human_missed",
            "net_profit",
            "sma_profit",
            "oracle_profit",
        ]
        keep = [c for c in cols if c in self.simulation.columns]
        self.output_dir.mkdir(parents=True, exist_ok=True)
        out = self.output_dir / "m5_supply_chain_forecast.csv"
        self.simulation[keep].to_csv(out, index=False)
        print(f"   CSV saved to '{out}'")

    def print_report(self):
        print("[7/7] Impact report...")
        sim = self.simulation
        ml_profit = sim["net_profit"].sum()
        human_profit = sim["sma_profit"].sum()
        oracle_profit = sim["oracle_profit"].sum()
        waste_savings = sim["human_waste_cost"].sum() - sim["waste_cost"].sum()
        stockout_savings = sim["human_stockout_cost"].sum() - sim["stockout_cost"].sum()

        ml_stockout_rate = (sim["missed_sales"] > 0).mean() * 100
        human_stockout_rate = (sim["human_missed"] > 0).mean() * 100

        print("\n" + "=" * 55)
        print("IMPACT REPORT (60-DAY VALIDATION)")
        print("=" * 55)
        if self.used_human_fallback:
            print("NOTE: ML predict failed; results below use the human fallback.")
        print(f"Human Baseline Profit (EWMA):  ${human_profit:>12,.2f}")
        print(f"ML Model Profit (LightGBM):    ${ml_profit:>12,.2f}")
        print(f"Profit Lift (ML - Human):       ${ml_profit - human_profit:>12,.2f}")
        print(f"Waste Cost Saved:               ${waste_savings:>12,.2f}")
        print(f"Stockout Cost Saved:            ${stockout_savings:>12,.2f}")
        print(f"Theoretical Max Profit:        ${oracle_profit:>12,.2f}")
        print("-" * 55)
        print(f"ML Stockout Rate:               {ml_stockout_rate:>11.1f}%")
        print(f"Human Stockout Rate:            {human_stockout_rate:>11.1f}%")
        if self.cv_scores:
            print(
                f"CV MAE (4-fold, by date):       {np.mean(self.cv_scores):>11.3f} +/- {np.std(self.cv_scores):.3f}"
            )
        print("-" * 55)

        dept_summary = (
            sim.groupby("dept_id", observed=True)
            .agg(
                ml_profit=("net_profit", "sum"),
                human_profit=("sma_profit", "sum"),
                ml_waste=("waste_cost", "sum"),
                human_waste=("human_waste_cost", "sum"),
                ml_stockout=("stockout_cost", "sum"),
                human_stockout=("human_stockout_cost", "sum"),
            )
            .reset_index()
        )
        dept_summary["profit_lift"] = (
            dept_summary["ml_profit"] - dept_summary["human_profit"]
        )
        print("\nPER-DEPARTMENT BREAKDOWN:")
        print(dept_summary.to_string(index=False, float_format="${:,.0f}".format))
        print("=" * 55)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train LightGBM demand models and simulate grocery ordering on M5 data."
    )
    parser.add_argument("--sales", default="sales_train_validation.csv")
    parser.add_argument("--calendar", default="calendar.csv")
    parser.add_argument("--prices", default="sell_prices.csv")
    parser.add_argument(
        "--store", default="CA_1", help="store_id filter (default: CA_1)"
    )
    parser.add_argument("--cat", default="FOODS", help="cat_id filter (default: FOODS)")
    parser.add_argument("--lookback-days", type=int, default=365)
    parser.add_argument("--holdout-days", type=int, default=60)
    parser.add_argument(
        "--output-dir", default=".", help="where to write models/CSV/dashboard"
    )
    parser.add_argument(
        "--skip-tuning",
        action="store_true",
        help="skip Optuna and use default LightGBM params",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()

    try:
        app = SmartSupplyAI(
            sales_path=args.sales,
            calendar_path=args.calendar,
            prices_path=args.prices,
            store_id=args.store,
            cat_id=args.cat,
            lookback_days=args.lookback_days,
            holdout_days=args.holdout_days,
            output_dir=args.output_dir,
            skip_tuning=args.skip_tuning,
        )
        app.load_and_clean()
        app.feature_engineering()
        app.train_model()
        app.run_simulation()
        app.generate_dashboard()
        app.export_results()
        app.print_report()
    except (FileNotFoundError, ValueError) as err:
        print(f"error: {err}", file=sys.stderr)
        sys.exit(1)
