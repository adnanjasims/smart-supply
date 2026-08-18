import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error
import warnings
import time
import joblib

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

warnings.filterwarnings('ignore')
plt.style.use('ggplot')
pd.options.mode.chained_assignment = None

# dept-level cost and spoilage assumptions
DEPT_COGS_RATIO = {
    'FOODS_1': 0.70,  # produce/perishable, thin margins
    'FOODS_2': 0.60,  # dairy/deli, medium margins
    'FOODS_3': 0.50,  # packaged/canned, higher margins
}
DEPT_SPOILAGE_RATE = {
    'FOODS_1': 0.20,  # perishable items spoil fast
    'FOODS_2': 0.10,  # dairy/deli moderate spoilage
    'FOODS_3': 0.01,  # shelf-stable, almost no daily spoilage
}
# lost-sale penalty: each missed unit costs this fraction of the margin
STOCKOUT_PENALTY_FRACTION = 0.75

REQUIRED_SALES_COLS = {'id', 'item_id', 'dept_id', 'cat_id', 'store_id'}
REQUIRED_CAL_COLS = {'date', 'wm_yr_wk', 'd', 'snap_CA'}
REQUIRED_PRICE_COLS = {'store_id', 'item_id', 'wm_yr_wk', 'sell_price'}


def _lgb_callbacks(stopping_rounds):
    """Quiet early-stopping callbacks (avoids noisy per-round console spam)."""
    return [
        lgb.early_stopping(stopping_rounds, verbose=False),
        lgb.log_evaluation(0),
    ]


class SmartSupplyAI:
    def __init__(self, sales_path, calendar_path, prices_path,
                 store_id='CA_1', cat_id='FOODS', lookback_days=365, holdout_days=60,
                 output_dir='.', n_cv_splits=4, num_boost_round=500,
                 cv_boost_round=300, skip_tuning=False):
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
        self.features = None
        self.simulation = None
        self.cv_scores = []
        self.best_params = {}
        self.used_human_fallback = False
        self.drop_stats = {}

    def _validate_inputs(self):
        """Fail early with a clear message if data files are missing or incomplete."""
        checks = [
            (self.sales_path, 'sales'),
            (self.calendar_path, 'calendar'),
            (self.prices_path, 'prices'),
        ]
        missing = [f"{label}: {path}" for path, label in checks if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing required M5 file(s):\n  - " + "\n  - ".join(missing)
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

        cal = pd.read_csv(self.calendar_path, parse_dates=['date'])
        self._require_columns(cal, REQUIRED_CAL_COLS, 'calendar.csv')
        cal_cols = ['date', 'wm_yr_wk', 'd', 'snap_CA']
        for optional in ['event_name_1', 'event_type_1']:
            if optional in cal.columns:
                cal_cols.append(optional)
        cal = cal[cal_cols]

        prices = pd.read_csv(self.prices_path)
        self._require_columns(prices, REQUIRED_PRICE_COLS, 'sell_prices.csv')

        sales = pd.read_csv(self.sales_path)
        self._require_columns(sales, REQUIRED_SALES_COLS, 'sales_train_validation.csv')
        sales = sales[(sales['store_id'] == self.store_id) & (sales['cat_id'] == self.cat_id)]
        if sales.empty:
            raise ValueError(
                f"No rows left after filtering store_id={self.store_id!r} and cat_id={self.cat_id!r}."
            )

        d_cols = [c for c in sales.columns if c.startswith('d_')]
        if len(d_cols) < 28:
            raise ValueError(f"Need at least 28 day columns, found {len(d_cols)}.")
        d_cols = d_cols[-self.lookback_days:]
        id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id']

        df = pd.melt(sales, id_vars=id_vars, value_vars=d_cols, var_name='d', value_name='sales')
        df = df.merge(cal, on='d', how='left')
        df = df.merge(prices, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')

        n_price_missing = int(df['sell_price'].isna().sum())
        self.drop_stats['missing_sell_price'] = n_price_missing
        if n_price_missing:
            pct = 100 * n_price_missing / len(df)
            print(f"   note: {n_price_missing:,} rows ({pct:.1f}%) have no sell_price "
                  f"(item not priced that week in M5). they are skipped later.")

        self.df = df
        print(f"   store={self.store_id} cat={self.cat_id} | "
              f"{df['id'].nunique():,} items x {df['date'].nunique()} days = "
              f"{len(self.df):,} rows in {time.time() - start_time:.2f}s")

    def feature_engineering(self):
        print("[2/7] Engineering features...")
        df = self.df.sort_values(['id', 'date']).copy()
        before = len(df)

        # lag features
        df['sales_lag_7'] = df.groupby('id')['sales'].shift(7)
        df['sales_lag_14'] = df.groupby('id')['sales'].shift(14)
        df['sales_lag_28'] = df.groupby('id')['sales'].shift(28)

        # rolling statistics
        df['rolling_mean_7'] = df.groupby('id')['sales'].transform(lambda x: x.shift(1).rolling(7).mean())
        df['rolling_std_7'] = df.groupby('id')['sales'].transform(lambda x: x.shift(1).rolling(7).std())
        df['rolling_mean_28'] = df.groupby('id')['sales'].transform(lambda x: x.shift(1).rolling(28).mean())
        df['rolling_min_7'] = df.groupby('id')['sales'].transform(lambda x: x.shift(1).rolling(7).min())
        df['rolling_max_7'] = df.groupby('id')['sales'].transform(lambda x: x.shift(1).rolling(7).max())

        # EWMA for stronger human baseline (computed here so it's available during simulation)
        df['ewma_7'] = df.groupby('id')['sales'].transform(lambda x: x.shift(1).ewm(span=7).mean())
        # seasonal naive: same weekday last week
        df['seasonal_naive'] = df.groupby('id')['sales'].shift(7)

        # price momentum and discount
        df['price_max'] = df.groupby('item_id')['sell_price'].transform('max')
        df['price_discount'] = df['sell_price'] / df['price_max']
        # price change flag: did price change vs. last week?
        # treat missing prev-week price as "no change" (NaN != x is True in pandas otherwise)
        df['price_prev_week'] = df.groupby('id')['sell_price'].shift(7)
        df['price_changed'] = (
            df['sell_price'].notna()
            & df['price_prev_week'].notna()
            & (df['sell_price'] != df['price_prev_week'])
        ).astype(int)

        # days since last zero-sale per item (sparsity signal)
        df['is_zero'] = (df['sales'] == 0).astype(int)
        df['days_since_zero'] = df.groupby('id')['is_zero'].transform(
            lambda x: x.groupby((x != x.shift()).cumsum()).cumcount()
        )
        df.drop(columns=['is_zero'], inplace=True)

        # date features
        df['day_of_week'] = df['date'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].ge(5).astype(int)
        df['day_of_month'] = df['date'].dt.day
        df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
        df['month'] = df['date'].dt.month
        df['day_of_year'] = df['date'].dt.dayofyear

        # fourier features for weekly and annual seasonality
        df['sin_dow'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['cos_dow'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['sin_doy'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
        df['cos_doy'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)

        # holiday / event flags from the M5 calendar
        if 'event_name_1' in df.columns:
            df['has_event'] = df['event_name_1'].notna().astype(int)
            df['event_type'] = df['event_type_1'].fillna('None').astype('category')
        else:
            df['has_event'] = 0
            df['event_type'] = pd.Categorical(['None'] * len(df))

        # categorical id features (LightGBM handles)
        for col in ['item_id', 'dept_id', 'cat_id', 'store_id']:
            df[col] = df[col].astype('category')

        # lag_28 / rolling_28 need ~28 prior days per item. those early rows are
        # expected NaNs (warmup), not bad data. missing prices are a separate slice.
        model_cols = self._get_feature_list() + ['sales', 'sell_price', 'ewma_7', 'seasonal_naive']
        keep_cols = [c for c in model_cols if c in df.columns]

        warmup_mask = df['sales_lag_28'].isna() | df['rolling_mean_28'].isna()
        price_mask = df['sell_price'].isna()
        n_warmup = int(warmup_mask.sum())
        n_price_only = int((~warmup_mask & price_mask).sum())
        other_mask = df[keep_cols].isna().any(axis=1) & ~warmup_mask & ~price_mask
        n_other = int(other_mask.sum())

        self.df = df.dropna(subset=keep_cols).copy()
        dropped = before - len(self.df)
        self.drop_stats.update({
            'rows_before': before,
            'rows_after': len(self.df),
            'dropped_total': dropped,
            'dropped_lag_warmup': n_warmup,
            'dropped_missing_price_only': n_price_only,
            'dropped_other': n_other,
        })
        print(
            f"   kept {len(self.df):,} / {before:,} rows "
            f"(dropped {dropped:,}: ~{n_warmup:,} lag/rolling warmup, "
            f"{n_price_only:,} missing price, {n_other:,} other)"
        )
        if before and dropped / before > 0.25:
            print("   note: large warmup drop is normal when lookback is short relative to a 28-day lag.")

    def _get_feature_list(self):
        return [
            # categorical (LightGBM native)
            'item_id', 'dept_id', 'event_type',
            # lags
            'sales_lag_7', 'sales_lag_14', 'sales_lag_28',
            # rolling stats
            'rolling_mean_7', 'rolling_mean_28', 'rolling_std_7',
            'rolling_min_7', 'rolling_max_7',
            # price
            'sell_price', 'price_discount', 'price_changed',
            # calendar
            'day_of_week', 'is_weekend', 'day_of_month', 'week_of_year', 'month',
            'snap_CA', 'has_event',
            # fourier
            'sin_dow', 'cos_dow', 'sin_doy', 'cos_doy',
            # sparsity
            'days_since_zero',
        ]

    def _date_fold_splits(self, train, n_splits=4):
        """Expanding-window CV by calendar date (avoids panel row leakage)."""
        unique_dates = np.sort(train['date'].unique())
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
            tr_fold = train[train['date'] <= train_end]
            te_fold = train[(train['date'] >= test_start) & (train['date'] <= test_end)]
            if len(tr_fold) == 0 or len(te_fold) == 0:
                continue
            folds.append((tr_fold, te_fold))
        return folds

    def _tune_hyperparams(self, train_data, val_data, features):
        """Run Optuna to find good LightGBM hyperparameters."""
        defaults = {
            'objective': 'poisson',
            'metric': 'mae',
            'learning_rate': 0.05,
            'subsample': 0.8,
            'feature_fraction': 0.9,
            'num_leaves': 64,
            'min_data_in_leaf': 50,
            'force_col_wise': True,
            'bagging_freq': 1,
            'seed': 42,
            'verbose': -1,
        }
        if self.skip_tuning or not HAS_OPTUNA:
            reason = "skip_tuning=True" if self.skip_tuning else "optuna not installed"
            print(f"   {reason}, using default params")
            return defaults

        train_set = lgb.Dataset(train_data[features], train_data['sales'],
                                categorical_feature=['item_id', 'dept_id', 'event_type'])
        val_set = lgb.Dataset(val_data[features], val_data['sales'],
                              categorical_feature=['item_id', 'dept_id', 'event_type'],
                              reference=train_set)

        def objective(trial):
            params = {
                'objective': 'poisson',
                'metric': 'mae',
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 16, 128),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
                'force_col_wise': True,
                'bagging_freq': 1,
                'seed': 42,
                'verbose': -1,
            }
            model = lgb.train(
                params, train_set, num_boost_round=min(300, self.cv_boost_round),
                valid_sets=[val_set],
                callbacks=_lgb_callbacks(30),
            )
            preds = model.predict(val_data[features])
            return mean_absolute_error(val_data['sales'], preds)

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=25, show_progress_bar=False)

        best = study.best_params
        best.update({
            'objective': 'poisson',
            'metric': 'mae',
            'force_col_wise': True,
            'bagging_freq': 1,
            'seed': 42,
            'verbose': -1,
        })
        print(f"   best trial MAE: {study.best_value:.3f}")
        return best

    def train_model(self):
        print("[3/7] Training model (LightGBM with tuning)...")
        self.features = self._get_feature_list()
        cat_features = ['item_id', 'dept_id', 'event_type']

        cutoff = self.df['date'].max() - pd.Timedelta(days=self.holdout_days)
        train = self.df[self.df['date'] <= cutoff]
        val = self.df[self.df['date'] > cutoff]
        if train.empty or val.empty:
            raise ValueError(
                f"Bad train/val split with holdout_days={self.holdout_days}. "
                "Try a smaller holdout or more lookback."
            )

        # date-based expanding-window CV (no panel row leakage)
        print(f"   running date-based time-series cross-validation ({self.n_cv_splits} folds)...")
        cv_scores = []
        fold_params = {
            'objective': 'poisson', 'metric': 'mae',
            'learning_rate': 0.05, 'num_leaves': 64, 'min_data_in_leaf': 50,
            'subsample': 0.8, 'feature_fraction': 0.9,
            'force_col_wise': True, 'bagging_freq': 1, 'seed': 42, 'verbose': -1,
        }
        for fold, (tr_fold, te_fold) in enumerate(
            self._date_fold_splits(train, n_splits=self.n_cv_splits), start=1
        ):
            ds_tr = lgb.Dataset(tr_fold[self.features], tr_fold['sales'],
                                categorical_feature=cat_features)
            ds_te = lgb.Dataset(te_fold[self.features], te_fold['sales'],
                                categorical_feature=cat_features, reference=ds_tr)
            m = lgb.train(
                fold_params, ds_tr, num_boost_round=self.cv_boost_round,
                valid_sets=[ds_te],
                callbacks=_lgb_callbacks(30),
            )
            preds = m.predict(te_fold[self.features])
            fold_mae = mean_absolute_error(te_fold['sales'], preds)
            cv_scores.append(fold_mae)
            print(f"   fold {fold}: MAE={fold_mae:.3f} "
                  f"(train through {tr_fold['date'].max().date()}, "
                  f"test {te_fold['date'].min().date()} -> {te_fold['date'].max().date()})")

        self.cv_scores = cv_scores
        if cv_scores:
            print(f"   CV MAE mean: {np.mean(cv_scores):.3f} +/- {np.std(cv_scores):.3f}")

        # hyperparameter tuning on train/val split
        print("   tuning hyperparameters...")
        self.best_params = self._tune_hyperparams(train, val, self.features)

        # train final point-prediction model with tuned params
        train_set = lgb.Dataset(train[self.features], train['sales'],
                                categorical_feature=cat_features)
        val_set = lgb.Dataset(val[self.features], val['sales'],
                              categorical_feature=cat_features, reference=train_set)

        self.model = lgb.train(
            self.best_params, train_set, num_boost_round=self.num_boost_round,
            valid_sets=[val_set],
            callbacks=_lgb_callbacks(50),
        )

        val_pred = self.model.predict(val[self.features])
        mae = mean_absolute_error(val['sales'], val_pred)
        print(f"   final model validation MAE: {mae:.3f}")

        # quantile models for confidence-based safety stock
        print("   training quantile models (q75, q90)...")
        for alpha, attr in [(0.75, 'model_q75'), (0.90, 'model_q90')]:
            q_params = dict(self.best_params)
            q_params['objective'] = 'quantile'
            q_params['alpha'] = alpha
            q_params['metric'] = 'quantile'
            q_model = lgb.train(
                q_params, train_set, num_boost_round=self.num_boost_round,
                valid_sets=[val_set],
                callbacks=_lgb_callbacks(50),
            )
            setattr(self, attr, q_model)

        self.val_df = val
        self._save_models()

    def _save_models(self):
        """Persist point + quantile models so the dashboard and future runs can reuse them."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, self.output_dir / 'lightgbm_model.pkl')
        joblib.dump(self.model_q75, self.output_dir / 'lightgbm_model_q75.pkl')
        joblib.dump(self.model_q90, self.output_dir / 'lightgbm_model_q90.pkl')
        joblib.dump({
            'point': self.model,
            'q75': self.model_q75,
            'q90': self.model_q90,
            'features': self.features,
            'best_params': self.best_params,
            'cv_scores': self.cv_scores,
        }, self.output_dir / 'lightgbm_bundle.pkl')
        print(f"   saved models under {self.output_dir.resolve()}")

    def _human_stock(self, df):
        human_demand = 0.6 * df['ewma_7'] + 0.4 * df['seasonal_naive']
        return np.clip(np.round(human_demand * 1.10), 0, None)

    def _dept_aware_stock(self, df):
        """
        Perishables (FOODS_1): tighter buffer (q75) to cut spoilage.
        Everything else: q90 upper bound is fine since waste is cheaper.
        Never order below the point forecast.
        """
        dept = df['dept_id'].astype(str)
        perishable = dept.eq('FOODS_1')
        tight = np.maximum(df['pred_demand'], df['pred_q75'])
        wide = np.maximum(df['pred_demand'], df['pred_q90'])
        raw = np.where(perishable, tight, wide)
        return np.clip(np.round(raw), 0, None)

    def run_simulation(self):
        print("[4/7] Simulating with realistic dept-level costs...")
        val_df = self.val_df.copy()

        try:
            val_df['pred_demand'] = self.model.predict(val_df[self.features])
            val_df['pred_q75'] = self.model_q75.predict(val_df[self.features])
            val_df['pred_q90'] = self.model_q90.predict(val_df[self.features])
            val_df['stock_decision'] = self._dept_aware_stock(val_df)
            self.used_human_fallback = False
        except Exception as exc:
            print(f"   warning: model predict failed ({exc}). falling back to human EWMA strategy.")
            val_df['pred_demand'] = 0.6 * val_df['ewma_7'] + 0.4 * val_df['seasonal_naive']
            val_df['pred_q75'] = val_df['pred_demand']
            val_df['pred_q90'] = val_df['pred_demand'] * 1.10
            val_df['stock_decision'] = self._human_stock(val_df)
            self.used_human_fallback = True

        # dept-level cost of goods and spoilage
        dept_str = val_df['dept_id'].astype(str)
        val_df['cogs_ratio'] = dept_str.map(DEPT_COGS_RATIO).fillna(0.60)
        val_df['spoilage_rate'] = dept_str.map(DEPT_SPOILAGE_RATE).fillna(0.10)

        val_df['cost_price'] = val_df['sell_price'] * val_df['cogs_ratio']
        val_df['margin'] = val_df['sell_price'] - val_df['cost_price']

        # ML strategy financials
        val_df['units_sold'] = np.minimum(val_df['stock_decision'], val_df['sales'])
        val_df['unsold_inventory'] = np.maximum(0, val_df['stock_decision'] - val_df['sales'])
        val_df['missed_sales'] = np.maximum(0, val_df['sales'] - val_df['stock_decision'])

        val_df['waste'] = val_df['unsold_inventory'] * val_df['spoilage_rate']
        val_df['revenue'] = val_df['units_sold'] * val_df['sell_price']
        val_df['cogs'] = val_df['units_sold'] * val_df['cost_price']
        val_df['waste_cost'] = val_df['waste'] * val_df['cost_price']
        val_df['stockout_cost'] = val_df['missed_sales'] * val_df['margin'] * STOCKOUT_PENALTY_FRACTION

        val_df['net_profit'] = val_df['revenue'] - val_df['cogs'] - val_df['waste_cost'] - val_df['stockout_cost']

        # oracle baseline (perfect foresight, no waste or stockouts)
        val_df['oracle_profit'] = val_df['sales'] * val_df['margin']

        # human baseline: EWMA(7) blended with seasonal naive, 1.1x buffer
        val_df['human_stock'] = self._human_stock(val_df)
        val_df['human_sold'] = np.minimum(val_df['human_stock'], val_df['sales'])
        val_df['human_unsold'] = np.maximum(0, val_df['human_stock'] - val_df['sales'])
        val_df['human_missed'] = np.maximum(0, val_df['sales'] - val_df['human_stock'])

        val_df['human_waste'] = val_df['human_unsold'] * val_df['spoilage_rate']
        val_df['human_waste_cost'] = val_df['human_waste'] * val_df['cost_price']
        val_df['human_stockout_cost'] = val_df['human_missed'] * val_df['margin'] * STOCKOUT_PENALTY_FRACTION

        val_df['sma_profit'] = (
            val_df['human_sold'] * val_df['sell_price']
            - val_df['human_sold'] * val_df['cost_price']
            - val_df['human_waste_cost']
            - val_df['human_stockout_cost']
        )

        self.simulation = val_df

    def collect_metrics(self) -> dict:
        """Snapshot of validation + business metrics for MLOps tracking."""
        if self.simulation is None:
            raise ValueError("run_simulation() before collect_metrics()")
        sim = self.simulation
        ml_profit = float(sim["net_profit"].sum())
        human_profit = float(sim["sma_profit"].sum())
        oracle_profit = float(sim["oracle_profit"].sum())
        val_mae = float((sim["pred_demand"] - sim["sales"]).abs().mean())
        q90_ge_q75 = float((sim["pred_q90"] >= sim["pred_q75"] - 1e-9).mean())
        metrics = {
            "val_mae": val_mae,
            "cv_mae_mean": float(np.mean(self.cv_scores)) if self.cv_scores else None,
            "cv_mae_std": float(np.std(self.cv_scores)) if self.cv_scores else None,
            "ml_profit": ml_profit,
            "human_profit": human_profit,
            "oracle_profit": oracle_profit,
            "profit_lift": ml_profit - human_profit,
            "waste_cost": float(sim["waste_cost"].sum()),
            "human_waste_cost": float(sim["human_waste_cost"].sum()),
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
        return metrics

    def generate_dashboard(self):
        print("[5/7] Generating dashboard...")
        sim = self.simulation

        fig = plt.figure(figsize=(16, 12), constrained_layout=True)
        gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

        # 1. profit comparison
        ax1 = fig.add_subplot(gs[0, 0])
        profits = [sim['sma_profit'].sum(), sim['net_profit'].sum(), sim['oracle_profit'].sum()]
        labels = ['Human Strategy\n(EWMA + Seasonal)', 'ML Strategy\n(LightGBM + Quantile)', 'Theoretical Max\n(Perfect)']
        colors = ['#7f8c8d', '#2980b9', '#27ae60']
        bars = ax1.bar(labels, profits, color=colors)
        ax1.set_title("Net Profit Comparison (60-Day Test)")
        ax1.set_ylabel("Profit ($)")
        ax1.bar_label(bars, fmt='$%.0f', padding=3)

        # 2. waste comparison
        ax2 = fig.add_subplot(gs[0, 1])
        waste = [sim['human_waste_cost'].sum(), sim['waste_cost'].sum()]
        bars2 = ax2.bar(['Human Strategy', 'ML Strategy'], waste, color=['#c0392b', '#2980b9'])
        ax2.set_title("Total Waste Cost")
        ax2.set_ylabel("$ Lost to Spoilage")
        ax2.bar_label(bars2, fmt='$%.0f', padding=3)

        # 3. stockout comparison
        ax3 = fig.add_subplot(gs[1, 0])
        stockouts = [sim['human_stockout_cost'].sum(), sim['stockout_cost'].sum()]
        bars3 = ax3.bar(['Human Strategy', 'ML Strategy'], stockouts, color=['#e67e22', '#2980b9'])
        ax3.set_title("Lost-Sale Cost (Stockouts)")
        ax3.set_ylabel("$ Lost to Stockouts")
        ax3.bar_label(bars3, fmt='$%.0f', padding=3)

        # 4. prediction error distribution
        ax4 = fig.add_subplot(gs[1, 1])
        errors = sim['pred_demand'] - sim['sales']
        sns.histplot(errors, kde=True, ax=ax4, color='#8e44ad', bins=40)
        ax4.set_xlim(-10, 10)
        ax4.set_title("Prediction Error Distribution")
        ax4.set_xlabel("Error (Predicted - Actual units)")

        # 5. daily profit tracking
        ax5 = fig.add_subplot(gs[2, 0])
        daily_profit = sim.groupby('date')[['net_profit', 'sma_profit']].sum().reset_index()
        sns.lineplot(data=daily_profit, x='date', y='net_profit', label='ML Model', ax=ax5, color='#2980b9')
        sns.lineplot(data=daily_profit, x='date', y='sma_profit', label='Human (EWMA)', ax=ax5, color='#7f8c8d')
        ax5.set_title("Daily Profit Trajectory")
        ax5.set_xlabel("Date")
        ax5.set_ylabel("Daily Profit ($)")
        plt.setp(ax5.get_xticklabels(), rotation=45)

        # 6. feature importance (top 15)
        ax6 = fig.add_subplot(gs[2, 1])
        importance = self.model.feature_importance(importance_type='gain')
        feat_imp = pd.Series(importance, index=self.features).nlargest(15)
        feat_imp.sort_values().plot.barh(ax=ax6, color='#2980b9')
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
            'date', 'item_id', 'dept_id', 'sales',
            'pred_demand', 'pred_q75', 'pred_q90',
            'stock_decision', 'waste', 'missed_sales',
            'waste_cost', 'stockout_cost',
            'human_waste_cost', 'human_stockout_cost', 'human_missed',
            'net_profit', 'sma_profit', 'oracle_profit',
        ]
        self.output_dir.mkdir(parents=True, exist_ok=True)
        out = self.output_dir / "m5_supply_chain_forecast.csv"
        self.simulation[cols].to_csv(out, index=False)
        print(f"   CSV saved to '{out}'")

    def print_report(self):
        print("[7/7] Impact report...")
        sim = self.simulation
        ml_profit = sim['net_profit'].sum()
        human_profit = sim['sma_profit'].sum()
        oracle_profit = sim['oracle_profit'].sum()
        waste_savings = sim['human_waste_cost'].sum() - sim['waste_cost'].sum()
        stockout_savings = sim['human_stockout_cost'].sum() - sim['stockout_cost'].sum()

        ml_stockout_rate = (sim['missed_sales'] > 0).mean() * 100
        human_stockout_rate = (sim['human_missed'] > 0).mean() * 100

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
            print(f"CV MAE (4-fold, by date):       {np.mean(self.cv_scores):>11.3f} +/- {np.std(self.cv_scores):.3f}")
        print("-" * 55)

        # per-department breakdown
        dept_summary = sim.groupby('dept_id', observed=True).agg(
            ml_profit=('net_profit', 'sum'),
            human_profit=('sma_profit', 'sum'),
            ml_waste=('waste_cost', 'sum'),
            human_waste=('human_waste_cost', 'sum'),
            ml_stockout=('stockout_cost', 'sum'),
            human_stockout=('human_stockout_cost', 'sum'),
        ).reset_index()
        dept_summary['profit_lift'] = dept_summary['ml_profit'] - dept_summary['human_profit']
        print("\nPER-DEPARTMENT BREAKDOWN:")
        print(dept_summary.to_string(index=False, float_format='${:,.0f}'.format))
        print("=" * 55)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train LightGBM demand models and simulate grocery ordering on M5 data."
    )
    parser.add_argument('--sales', default='sales_train_validation.csv')
    parser.add_argument('--calendar', default='calendar.csv')
    parser.add_argument('--prices', default='sell_prices.csv')
    parser.add_argument('--store', default='CA_1', help='store_id filter (default: CA_1)')
    parser.add_argument('--cat', default='FOODS', help='cat_id filter (default: FOODS)')
    parser.add_argument('--lookback-days', type=int, default=365)
    parser.add_argument('--holdout-days', type=int, default=60)
    parser.add_argument('--output-dir', default='.', help='where to write models/CSV/dashboard')
    parser.add_argument('--skip-tuning', action='store_true',
                        help='skip Optuna and use default LightGBM params')
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
