import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
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


class SmartSupplyAI:
    def __init__(self, sales_path, calendar_path, prices_path):
        self.sales_path = sales_path
        self.calendar_path = calendar_path
        self.prices_path = prices_path
        self.df = None
        self.val_df = None
        self.model = None
        self.model_q75 = None
        self.model_q90 = None
        self.features = None
        self.simulation = None
        self.cv_scores = []
        self.best_params = {}

    def load_and_clean(self):
        print("[1/7] Loading M5 datasets...")
        start_time = time.time()

        cal = pd.read_csv(self.calendar_path, parse_dates=['date'])
        cal = cal[['date', 'wm_yr_wk', 'd', 'snap_CA']]

        prices = pd.read_csv(self.prices_path)

        sales = pd.read_csv(self.sales_path)
        sales = sales[(sales['store_id'] == 'CA_1') & (sales['cat_id'] == 'FOODS')]

        d_cols = [c for c in sales.columns if 'd_' in c][-365:]
        id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id']

        df = pd.melt(sales, id_vars=id_vars, value_vars=d_cols, var_name='d', value_name='sales')
        df = df.merge(cal, on='d', how='left')
        df = df.merge(prices, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')

        self.df = df
        print(f"   Processed {len(self.df):,} rows in {time.time() - start_time:.2f}s")

    def feature_engineering(self):
        print("[2/7] Engineering features...")
        df = self.df.sort_values(['id', 'date']).copy()

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
        df['price_prev_week'] = df.groupby('id')['sell_price'].shift(7)
        df['price_changed'] = (df['sell_price'] != df['price_prev_week']).astype(int)

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

        # categorical id features (LightGBM handles)
        for col in ['item_id', 'dept_id', 'cat_id', 'store_id']:
            df[col] = df[col].astype('category')

        self.df = df.dropna()

    def _get_feature_list(self):
        return [
            # categorical (LightGBM native)
            'item_id', 'dept_id',
            # lags
            'sales_lag_7', 'sales_lag_14', 'sales_lag_28',
            # rolling stats
            'rolling_mean_7', 'rolling_mean_28', 'rolling_std_7',
            'rolling_min_7', 'rolling_max_7',
            # price
            'sell_price', 'price_discount', 'price_changed',
            # calendar
            'day_of_week', 'is_weekend', 'day_of_month', 'week_of_year', 'month',
            'snap_CA',
            # fourier
            'sin_dow', 'cos_dow', 'sin_doy', 'cos_doy',
            # sparsity
            'days_since_zero',
        ]

    def _tune_hyperparams(self, train_data, val_data, features):
        """Run Optuna to find good LightGBM hyperparameters."""
        if not HAS_OPTUNA:
            print("   optuna not installed, using default params")
            return {
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

        train_set = lgb.Dataset(train_data[features], train_data['sales'])
        val_set = lgb.Dataset(val_data[features], val_data['sales'])

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
                params, train_set, num_boost_round=300,
                valid_sets=[val_set],
                callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)]
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

        cutoff = self.df['date'].max() - pd.Timedelta(days=60)
        train = self.df[self.df['date'] <= cutoff]
        val = self.df[self.df['date'] > cutoff]

        # time-series cross-validation on training set for reliable error estimate
        print("   running time-series cross-validation (4 folds)...")
        tscv = TimeSeriesSplit(n_splits=4)
        train_sorted = train.sort_values('date')
        cv_scores = []
        for fold, (tr_idx, te_idx) in enumerate(tscv.split(train_sorted)):
            tr_fold = train_sorted.iloc[tr_idx]
            te_fold = train_sorted.iloc[te_idx]
            ds_tr = lgb.Dataset(tr_fold[self.features], tr_fold['sales'])
            ds_te = lgb.Dataset(te_fold[self.features], te_fold['sales'])
            fold_params = {
                'objective': 'poisson', 'metric': 'mae',
                'learning_rate': 0.05, 'num_leaves': 64, 'min_data_in_leaf': 50,
                'subsample': 0.8, 'feature_fraction': 0.9,
                'force_col_wise': True, 'bagging_freq': 1, 'seed': 42, 'verbose': -1,
            }
            m = lgb.train(fold_params, ds_tr, num_boost_round=300,
                          valid_sets=[ds_te],
                          callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])
            preds = m.predict(te_fold[self.features])
            fold_mae = mean_absolute_error(te_fold['sales'], preds)
            cv_scores.append(fold_mae)

        self.cv_scores = cv_scores
        print(f"   CV MAE per fold: {[round(s, 3) for s in cv_scores]}")
        print(f"   CV MAE mean: {np.mean(cv_scores):.3f} +/- {np.std(cv_scores):.3f}")

        # hyperparameter tuning on train/val split
        print("   tuning hyperparameters...")
        self.best_params = self._tune_hyperparams(train, val, self.features)

        # train final point-prediction model with tuned params
        train_set = lgb.Dataset(train[self.features], train['sales'])
        val_set = lgb.Dataset(val[self.features], val['sales'])

        self.model = lgb.train(
            self.best_params, train_set, num_boost_round=500,
            valid_sets=[val_set],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
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
                q_params, train_set, num_boost_round=500,
                valid_sets=[val_set],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )
            setattr(self, attr, q_model)

        self.val_df = val
        joblib.dump(self.model, 'lightgbm_model.pkl')

    def run_simulation(self):
        print("[4/7] Simulating with realistic dept-level costs...")
        val_df = self.val_df.copy()

        # point prediction and quantile predictions
        val_df['pred_demand'] = self.model.predict(val_df[self.features])
        val_df['pred_q75'] = self.model_q75.predict(val_df[self.features])
        val_df['pred_q90'] = self.model_q90.predict(val_df[self.features])

        # dept-level cost of goods and spoilage
        dept_str = val_df['dept_id'].astype(str)
        val_df['cogs_ratio'] = dept_str.map(DEPT_COGS_RATIO).fillna(0.60)
        val_df['spoilage_rate'] = dept_str.map(DEPT_SPOILAGE_RATE).fillna(0.10)

        val_df['cost_price'] = val_df['sell_price'] * val_df['cogs_ratio']
        val_df['margin'] = val_df['sell_price'] - val_df['cost_price']

        # confidence-based safety stock: order the q90 prediction (upper bound)
        # but never less than the point prediction
        raw_stock = np.maximum(val_df['pred_demand'], val_df['pred_q90'])
        val_df['stock_decision'] = np.clip(np.round(raw_stock), 0, None)

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

        # oracle baseline (considering perfect foresight, no waste or stockouts)
        val_df['oracle_profit'] = val_df['sales'] * val_df['margin']

        # human baseline: EWMA(7) blended with seasonal naive, 1.1x buffer
        human_demand = 0.6 * val_df['ewma_7'] + 0.4 * val_df['seasonal_naive']
        val_df['human_stock'] = np.clip(np.round(human_demand * 1.10), 0, None)
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

    def generate_dashboard(self):
        print("[5/7] Generating dashboard...")
        sim = self.simulation

        fig = plt.figure(figsize=(16, 12))
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

        plt.tight_layout()
        plt.savefig("smart_supply_dashboard.png", dpi=150)
        print("   Dashboard saved to 'smart_supply_dashboard.png'")

    def export_results(self):
        print("[6/7] Exporting forecasts...")
        cols = [
            'date', 'item_id', 'dept_id', 'sales',
            'pred_demand', 'pred_q75', 'pred_q90',
            'stock_decision', 'waste', 'missed_sales',
            'waste_cost', 'stockout_cost',
            'net_profit', 'sma_profit', 'oracle_profit',
        ]
        self.simulation[cols].to_csv("m5_supply_chain_forecast.csv", index=False)
        print("   CSV saved to 'm5_supply_chain_forecast.csv'")

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
        print(f"Human Baseline Profit (EWMA):  ${human_profit:>12,.2f}")
        print(f"ML Model Profit (LightGBM):    ${ml_profit:>12,.2f}")
        print(f"Profit Lift (ML - Human):       ${ml_profit - human_profit:>12,.2f}")
        print(f"Waste Cost Saved:               ${waste_savings:>12,.2f}")
        print(f"Stockout Cost Saved:            ${stockout_savings:>12,.2f}")
        print(f"Theoretical Max Profit:        ${oracle_profit:>12,.2f}")
        print("-" * 55)
        print(f"ML Stockout Rate:               {ml_stockout_rate:>11.1f}%")
        print(f"Human Stockout Rate:            {human_stockout_rate:>11.1f}%")
        print(f"CV MAE (4-fold):                {np.mean(self.cv_scores):>11.3f} +/- {np.std(self.cv_scores):.3f}")
        print("-" * 55)

        # per-department breakdown
        dept_summary = sim.groupby('dept_id').agg(
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


if __name__ == "__main__":

    app = SmartSupplyAI(
        sales_path="sales_train_validation.csv",
        calendar_path="calendar.csv",
        prices_path="sell_prices.csv"
    )

    app.load_and_clean()
    app.feature_engineering()
    app.train_model()
    app.run_simulation()
    app.print_report()
    app.generate_dashboard()
    app.export_results()
