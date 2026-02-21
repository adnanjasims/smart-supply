import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error
import warnings
import time

warnings.filterwarnings('ignore')
plt.style.use('ggplot')
pd.options.mode.chained_assignment = None

class SmartSupplyAI:
    def __init__(self, sales_path, calendar_path, prices_path):
        self.sales_path = sales_path
        self.calendar_path = calendar_path
        self.prices_path = prices_path
        self.df = None
        self.val_df = None
        self.model = None
        self.features = None
        self.simulation = None

    def load_and_clean(self):
        print("[1/6] Loading M5 datasets (this may take a moment)...")
        start_time = time.time()

        # load calendar and prices
        cal = pd.read_csv(self.calendar_path, parse_dates=['date'])
        cal = cal[['date', 'wm_yr_wk', 'd', 'snap_CA']]

        prices = pd.read_csv(self.prices_path)

        # load and filter sales data to save memory (store CA_1, FOODS category)
        sales = pd.read_csv(self.sales_path)
        sales = sales[(sales['store_id'] == 'CA_1') & (sales['cat_id'] == 'FOODS')]

        # melt from wide format to long format (limiting to the last 365 days of data)
        d_cols = [c for c in sales.columns if 'd_' in c][-365:]
        id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id']
        
        df = pd.melt(sales, id_vars=id_vars, value_vars=d_cols, var_name='d', value_name='sales')

        # merge dataframes to create one unified dataset
        df = df.merge(cal, on='d', how='left')
        df = df.merge(prices, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')

        self.df = df
        print(f"   Processed {len(self.df):,} rows in {time.time() - start_time:.2f}s")

    def feature_engineering(self):
        print("[2/6] Engineering time-series features...")
        df = self.df.sort_values(['id', 'date']).copy()

        # lag features (what did sales look like 7 and 28 days ago?)
        df['sales_lag_7'] = df.groupby('id')['sales'].shift(7)
        df['sales_lag_28'] = df.groupby('id')['sales'].shift(28)

        # rolling statistics for trend detection
        df['rolling_mean_7'] = df.groupby('id')['sales'].transform(lambda x: x.shift(1).rolling(7).mean())
        df['rolling_std_7'] = df.groupby('id')['sales'].transform(lambda x: x.shift(1).rolling(7).std())

        # price momentum and discount tracking
        df['price_max'] = df.groupby('item_id')['sell_price'].transform('max')
        df['price_discount'] = df['sell_price'] / df['price_max']

        # date specific features
        df['day_of_week'] = df['date'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].ge(5).astype(int)

        # drop rows with nan values caused by the lag shifting
        self.df = df.dropna()

    def train_model(self):
        print("[3/6] Training model (LightGBM)...")
        
        # split train and validation sets (using the last 60 days for validation)
        cutoff = self.df['date'].max() - pd.Timedelta(days=60)
        train = self.df[self.df['date'] <= cutoff]
        val = self.df[self.df['date'] > cutoff]

        self.features = ['sales_lag_7', 'sales_lag_28', 'rolling_mean_7', 'rolling_std_7', 
                         'sell_price', 'price_discount', 'day_of_week', 'is_weekend', 'snap_CA']

        train_set = lgb.Dataset(train[self.features], train['sales'])
        val_set = lgb.Dataset(val[self.features], val['sales'])

        # configure LightGBM parameters for poisson regression (best for item counts)
        params = {
            'objective': 'poisson',
            'metric': 'rmse',
            'learning_rate': 0.07,
            'subsample': 0.8,
            'force_col_wise': True,
            'verbose': -1
        }

        # train the model
        self.model = lgb.train(
            params, 
            train_set, 
            num_boost_round=500,
            valid_sets=[val_set],
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(0)]
        )

        self.val_df = val

    def run_simulation(self):
        print("[4/6] Simulating real-world profit logic...")
        val_df = self.val_df.copy()

        # predict daily demand
        val_df['pred_demand'] = self.model.predict(val_df[self.features])

        # calculate real costs and margins (assuming cost is 60 percent of retail price)
        val_df['cost_price'] = val_df['sell_price'] * 0.60
        val_df['margin'] = val_df['sell_price'] - val_df['cost_price']

        # dynamic safety stock calculation based on profit margins
        val_df['safety_factor'] = np.where(val_df['margin'] > 2.0, 1.25, 1.05)
        
        # FIX 1: Use normal rounding, not ceil. This stops ordering 1 unit for items with 0.1 demand.
        val_df['stock_decision'] = np.round(val_df['pred_demand'] * val_df['safety_factor'])

        # financial outcomes for the model
        val_df['units_sold'] = np.minimum(val_df['stock_decision'], val_df['sales'])
        val_df['unsold_inventory'] = np.maximum(0, val_df['stock_decision'] - val_df['sales'])
        
        # applying realistic 10% daily spoilage rate instead of throwing everything away
        SPOILAGE_RATE = 0.10 
        val_df['waste'] = val_df['unsold_inventory'] * SPOILAGE_RATE
        
        val_df['revenue'] = val_df['units_sold'] * val_df['sell_price']
        val_df['cogs'] = val_df['units_sold'] * val_df['cost_price']
        val_df['waste_cost'] = val_df['waste'] * val_df['cost_price']
        
        val_df['net_profit'] = val_df['revenue'] - val_df['cogs'] - val_df['waste_cost']

        # oracle baseline (what profit looks like with a crystal ball/perfect prediction)
        val_df['oracle_profit'] = (val_df['sales'] * val_df['sell_price']) - (val_df['sales'] * val_df['cost_price'])

        # human baseline (simulating a manager ordering based on a 7-day moving average)
        
        val_df['sma_stock'] = np.round(val_df['rolling_mean_7'] * 1.1)
        val_df['sma_sold'] = np.minimum(val_df['sma_stock'], val_df['sales'])
        val_df['sma_unsold'] = np.maximum(0, val_df['sma_stock'] - val_df['sales'])
        
        val_df['sma_waste'] = val_df['sma_unsold'] * SPOILAGE_RATE
        val_df['sma_waste_cost'] = val_df['sma_waste'] * val_df['cost_price']
        
        val_df['sma_profit'] = (val_df['sma_sold'] * val_df['sell_price']) - (val_df['sma_stock'] * val_df['cost_price']) + (val_df['sma_unsold'] * val_df['cost_price'] * (1 - SPOILAGE_RATE))
        # note on the line above: to calculate true profit, we retain the value of unsold goods that didn't spoil. 
        # for simplicity,an easier way to write both profit formulas is just: Revenue - COGS - Waste_Cost
        val_df['sma_profit'] = (val_df['sma_sold'] * val_df['sell_price']) - (val_df['sma_sold'] * val_df['cost_price']) - val_df['sma_waste_cost']
        val_df['net_profit'] = val_df['revenue'] - val_df['cogs'] - val_df['waste_cost']

        self.simulation = val_df

    def generate_dashboard(self):
        print("[5/6] Generating advanced dashboard...")
        sim = self.simulation
        
        fig = plt.figure(figsize=(14, 8))
        gs = fig.add_gridspec(2, 2)
        
        # 1. profit comparison
        ax1 = fig.add_subplot(gs[0, 0])
        profits = [sim['sma_profit'].sum(), sim['net_profit'].sum(), sim['oracle_profit'].sum()]
        labels = ['Human Strategy\n(Moving Avg)', 'AI Strategy\n(LightGBM)', 'Theoretical Max\n(Perfect)']
        colors = ['#7f8c8d', '#2980b9', '#27ae60']
        bars = ax1.bar(labels, profits, color=colors)
        ax1.set_title("Net Profit Comparison (60-Day Test)")
        ax1.set_ylabel("Profit ($)")
        ax1.bar_label(bars, fmt='$%.0f', padding=3)

        # 2. waste comparison
        ax2 = fig.add_subplot(gs[0, 1])
        waste = [sim['sma_waste_cost'].sum(), sim['waste_cost'].sum()]
        ax2.bar(['Human Strategy', 'AI Strategy'], waste, color=['#c0392b', '#2980b9'])
        ax2.set_title("Total Waste Cost Comparison")
        ax2.set_ylabel("Capital Lost to Spoilage ($)")

        # 3. prediction error distribution
        ax3 = fig.add_subplot(gs[1, 0])
        errors = sim['pred_demand'] - sim['sales']
        sns.histplot(errors, kde=True, ax=ax3, color='#8e44ad', bins=40)
        ax3.set_xlim(-10, 10)
        ax3.set_title("Prediction Error Distribution")
        ax3.set_xlabel("Error (Predicted units - Actual units)")

        # 4. daily profit tracking
        ax4 = fig.add_subplot(gs[1, 1])
        daily_profit = sim.groupby('date')[['net_profit', 'sma_profit']].sum().reset_index()
        sns.lineplot(data=daily_profit, x='date', y='net_profit', label='AI Profit', ax=ax4, color='#2980b9')
        sns.lineplot(data=daily_profit, x='date', y='sma_profit', label='Human Profit', ax=ax4, color='#7f8c8d')
        ax4.set_title("Daily Profit Trajectory")
        ax4.set_xlabel("Date")
        ax4.set_ylabel("Daily Profit ($)")
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig("smart_supply_dashboard.png")
        print("   Dashboard saved to 'smart_supply_dashboard.png'")

    def export_results(self):
        print("[6/6] Exporting forecasts...")
        cols = ['date', 'item_id', 'sales', 'pred_demand', 'stock_decision', 
                'waste', 'net_profit', 'sma_profit']
        self.simulation[cols].to_csv("m5_supply_chain_forecast.csv", index=False)
        print("   CSV saved to 'm5_supply_chain_forecast.csv'")

# ==========================================
# main execution
# ==========================================
if __name__ == "__main__":
    
    # initialize the application with the three kaggle files
    app = SmartSupplyAI(
        sales_path="sales_train_validation.csv",
        calendar_path="calendar.csv",
        prices_path="sell_prices.csv"
    )
    
    app.load_and_clean()
    app.feature_engineering()
    app.train_model()
    app.run_simulation()
    
    sim = app.simulation
    ai_profit = sim['net_profit'].sum()
    human_profit = sim['sma_profit'].sum()
    oracle_profit = sim['oracle_profit'].sum()
    waste_savings = sim['sma_waste_cost'].sum() - sim['waste_cost'].sum()
    
    print("\nPROJECTED IMPACT REPORT (60-DAY VALIDATION)")
    print("-" * 45)
    print(f"Human Baseline Profit:  ${human_profit:,.2f}")
    print(f"AI Optimized Profit:    ${ai_profit:,.2f}")
    print(f"Net Profit Increase:    ${ai_profit - human_profit:,.2f}")
    print(f"Waste Capital Saved:    ${waste_savings:,.2f}")
    print(f"Theoretical Max Profit: ${oracle_profit:,.2f}")
    print("-" * 45)
    
    app.generate_dashboard()
    app.export_results()