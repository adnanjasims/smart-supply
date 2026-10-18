# Smart Supply

Grocery inventory optimization on real Walmart M5 data. The aim is to forecast daily demand with LightGBM, decide how much to stock, then score that decision on profit after spoilage and stockouts.

The model has to beat a realistic human baseline on money

## What it does

- Trains a LightGBM demand model (Poisson objective) on one store + FOODS category from M5
- Builds lag, rolling, price, SNAP, holiday/event, and Fourier seasonality features
- Tunes hyperparameters with Optuna when it is installed
- Trains q75 / q90 quantile models for uncertainty
- Uses dept-aware safety stock: perishables (FOODS_1) order closer to q75 to cut waste; other depts use q90
- Compares ML ordering vs an EWMA + seasonal-naive human strategy and a perfect-foresight oracle
- Writes a CSV of daily decisions plus a 6-panel dashboard image
- Serves a FastAPI + React app (optional Gemini chat over the holdout CSV); Streamlit `app.py` still works
- MLOps: YAML config, run tracking, data hashes, quality gates, and a local model registry

Spoilage and COGS are set by department (produce vs dairy vs packaged)

## Setup

```bash
git clone https://github.com/adnanjasims/smart-supply.git
cd smart-supply

python3 -m venv .venv
source .venv/bin/activate

python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

Grab these from the [Kaggle M5 Forecasting competition](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data) and drop them in the project root:

- `sales_train_validation.csv`
- `calendar.csv`
- `sell_prices.csv`



## Run the pipeline

```bash
python3 grocery.py
```

Useful flags if you want to change the scope:

```bash
python3 grocery.py --store CA_1 --cat FOODS --lookback-days 365 --holdout-days 60
```

If a required file is missing or a column is wrong, the script exits with a clear error instead of failing halfway through a melt.

Stocking uses a per-dept newsvendor critical fractile over the point / q75 / q90 ladder (perishables sit lower; shelf-stable sits higher). To re-score an existing holdout CSV after changing that rule, without retraining:

```bash
python3 resim_stock.py
```

## Outputs

- `lightgbm_model.pkl` - point forecast model
- `lightgbm_model_q75.pkl` / `lightgbm_model_q90.pkl` - quantile models
- `lightgbm_bundle.pkl` - all three models plus feature list and CV scores
- `smart_supply_dashboard.png` - profit, waste, stockouts, error hist, daily P&L, feature importance
- `m5_supply_chain_forecast.csv` - per item/day forecasts, stock decisions, and both ML and human financials



## Web app (FastAPI + React)

After at least one successful `grocery.py` run:

```bash
# terminal 1: API
pip install -r requirements.txt
uvicorn server.main:app --reload --port 8000

# terminal 2: UI
cd web
npm install
npm run dev
```

Open `http://localhost:5173`. Vite proxies `/api` to the FastAPI process. You get KPIs, profit/demand charts, dept breakdown, feature importance, item filters, and an **Ask the data** chat panel.

```bash
cp .env.example .env
# paste a key from https://aistudio.google.com/apikey into GEMINI_API_KEY
```

If no key is inputted, chat stays off but the rest of the app still works.

### Streamlit (optional)

The older dashboard is still there if needed:

```bash
python3 -m streamlit run app.py
```

## How the money math works

For each item-day:

1. Predict demand (point + quantiles)
2. Choose a stock level (dept-aware quantile rule above)
3. Sell `min(stock, actual demand)`
4. Unsold units spoil at the dept rate and eat COGS
5. Missed demand pays a stockout penalty as a fraction of margin

Human baseline uses `0.6 * EWMA(7) + 0.4 * seasonal_naive`, then a 1.1x buffer. Uses same cost model so comparison is fair.

## Notes

- Cross-validation is an expanding window by calendar date, not a row-index `TimeSeriesSplit`. That matters on panel data where many SKUs share the same day.
- The first ~28 days per item are dropped on purpose after feature engineering. Those rows cannot fill a 28-day lag / rolling window yet. On CA_1 FOODS that is about 40k rows of warmup, plus a tiny slice of weeks with no sell_price in M5. It is not silent data loss.
- If model prediction blows up at simulate time, the run falls back to the human EWMA rule and flags it in the report.
- Optuna is optional. Without it (or with `--skip-tuning`), training still runs with solid default LightGBM params.



## MLOps

Tracked training lives under `artifacts/`. Config is `config/default.yaml`.

```bash
#train with tracking + quality gates
python -m mlops train

#same, then promote to production if gates pass
python -m mlops train --promote

#inspect runs
python -m mlops list
python -m mlops show <run_id>
python -m mlops current
python -m mlops promote <run_id>
```

Each run stores params, metrics, data file hashes, model bundle, forecast CSV, and dashboard under `artifacts/runs/<run_id>/`. Gates in the config can reject a run (e.g. MAE too high or fallback used). `promote` copies a good run into `artifacts/registry/production/`.

## Tests

```bash
pip install -r requirements.txt
python3 -m pytest -q
```

Tests build a tiny synthetic M5-shaped dataset, so they do not need the full Kaggle download.