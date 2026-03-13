# Smart Supply

An AI-driven inventory optimization engine designed to minimize perishable food waste and maximize net profit using time-series demand forecasting and machine learning. 

This project uses the real-world Walmart M5 Forecasting dataset to simulate realistic retail environments, balancing the cost of stockouts against the cost of inventory spoilage.

## Features

- **Advanced Time-Series Forecasting:** Uses LightGBM (with a Poisson objective) to predict daily product demand based on historical sales data.
- **Feature Engineering:** Implements lag features, rolling statistics (7-day moving averages), and price momentum indicators to capture market trends and seasonality.
- **Dynamic Safety Stock:** Automatically adjusts buffer stock levels based on product profit margins (higher margin = higher safety factor).
- **Realistic Business Simulation:** Calculates true net profit by incorporating a realistic 10% daily spoilage rate and probability-based ordering to prevent overstocking slow-moving items.
- **Strategic Baselines:** Compares AI performance against a Human Strategy (7-day moving average) and an Oracle Baseline (theoretical maximum profit with perfect prediction).
- **Comprehensive Analytics:** Generates a 4-panel analytical dashboard visualizing profit trajectories, capital lost to waste, and prediction error distributions.
- **Data Export:** Outputs full daily forecast decisions and financial metrics to a CSV file.

## Setup

1. Clone the repository:

```bash
git clone https://github.com/adnanjasims/smart-supply.git
cd smart-supply
```

2. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies (if you are not using `requirements.txt`):

```bash
python3 -m pip install --upgrade pip
python3 -m pip install pandas numpy lightgbm matplotlib seaborn scikit-learn joblib streamlit
```

If you prefer, you can instead install from a `requirements.txt` file:

```bash
pip install -r requirements.txt
```

4. Download the M5 Dataset:

Download `sales_train_validation.csv`, `calendar.csv`, and `sell_prices.csv` from the Kaggle M5 Forecasting Competition and place them in your project root directory.

## Usage (Core Pipeline)

Run the main pipeline to process data, train the model, and simulate profits:

```bash
python3 grocery.py
```

## Output

The script generates two key files:
- `smart_supply_dashboard.png`: Visualizes net profit comparisons, waste reduction, prediction errors, and daily profit trajectories.
- `m5_supply_chain_forecast.csv`: A detailed report containing daily predicted demand, automated stock decisions, simulated waste, and net profit per item.

## Web Frontend (Streamlit Dashboard)

After you have run `python3 grocery.py` at least once and `m5_supply_chain_forecast.csv` has been generated, you can launch the interactive browser dashboard:

```bash
python3 -m streamlit run app.py
```

The Streamlit app will:
- read `m5_supply_chain_forecast.csv` and compare AI-driven strategy vs a simple human baseline,
- show key KPIs (human profit, AI/ML model profit, profit lift),
- plot daily profit trajectories for both strategies, and
- expose an item-level table of recommendations with filters and a legend explaining each column.

Open the URL printed by Streamlit (usually `http://localhost:8501`) in your browser to explore the results.
