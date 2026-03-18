import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="smartsupply", layout="wide")

@st.cache_data
def load_simulation():
    sim = pd.read_csv("m5_supply_chain_forecast.csv", parse_dates=["date"])
    return sim

@st.cache_resource
def load_model():
    try:
        return joblib.load("lightgbm_model.pkl")
    except FileNotFoundError:
        return None

st.title("smartsupply – ml-optimized grocery ordering")

st.markdown(
    "this dashboard compares a tuned lightgbm model against a realistic human baseline "
    "(ewma + seasonal naive) on the public walmart m5 dataset. "
    "costs vary by department and include spoilage *and* lost-sale penalties."
)

with st.expander("about the walmart m5 data", expanded=False):
    st.markdown(
        "- **sales_train_validation.csv**: historical daily unit sales for thousands of walmart items, "
        "broken down by store, category, and item id.\n"
        "- **calendar.csv**: the retail calendar with real dates, events, and snap (food-stamp) indicators.\n"
        "- **sell_prices.csv**: weekly store-level prices so the model can learn how price affects demand.\n\n"
        "smartsupply filters this down to a single store and the foods category, learns demand patterns, and then "
        "simulates profit and waste when the ml model controls ordering instead of a human strategy."
    )
    try:
        st.image(
            "smart_supply_dashboard.png",
            caption="offline dashboard generated from the same m5 simulation.",
            use_column_width=True,
        )
    except Exception:
        pass

if "sim" not in st.session_state:
    try:
        st.session_state["sim"] = load_simulation()
    except FileNotFoundError:
        st.error("could not find m5_supply_chain_forecast.csv. run `python3 grocery.py` first.")
        st.stop()

sim = st.session_state["sim"]

# top-level KPIs
ml_profit = sim["net_profit"].sum()
human_profit = sim["sma_profit"].sum()
profit_lift = ml_profit - human_profit
oracle_profit = sim["oracle_profit"].sum()

total_item_days = len(sim)
ml_stockout_rate = (sim["missed_sales"] > 0).mean() * 100
ml_waste_total = sim["waste_cost"].sum()
human_waste_total = sim.get("human_waste_cost", sim["waste_cost"]).sum() if "human_waste_cost" in sim.columns else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("human profit (60d)", f"${human_profit:,.0f}")
c2.metric("ml model profit (60d)", f"${ml_profit:,.0f}")
c3.metric("profit lift", f"${profit_lift:,.0f}")
c4.metric("ml stockout rate", f"{ml_stockout_rate:.1f}%")

st.markdown("---")

# daily profit chart with prediction interval
st.subheader("daily profit trajectory")
daily = sim.groupby("date").agg(
    ml_profit=("net_profit", "sum"),
    human_profit=("sma_profit", "sum"),
).reset_index().set_index("date")
st.line_chart(daily)

st.markdown("---")

# prediction interval chart
st.subheader("demand forecast with confidence intervals")
st.markdown("aggregated daily demand vs. point prediction and q75/q90 upper bounds.")
daily_demand = sim.groupby("date").agg(
    actual=("sales", "sum"),
    predicted=("pred_demand", "sum"),
    q75=("pred_q75", "sum"),
    q90=("pred_q90", "sum"),
).reset_index().set_index("date")
st.line_chart(daily_demand)

st.markdown("---")

# per-department breakdown
st.subheader("per-department breakdown")
if "dept_id" in sim.columns:
    dept = sim.groupby("dept_id").agg(
        ml_profit=("net_profit", "sum"),
        human_profit=("sma_profit", "sum"),
        waste_cost=("waste_cost", "sum"),
        stockout_cost=("stockout_cost", "sum"),
        missed_pct=("missed_sales", lambda x: (x > 0).mean() * 100),
    ).reset_index()
    dept.columns = ["department", "ml profit ($)", "human profit ($)", "waste cost ($)", "stockout cost ($)", "stockout %"]
    dept["profit lift ($)"] = dept["ml profit ($)"] - dept["human profit ($)"]
    st.dataframe(
        dept.style.format({
            "ml profit ($)": "${:,.0f}",
            "human profit ($)": "${:,.0f}",
            "waste cost ($)": "${:,.0f}",
            "stockout cost ($)": "${:,.0f}",
            "stockout %": "{:.1f}%",
            "profit lift ($)": "${:,.0f}",
        }),
        use_container_width=True,
    )
else:
    st.info("department column not found in the exported csv.")

st.markdown("---")

# feature importance
st.subheader("feature importance (lightgbm gain)")
model = load_model()
if model is not None:
    importance = model.feature_importance(importance_type="gain")
    feat_names = model.feature_name()
    feat_df = pd.DataFrame({"feature": feat_names, "gain": importance})
    feat_df = feat_df.nlargest(15, "gain").sort_values("gain")
    st.bar_chart(feat_df.set_index("feature")["gain"])
else:
    st.info("model file not found. run `python3 grocery.py` to generate lightgbm_model.pkl.")

st.markdown("---")

# item-level table with filters
st.subheader("item-level recommendations")

item_filter = st.text_input("filter by item_id (optional):").strip()
date_range = st.date_input(
    "date range",
    value=(daily.index.min().date(), daily.index.max().date()),
)

start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
filtered = sim[(sim["date"] >= start_date) & (sim["date"] <= end_date)]

if item_filter:
    filtered = filtered[filtered["item_id"].str.contains(item_filter, case=False)]

cols_to_show = [
    "date", "item_id", "dept_id", "sales",
    "pred_demand", "pred_q75", "pred_q90",
    "stock_decision", "waste", "missed_sales",
    "net_profit", "sma_profit",
]
cols_to_show = [c for c in cols_to_show if c in filtered.columns]

st.dataframe(
    filtered[cols_to_show]
        .sort_values(["date", "item_id"])
        .head(500),
    use_container_width=True,
)

st.markdown(
    """
**table legend**
- `sales`: actual units sold that date.
- `pred_demand`: lightgbm point forecast.
- `pred_q75` / `pred_q90`: upper quantile forecasts (75th and 90th percentile). used for confidence-based safety stock.
- `stock_decision`: units ordered — set to the q90 upper bound so the model orders more when uncertain.
- `waste`: units spoiled (dept-specific daily spoilage rate applied to unsold inventory).
- `missed_sales`: units of demand that could not be fulfilled (stockouts).
- `net_profit`: ml strategy profit after revenue, cogs, waste, and stockout penalty.
- `sma_profit`: human strategy profit using ewma + seasonal naive ordering.
"""
)
