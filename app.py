import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="smartsupply", layout="wide")

@st.cache_data
def load_simulation():
    # load the csv exported by grocery.py
    sim = pd.read_csv("m5_supply_chain_forecast.csv", parse_dates=["date"])
    return sim

st.title("smartsupply – ml vs human inventory strategy")

# high-level context on what this app is doing
st.markdown(
    "this dashboard uses your trained lightgbm model on the public walmart m5 dataset "
    "(daily item-level sales, calendar, and price data) to compare the ml model's ordering policy "
    "against a simple human baseline."
)

with st.expander("about the walmart m5 data", expanded=False):
    st.markdown(
        "- **sales_train_validation.csv**: historical daily unit sales for thousands of walmart items, "
        "broken down by store, category, and item id.\n"
        "- **calendar.csv**: the retail calendar with real dates, events, and snap (food-stamp) indicators.\n"
        "- **sell_prices.csv**: weekly store-level prices so the model can learn how price affects demand.\n\n"
        "smartsupply filters this down to a single store and category, learns demand patterns, and then "
        "simulates profit and waste when the ml model controls ordering instead of a simple moving-average human strategy."
    )
    # optional: include the offline matplotlib dashboard image without breaking the dark layout
    st.image(
        "smart_supply_dashboard.png",
        caption="offline dashboard generated from the same m5 simulation (profit, waste, and error views).",
        use_column_width=True,
    )

if "sim" not in st.session_state:
    try:
        st.session_state["sim"] = load_simulation()
    except FileNotFoundError:
        st.error("could not find m5_supply_chain_forecast.csv. run `python3 grocery.py` first.")
        st.stop()

sim = st.session_state["sim"]

# top kpis
ml_profit = sim["net_profit"].sum()
human_profit = sim["sma_profit"].sum()
profit_lift = ml_profit - human_profit

c1, c2, c3 = st.columns(3)
c1.metric("human profit (60d)", f"${human_profit:,.0f}")
c2.metric("ml model profit (60d)", f"${ml_profit:,.0f}")
c3.metric("profit lift (ml - human)", f"${profit_lift:,.0f}")

st.markdown("---")

# daily profit chart
st.subheader("daily profit trajectory")
daily = sim.groupby("date")[["net_profit", "sma_profit"]].sum().reset_index()
daily = daily.set_index("date")
st.line_chart(daily.rename(columns={"net_profit": "ml_model_profit", "sma_profit": "human_profit"}))

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
    "date",
    "item_id",
    "sales",
    "pred_demand",
    "stock_decision",
    "waste",
    "net_profit",
    "sma_profit",
]
st.dataframe(
    filtered[cols_to_show]
        .sort_values(["date", "item_id"])
        .head(500),
    use_container_width=True,
)

st.markdown(
    """
**table legend**
- `date`: the calendar date of the transaction in the m5 dataset.
- `item_id`: unique walmart item identifier within the selected store/category slice.
- `sales`: actual units sold on that date (what really happened in history).
- `pred_demand`: units of demand forecasted by the lightgbm model for that date.
- `stock_decision`: how many units smartsupply chooses to put on the shelf after applying a safety factor.
- `waste`: units that ended up unsold and spoiled (based on the configured spoilage rate).
- `net_profit`: profit in dollars under the ml model strategy for that item and date after revenue, cogs, and waste.
- `sma_profit`: profit in dollars if a human had ordered using a simple 7-day moving-average rule instead.
"""
)