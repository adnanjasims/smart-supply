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

@st.cache_resource
def load_bundle():
    try:
        return joblib.load("lightgbm_bundle.pkl")
    except FileNotFoundError:
        return None


def render_dashboard(sim: pd.DataFrame) -> None:
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
                use_container_width=True,
            )
        except Exception:
            st.caption("run `python3 grocery.py` to generate smart_supply_dashboard.png.")

    ml_profit = sim["net_profit"].sum()
    human_profit = sim["sma_profit"].sum()
    profit_lift = ml_profit - human_profit
    oracle_profit = sim["oracle_profit"].sum()

    ml_stockout_rate = (sim["missed_sales"] > 0).mean() * 100
    ml_waste_total = sim["waste_cost"].sum()
    if "human_waste_cost" in sim.columns:
        human_waste_total = sim["human_waste_cost"].sum()
    else:
        human_waste_total = None

    c1, c2, c3, c4 = st.columns(4)
c1.metric("baseline profit (60d)", f"${human_profit:,.0f}")
c2.metric("ml model profit (60d)", f"${ml_profit:,.0f}")
c3.metric("profit lift", f"${profit_lift:,.0f}")
c4.metric("ml stockout rate", f"{ml_stockout_rate:.1f}%")

c5, c6, c7, c8 = st.columns(4)
c5.metric("ml spoilage cost", f"${ml_waste_total:,.0f}")
if human_waste_total is not None:
    c6.metric("baseline spoilage cost", f"${human_waste_total:,.0f}")
    c7.metric("spoilage avoided vs baseline", f"${human_waste_total - ml_waste_total:,.0f}")
else:
    c6.metric("baseline spoilage cost", "n/a")
    c7.metric("spoilage avoided vs baseline", "n/a")
c8.metric("perfect-foresight profit", f"${oracle_profit:,.0f}")

    st.markdown("---")

    st.subheader("daily profit trajectory")
    daily = sim.groupby("date").agg(
        ml_profit=("net_profit", "sum"),
        human_profit=("sma_profit", "sum"),
    ).reset_index().set_index("date")
    st.line_chart(daily)

    st.markdown("---")

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

    st.subheader("feature importance (lightgbm gain)")
    bundle = load_bundle()
    model = bundle["point"] if bundle is not None else load_model()
    if model is not None:
        importance = model.feature_importance(importance_type="gain")
        feat_names = model.feature_name()
        feat_df = pd.DataFrame({"feature": feat_names, "gain": importance})
        feat_df = feat_df.nlargest(15, "gain").sort_values("gain")
        st.bar_chart(feat_df.set_index("feature")["gain"])
        if bundle is not None and bundle.get("cv_scores"):
            scores = bundle["cv_scores"]
            st.caption(
                f"date-based cv mae: {float(np.mean(scores)):.3f} +/- {float(np.std(scores)):.3f} "
                f"across {len(scores)} folds"
            )
    else:
        st.info("model file not found. run `python3 grocery.py` to generate lightgbm_model.pkl.")

    st.markdown("---")

    st.subheader("item-level recommendations")

    item_filter = st.text_input("filter by item_id (optional):").strip()
    date_range = st.date_input(
        "date range",
        value=(daily.index.min().date(), daily.index.max().date()),
    )

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    else:
        start_date = end_date = pd.to_datetime(date_range)

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
- `pred_q75` / `pred_q90`: upper quantile forecasts (75th and 90th percentile). used for dept-aware safety stock.
- `stock_decision`: units ordered. perishables (FOODS_1) use q75 to cut waste; other depts use q90.
- `waste`: units spoiled (dept-specific daily spoilage rate applied to unsold inventory).
- `missed_sales`: units of demand that could not be fulfilled (stockouts).
- `net_profit`: ml strategy profit after revenue, cogs, waste, and stockout penalty.
- `sma_profit`: human strategy profit using ewma + seasonal naive ordering.
"""
    )


def render_chat(sim: pd.DataFrame) -> None:
    from chat import answer_question, chat_config_present
    from chat.gemini import SUGGESTIONS

    st.markdown(
        "type a question about the holdout run (profit lift, waste, stockouts, "
        "depts, or an `item_id`). replies pull from the same csv as the charts."
    )

    if not chat_config_present():
        st.warning(
            "set `GEMINI_API_KEY` in `.env` to turn this on "
            "(see `.env.example`). charts work either way."
        )
        return

    depts = ["(all)"] + sorted(sim["dept_id"].astype(str).unique().tolist())
    c1, c2 = st.columns([1, 2])
    with c1:
        dept = st.selectbox("scope department", depts, index=0, key="chat_dept")
    with c2:
        st.caption("filters the queries. (all) = whole store.")

    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []

    for msg in st.session_state["chat_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if not st.session_state["chat_messages"]:
        st.caption("try one of these:")
        cols = st.columns(len(SUGGESTIONS))
        for i, suggestion in enumerate(SUGGESTIONS):
            if cols[i].button(suggestion, key=f"suggest_{i}"):
                st.session_state["chat_pending"] = suggestion
                st.rerun()

    prompt = st.chat_input("e.g. which dept saved the most on waste?")
    if "chat_pending" in st.session_state:
        prompt = st.session_state.pop("chat_pending")

    if not prompt:
        return

    st.session_state["chat_messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("running query..."):
            result = answer_question(
                st.session_state["chat_messages"],
                sim,
                dept_id=dept,
            )
        if result.get("ok"):
            reply = result["reply"]
            st.markdown(reply)
            tools = result.get("tools_used") or []
            model = result.get("model") or ""
            if tools or model:
                st.caption(f"{model} | tools: {', '.join(tools)}")
        else:
            reply = f"could not answer: {result.get('error', 'unknown error')}"
            st.error(reply)

    st.session_state["chat_messages"].append({"role": "assistant", "content": reply})


st.title("smartsupply - ml-optimized grocery ordering")

if "sim" not in st.session_state:
    try:
        st.session_state["sim"] = load_simulation()
    except FileNotFoundError:
        st.error("could not find m5_supply_chain_forecast.csv. run `python3 grocery.py` first.")
        st.stop()

sim = st.session_state["sim"]

tab_dash, tab_chat = st.tabs(["dashboard", "ask the data"])
with tab_dash:
    render_dashboard(sim)
with tab_chat:
    render_chat(sim)
