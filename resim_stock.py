"""
Re-apply the current stocking rule to an existing holdout CSV.

Uses the saved point/q75/q90 forecasts; no retrain. Rewrites financial
columns in place so the web app picks up the new policy immediately.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from grocery import (
    DEPT_COGS_RATIO,
    DEPT_SPOILAGE_RATE,
    STOCKOUT_PENALTY_FRACTION,
    SmartSupplyAI,
)

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "m5_supply_chain_forecast.csv"


def _recover_sell_price(sim: pd.DataFrame) -> pd.Series:
    # oracle_profit = sales * sell_price * (1 - cogs_ratio)
    dept = sim["dept_id"].astype(str)
    margin_r = 1.0 - dept.map(DEPT_COGS_RATIO).fillna(0.60)
    sell = np.where(
        (sim["sales"] > 0) & (margin_r > 0),
        sim["oracle_profit"] / (sim["sales"] * margin_r),
        np.nan,
    )
    out = pd.Series(sell, index=sim.index, dtype=float)
    out = out.groupby(sim["item_id"]).transform(lambda s: s.ffill().bfill())
    for d, idx in sim.groupby("dept_id").groups.items():
        med = out.loc[idx].median()
        out.loc[idx] = out.loc[idx].fillna(med)
    return out


def main() -> None:
    if not CSV_PATH.exists():
        raise SystemExit(f"Missing {CSV_PATH.name}. Run grocery.py first.")

    sim = pd.read_csv(CSV_PATH, parse_dates=["date"])
    app = SmartSupplyAI("a.csv", "b.csv", "c.csv")
    stock = app._dept_aware_stock(sim)

    sell_price = _recover_sell_price(sim)
    dept = sim["dept_id"].astype(str)
    cogs_r = dept.map(DEPT_COGS_RATIO).fillna(0.60)
    spoil = dept.map(DEPT_SPOILAGE_RATE).fillna(0.10)
    cost_price = sell_price * cogs_r
    margin = sell_price - cost_price

    sales = sim["sales"].to_numpy(dtype=float)
    units_sold = np.minimum(stock, sales)
    unsold = np.maximum(0.0, stock - sales)
    missed = np.maximum(0.0, sales - stock)
    waste = unsold * spoil.to_numpy(dtype=float)

    waste_cost = waste * cost_price.to_numpy(dtype=float)
    stockout_cost = missed * margin.to_numpy(dtype=float) * STOCKOUT_PENALTY_FRACTION
    revenue = units_sold * sell_price.to_numpy(dtype=float)
    cogs = units_sold * cost_price.to_numpy(dtype=float)
    net_profit = revenue - cogs - waste_cost - stockout_cost

    before_lift = float(sim["net_profit"].sum() - sim["sma_profit"].sum())
    before_waste = float(sim["waste_cost"].sum())

    sim["stock_decision"] = stock
    sim["waste"] = waste
    sim["missed_sales"] = missed
    sim["waste_cost"] = waste_cost
    sim["stockout_cost"] = stockout_cost
    sim["net_profit"] = net_profit

    sim.to_csv(CSV_PATH, index=False)

    after_lift = float(sim["net_profit"].sum() - sim["sma_profit"].sum())
    after_waste = float(sim["waste_cost"].sum())
    base_waste = float(sim["human_waste_cost"].sum())
    print(f"Wrote {CSV_PATH}")
    print(f"Profit lift: ${before_lift:,.0f} -> ${after_lift:,.0f}")
    print(f"ML spoilage: ${before_waste:,.0f} -> ${after_waste:,.0f} (baseline ${base_waste:,.0f})")
    print(
        f"Spoilage avoided vs baseline: ${base_waste - after_waste:,.0f} "
        f"(was ${base_waste - before_waste:,.0f})"
    )


if __name__ == "__main__":
    main()
