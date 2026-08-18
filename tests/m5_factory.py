"""Build tiny M5-shaped CSVs for offline tests."""

from __future__ import annotations

import numpy as np
import pandas as pd


def write_mini_m5(tmp_path, n_days=90, n_items=3, store_id="CA_1", missing_price_days=0):
    """Create small sales/calendar/prices files that mirror M5 schema."""
    rng = np.random.default_rng(42)
    start = pd.Timestamp("2015-01-01")
    dates = pd.date_range(start, periods=n_days, freq="D")

    cal_rows = []
    for i, dt in enumerate(dates, start=1):
        cal_rows.append({
            "date": dt,
            "wm_yr_wk": int(dt.strftime("%y%W")) + 11000,
            "weekday": dt.day_name(),
            "wday": dt.weekday() + 1,
            "month": dt.month,
            "year": dt.year,
            "d": f"d_{i}",
            "event_name_1": "SuperBowl" if i == 40 else np.nan,
            "event_type_1": "Sporting" if i == 40 else np.nan,
            "event_name_2": np.nan,
            "event_type_2": np.nan,
            "snap_CA": int(dt.day <= 10),
            "snap_TX": 0,
            "snap_WI": 0,
        })
    calendar = pd.DataFrame(cal_rows)
    calendar_path = tmp_path / "calendar.csv"
    calendar.to_csv(calendar_path, index=False)

    depts = ["FOODS_1", "FOODS_2", "FOODS_3"]
    sales_rows = []
    price_rows = []
    for j in range(n_items):
        item_id = f"FOODS_item_{j}"
        dept = depts[j % 3]
        series = {
            "id": f"{item_id}_{store_id}_validation",
            "item_id": item_id,
            "dept_id": dept,
            "cat_id": "FOODS",
            "store_id": store_id,
            "state_id": "CA",
        }
        for i, dt in enumerate(dates, start=1):
            sales = max(0, int(3 + 2 * np.sin(2 * np.pi * dt.dayofweek / 7) + rng.integers(0, 3)))
            series[f"d_{i}"] = sales
            if not (j == 0 and i <= missing_price_days):
                price_rows.append({
                    "store_id": store_id,
                    "item_id": item_id,
                    "wm_yr_wk": int(dt.strftime("%y%W")) + 11000,
                    "sell_price": round(2.0 + 0.5 * j + (0.2 if i > n_days // 2 else 0.0), 2),
                })
        sales_rows.append(series)

    sales = pd.DataFrame(sales_rows)
    prices = pd.DataFrame(price_rows).drop_duplicates(
        subset=["store_id", "item_id", "wm_yr_wk"], keep="last"
    )

    sales_path = tmp_path / "sales_train_validation.csv"
    prices_path = tmp_path / "sell_prices.csv"
    sales.to_csv(sales_path, index=False)
    prices.to_csv(prices_path, index=False)

    return {
        "sales": sales_path,
        "calendar": calendar_path,
        "prices": prices_path,
        "n_days": n_days,
        "n_items": n_items,
        "store_id": store_id,
    }
