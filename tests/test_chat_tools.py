"""Chat tool smoke tests. No API key needed."""

from __future__ import annotations

import pandas as pd
import pytest

from chat.tools import (
    ChatScope,
    execute_tool,
    tool_dept_breakdown,
    tool_item_lookup,
    tool_run_summary,
    tool_worst_items,
)


@pytest.fixture
def sim() -> pd.DataFrame:
    rows = []
    for day, date in enumerate(pd.date_range("2016-03-01", periods=5, freq="D")):
        for item, dept in [
            ("FOODS_1_001", "FOODS_1"),
            ("FOODS_2_010", "FOODS_2"),
            ("FOODS_3_100", "FOODS_3"),
        ]:
            rows.append(
                {
                    "date": date,
                    "item_id": item,
                    "dept_id": dept,
                    "sales": 2 + day,
                    "pred_demand": 2.0,
                    "pred_q75": 3.0,
                    "pred_q90": 4.0,
                    "stock_decision": 3.0,
                    "waste": 0.5 if dept == "FOODS_1" else 0.1,
                    "missed_sales": 1.0 if dept == "FOODS_2" else 0.0,
                    "waste_cost": 1.0 if dept == "FOODS_1" else 0.2,
                    "stockout_cost": 2.0 if dept == "FOODS_2" else 0.0,
                    "human_waste_cost": 2.0,
                    "human_stockout_cost": 1.0,
                    "human_missed": 0.5,
                    "net_profit": 10.0 if dept == "FOODS_3" else 5.0,
                    "sma_profit": 4.0,
                    "oracle_profit": 12.0,
                }
            )
    return pd.DataFrame(rows)


def test_run_summary_totals(sim):
    out = tool_run_summary(sim, ChatScope())
    assert out["ok"] is True
    assert out["items"] == 3
    assert out["profit_lift"] == pytest.approx(out["ml_profit"] - out["human_profit"])


def test_dept_filter(sim):
    out = tool_run_summary(sim, ChatScope(dept_id="FOODS_1"))
    assert out["ok"] is True
    assert out["items"] == 1


def test_dept_breakdown(sim):
    out = tool_dept_breakdown(sim, ChatScope())
    assert out["ok"] is True
    assert len(out["departments"]) == 3


def test_worst_items_by_waste(sim):
    out = tool_worst_items(sim, ChatScope(), metric="waste_cost", limit=2)
    assert out["ok"] is True
    assert out["items"][0]["item_id"] == "FOODS_1_001"


def test_item_lookup(sim):
    out = tool_item_lookup(sim, ChatScope(), "FOODS_3_100")
    assert out["ok"] is True
    assert out["dept_id"] == "FOODS_3"
    assert len(out["recent_days"]) == 5


def test_execute_unknown_tool(sim):
    raw = execute_tool("nope", {}, sim, ChatScope())
    assert "Unknown tool" in raw
