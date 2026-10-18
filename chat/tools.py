"""
Pandas helpers the chat layer can call.

Gemini only gets these aggregates back, not the full frame. Optional
dept / date filters live on ChatScope.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pandas as pd

DEPT_LABELS = {
    "FOODS_1": "produce / perishables",
    "FOODS_2": "dairy / deli",
    "FOODS_3": "packaged / shelf-stable",
}

MAX_ROWS = 20


@dataclass
class ChatScope:
    dept_id: str = "(all)"
    start_date: str | None = None
    end_date: str | None = None


def _json(obj: Any) -> str:
    return json.dumps(obj, default=str)


def _apply_scope(sim: pd.DataFrame, scope: ChatScope) -> pd.DataFrame:
    out = sim
    if scope.dept_id and scope.dept_id != "(all)":
        out = out[out["dept_id"].astype(str) == scope.dept_id]
    if scope.start_date:
        out = out[out["date"] >= pd.to_datetime(scope.start_date)]
    if scope.end_date:
        out = out[out["date"] <= pd.to_datetime(scope.end_date)]
    return out


def _money(x: float) -> float:
    return round(float(x), 2)


def _pct(x: float) -> float:
    return round(float(x) * 100.0, 2)


def tool_run_summary(sim: pd.DataFrame, scope: ChatScope) -> dict[str, Any]:
    """Roll up holdout profit / waste / stockouts for the active scope."""
    df = _apply_scope(sim, scope)
    if df.empty:
        return {"ok": False, "error": "No rows match the current filters."}

    ml_profit = df["net_profit"].sum()
    human_profit = df["sma_profit"].sum()
    oracle = df["oracle_profit"].sum() if "oracle_profit" in df.columns else None

    out: dict[str, Any] = {
        "ok": True,
        "rows": int(len(df)),
        "items": int(df["item_id"].nunique()),
        "date_start": str(df["date"].min().date()),
        "date_end": str(df["date"].max().date()),
        "ml_profit": _money(ml_profit),
        "human_profit": _money(human_profit),
        "profit_lift": _money(ml_profit - human_profit),
        "ml_waste_cost": _money(df["waste_cost"].sum()),
        "ml_stockout_cost": _money(df["stockout_cost"].sum()),
        "ml_stockout_rate_pct": _pct((df["missed_sales"] > 0).mean()),
        "ml_units_wasted": _money(df["waste"].sum()),
        "ml_units_missed": _money(df["missed_sales"].sum()),
    }
    if "human_waste_cost" in df.columns:
        out["human_waste_cost"] = _money(df["human_waste_cost"].sum())
        out["waste_saved"] = _money(df["human_waste_cost"].sum() - df["waste_cost"].sum())
    if "human_missed" in df.columns:
        out["human_stockout_rate_pct"] = _pct((df["human_missed"] > 0).mean())
    if oracle is not None:
        out["oracle_profit"] = _money(oracle)
        out["gap_to_oracle"] = _money(oracle - ml_profit)
    return out


def tool_dept_breakdown(sim: pd.DataFrame, scope: ChatScope) -> dict[str, Any]:
    """Side-by-side dept P&L. Skips the dept filter on purpose so you always get all three."""
    scoped = ChatScope(start_date=scope.start_date, end_date=scope.end_date)
    df = _apply_scope(sim, scoped)
    if df.empty or "dept_id" not in df.columns:
        return {"ok": False, "error": "No department data available."}

    rows = []
    for dept, g in df.groupby("dept_id"):
        ml_p = g["net_profit"].sum()
        hum_p = g["sma_profit"].sum()
        rows.append(
            {
                "dept_id": str(dept),
                "label": DEPT_LABELS.get(str(dept), str(dept)),
                "ml_profit": _money(ml_p),
                "human_profit": _money(hum_p),
                "profit_lift": _money(ml_p - hum_p),
                "waste_cost": _money(g["waste_cost"].sum()),
                "stockout_cost": _money(g["stockout_cost"].sum()),
                "stockout_rate_pct": _pct((g["missed_sales"] > 0).mean()),
                "items": int(g["item_id"].nunique()),
            }
        )
    rows.sort(key=lambda r: r["profit_lift"], reverse=True)
    return {"ok": True, "departments": rows}


def tool_worst_items(
    sim: pd.DataFrame,
    scope: ChatScope,
    *,
    metric: str = "profit_lift",
    limit: int = 10,
) -> dict[str, Any]:
    """
    Top offenders for a given metric.

    profit_lift sorts ascending (ML lost to human). waste / missed / stockout sort descending.
    """
    df = _apply_scope(sim, scope)
    if df.empty:
        return {"ok": False, "error": "No rows match the current filters."}

    lim = max(1, min(int(limit), MAX_ROWS))
    metric = (metric or "profit_lift").strip().lower()

    agg = df.groupby(["item_id", "dept_id"], as_index=False).agg(
        ml_profit=("net_profit", "sum"),
        human_profit=("sma_profit", "sum"),
        waste_cost=("waste_cost", "sum"),
        stockout_cost=("stockout_cost", "sum"),
        missed_sales=("missed_sales", "sum"),
        waste=("waste", "sum"),
        avg_sales=("sales", "mean"),
    )
    agg["profit_lift"] = agg["ml_profit"] - agg["human_profit"]

    if metric == "profit_lift":
        ranked = agg.nsmallest(lim, "profit_lift")
    elif metric in {"waste_cost", "missed_sales", "stockout_cost", "waste"}:
        ranked = agg.nlargest(lim, metric)
    else:
        return {
            "ok": False,
            "error": "metric must be one of: profit_lift, waste_cost, missed_sales, stockout_cost, waste",
        }

    items = []
    for _, r in ranked.iterrows():
        items.append(
            {
                "item_id": str(r["item_id"]),
                "dept_id": str(r["dept_id"]),
                "label": DEPT_LABELS.get(str(r["dept_id"]), str(r["dept_id"])),
                "ml_profit": _money(r["ml_profit"]),
                "human_profit": _money(r["human_profit"]),
                "profit_lift": _money(r["profit_lift"]),
                "waste_cost": _money(r["waste_cost"]),
                "stockout_cost": _money(r["stockout_cost"]),
                "missed_sales": _money(r["missed_sales"]),
                "avg_daily_sales": round(float(r["avg_sales"]), 3),
            }
        )
    return {"ok": True, "metric": metric, "items": items}


def tool_item_lookup(sim: pd.DataFrame, scope: ChatScope, item_id: str) -> dict[str, Any]:
    """One SKU: totals + last couple weeks of daily rows."""
    q = str(item_id or "").strip()
    if len(q) < 3:
        return {"ok": False, "error": "item_id must be at least 3 characters."}

    df = _apply_scope(sim, scope)
    # substring match: FOODS_1_001 and bare 001 both fine
    matches = df[df["item_id"].astype(str).str.contains(q, case=False, regex=False)]
    if matches.empty:
        return {"ok": False, "error": f"No item matching '{q}' in scope."}

    ids = matches["item_id"].astype(str).unique().tolist()
    if len(ids) > 1:
        return {
            "ok": False,
            "error": f"Ambiguous match ({len(ids)} items). Be more specific.",
            "candidates": ids[:15],
        }

    g = matches
    item = str(ids[0])
    dept = str(g["dept_id"].iloc[0])
    daily = (
        g.sort_values("date")[["date", "sales", "pred_demand", "stock_decision", "waste", "missed_sales", "net_profit"]]
        .tail(14)
        .copy()
    )
    daily["date"] = daily["date"].dt.strftime("%Y-%m-%d")

    return {
        "ok": True,
        "item_id": item,
        "dept_id": dept,
        "label": DEPT_LABELS.get(dept, dept),
        "days": int(len(g)),
        "ml_profit": _money(g["net_profit"].sum()),
        "human_profit": _money(g["sma_profit"].sum()),
        "profit_lift": _money(g["net_profit"].sum() - g["sma_profit"].sum()),
        "waste_cost": _money(g["waste_cost"].sum()),
        "stockout_cost": _money(g["stockout_cost"].sum()),
        "units_wasted": _money(g["waste"].sum()),
        "units_missed": _money(g["missed_sales"].sum()),
        "mae": _money((g["sales"] - g["pred_demand"]).abs().mean()),
        "recent_days": daily.to_dict(orient="records"),
    }


def tool_daily_trend(sim: pd.DataFrame, scope: ChatScope) -> dict[str, Any]:
    """Store-level daily series for charts / "which day tanked" questions."""
    df = _apply_scope(sim, scope)
    if df.empty:
        return {"ok": False, "error": "No rows match the current filters."}

    daily = (
        df.groupby("date", as_index=False)
        .agg(
            ml_profit=("net_profit", "sum"),
            human_profit=("sma_profit", "sum"),
            actual_demand=("sales", "sum"),
            pred_demand=("pred_demand", "sum"),
            waste_cost=("waste_cost", "sum"),
            stockout_cost=("stockout_cost", "sum"),
        )
        .sort_values("date")
    )
    daily["profit_lift"] = daily["ml_profit"] - daily["human_profit"]
    daily["date"] = daily["date"].dt.strftime("%Y-%m-%d")

    # holdout is ~60d; clamp anyway so we don't dump a huge json blob
    records = daily.to_dict(orient="records")
    if len(records) > 90:
        records = records[-90:]

    best = max(records, key=lambda r: r["profit_lift"]) if records else None
    worst = min(records, key=lambda r: r["profit_lift"]) if records else None
    return {
        "ok": True,
        "days": len(records),
        "best_day": best,
        "worst_day": worst,
        "series": records,
    }


def execute_tool(
    name: str,
    arguments: dict[str, Any],
    sim: pd.DataFrame,
    scope: ChatScope,
) -> str:
    if name == "run_summary":
        return _json(tool_run_summary(sim, scope))
    if name == "dept_breakdown":
        return _json(tool_dept_breakdown(sim, scope))
    if name == "worst_items":
        return _json(
            tool_worst_items(
                sim,
                scope,
                metric=str(arguments.get("metric") or "profit_lift"),
                limit=int(arguments.get("limit") or 10),
            )
        )
    if name == "item_lookup":
        return _json(tool_item_lookup(sim, scope, str(arguments.get("item_id") or "")))
    if name == "daily_trend":
        return _json(tool_daily_trend(sim, scope))
    return _json({"ok": False, "error": f"Unknown tool: {name}"})


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "run_summary",
            "description": (
                "Holdout totals for the current filters: ML vs human profit, "
                "waste, stockouts, gap to oracle."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "dept_breakdown",
            "description": (
                "FOODS_1 (produce) vs FOODS_2 (dairy/deli) vs FOODS_3 (packaged): "
                "profit lift, waste, stockouts."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "worst_items",
            "description": (
                "Rank SKUs. profit_lift = where ML lost to the human baseline; "
                "waste_cost / missed_sales for spoilage or stockouts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "metric": {
                        "type": "string",
                        "description": "profit_lift | waste_cost | missed_sales | stockout_cost | waste",
                    },
                    "limit": {"type": "integer", "description": "Max items, default 10, max 20"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "item_lookup",
            "description": "Lookup one item_id (substring ok). Returns totals + recent daily rows.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_id": {"type": "string", "description": "e.g. FOODS_1_001"},
                },
                "required": ["item_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "daily_trend",
            "description": "Daily ML vs human profit and demand, plus best/worst day.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]
