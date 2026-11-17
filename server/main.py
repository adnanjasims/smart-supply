"""
FastAPI entrypoint.

  uvicorn server.main:app --reload --port 8000
"""

from __future__ import annotations

import json
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from server.data import DataNotReady, cv_caption, feature_importance, load_bundle, load_sim

# mirrored from grocery.py so uvicorn never pays matplotlib/seaborn/lightgbm import cost
DEPT_COGS_RATIO = {
    "FOODS_1": 0.70,
    "FOODS_2": 0.60,
    "FOODS_3": 0.50,
}
DEPT_SPOILAGE_RATE = {
    "FOODS_1": 0.20,
    "FOODS_2": 0.10,
    "FOODS_3": 0.01,
}
STOCKOUT_PENALTY_FRACTION = 0.75


def critical_fractile(dept_id: str) -> float:
    cogs = DEPT_COGS_RATIO.get(dept_id, 0.60)
    spoil = DEPT_SPOILAGE_RATE.get(dept_id, 0.10)
    margin = 1.0 - cogs
    under = margin * (1.0 + STOCKOUT_PENALTY_FRACTION)
    over = cogs * spoil
    denom = under + over
    return under / denom if denom > 0 else 0.75


app = FastAPI(title="Smart Supply", version="1.0.0")

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _sim_or_503() -> pd.DataFrame:
    try:
        return load_sim()
    except DataNotReady as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


def _money(x: float) -> float:
    return round(float(x), 2)


def _pct(x: float) -> float:
    return round(float(x) * 100.0, 2)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1)
    dept_id: str = "(all)"


class SimulateRequest(BaseModel):
    spoilage_multiplier: float = Field(1.0, ge=0.0, le=5.0)
    stockout_penalty: float = Field(STOCKOUT_PENALTY_FRACTION, ge=0.0, le=3.0)
    dept_id: str = "(all)"


@app.get("/api/health")
def health() -> dict[str, Any]:
    ready = True
    detail = "ok"
    try:
        load_sim()
    except DataNotReady as e:
        ready = False
        detail = str(e)
    from chat import chat_config_present

    return {
        "ok": True,
        "data_ready": ready,
        "detail": detail,
        "chat_enabled": chat_config_present(),
    }


@app.get("/api/summary")
def summary() -> dict[str, Any]:
    sim = _sim_or_503()
    ml_profit = float(sim["net_profit"].sum())
    human_profit = float(sim["sma_profit"].sum())
    out: dict[str, Any] = {
        "date_start": str(sim["date"].min().date()),
        "date_end": str(sim["date"].max().date()),
        "items": int(sim["item_id"].nunique()),
        "rows": int(len(sim)),
        "ml_profit": _money(ml_profit),
        "human_profit": _money(human_profit),
        "profit_lift": _money(ml_profit - human_profit),
        "oracle_profit": _money(sim["oracle_profit"].sum()),
        "ml_waste_cost": _money(sim["waste_cost"].sum()),
        "ml_stockout_cost": _money(sim["stockout_cost"].sum()),
        "ml_stockout_rate_pct": _pct((sim["missed_sales"] > 0).mean()),
    }
    if "human_waste_cost" in sim.columns:
        hw = float(sim["human_waste_cost"].sum())
        out["human_waste_cost"] = _money(hw)
        out["waste_saved"] = _money(hw - float(sim["waste_cost"].sum()))
    if "waste" in sim.columns:
        out["ml_waste_units"] = round(float(sim["waste"].sum()), 1)
    if "human_waste" in sim.columns:
        hwu = float(sim["human_waste"].sum())
        out["human_waste_units"] = round(hwu, 1)
        out["waste_units_saved"] = round(hwu - float(sim["waste"].sum()), 1)
    return out


@app.get("/api/daily")
def daily() -> dict[str, Any]:
    sim = _sim_or_503()
    profit = (
        sim.groupby("date", as_index=False)
        .agg(ml_profit=("net_profit", "sum"), human_profit=("sma_profit", "sum"))
        .sort_values("date")
    )
    demand = (
        sim.groupby("date", as_index=False)
        .agg(
            actual=("sales", "sum"),
            predicted=("pred_demand", "sum"),
            q75=("pred_q75", "sum"),
            q90=("pred_q90", "sum"),
        )
        .sort_values("date")
    )
    return {
        "profit": [
            {
                "date": r.date.strftime("%Y-%m-%d"),
                "ml_profit": _money(r.ml_profit),
                "human_profit": _money(r.human_profit),
            }
            for r in profit.itertuples(index=False)
        ],
        "demand": [
            {
                "date": r.date.strftime("%Y-%m-%d"),
                "actual": _money(r.actual),
                "predicted": _money(r.predicted),
                "q75": _money(r.q75),
                "q90": _money(r.q90),
            }
            for r in demand.itertuples(index=False)
        ],
    }


@app.get("/api/departments")
def departments() -> dict[str, Any]:
    sim = _sim_or_503()
    if "dept_id" not in sim.columns:
        return {"departments": []}
    rows = []
    for dept, g in sim.groupby("dept_id"):
        ml_p = float(g["net_profit"].sum())
        hum_p = float(g["sma_profit"].sum())
        rows.append(
            {
                "dept_id": str(dept),
                "ml_profit": _money(ml_p),
                "human_profit": _money(hum_p),
                "profit_lift": _money(ml_p - hum_p),
                "waste_cost": _money(g["waste_cost"].sum()),
                "human_waste_cost": _money(g["human_waste_cost"].sum())
                if "human_waste_cost" in g.columns
                else None,
                "waste_saved": _money(g["human_waste_cost"].sum() - g["waste_cost"].sum())
                if "human_waste_cost" in g.columns
                else None,
                "stockout_cost": _money(g["stockout_cost"].sum()),
                "stockout_rate_pct": _pct((g["missed_sales"] > 0).mean()),
            }
        )
    rows.sort(key=lambda r: r["profit_lift"], reverse=True)
    return {"departments": rows}


@app.get("/api/items")
def items(
    item_id: str = Query("", description="substring filter"),
    start: str | None = None,
    end: str | None = None,
    limit: int = Query(200, ge=1, le=1000),
) -> dict[str, Any]:
    sim = _sim_or_503()
    df = sim
    if start:
        df = df[df["date"] >= pd.to_datetime(start)]
    if end:
        df = df[df["date"] <= pd.to_datetime(end)]
    q = item_id.strip()
    if q:
        df = df[df["item_id"].astype(str).str.contains(q, case=False, regex=False)]

    cols = [
        c
        for c in [
            "date",
            "item_id",
            "dept_id",
            "sales",
            "pred_demand",
            "pred_q75",
            "pred_q90",
            "stock_decision",
            "waste",
            "missed_sales",
            "net_profit",
            "sma_profit",
        ]
        if c in df.columns
    ]
    sliced = df[cols].sort_values(["date", "item_id"]).head(limit).copy()
    sliced["date"] = sliced["date"].dt.strftime("%Y-%m-%d")
    for c in sliced.columns:
        if c not in {"date", "item_id", "dept_id"}:
            sliced[c] = sliced[c].map(lambda v: _money(v) if pd.notna(v) else None)
    return {"rows": sliced.to_dict(orient="records"), "returned": int(len(sliced))}


@app.get("/api/features")
def features() -> dict[str, Any]:
    _sim_or_503()
    return {
        "importance": feature_importance(15),
        "cv_caption": cv_caption(),
    }


@app.post("/api/chat")
def chat(body: ChatRequest) -> dict[str, Any]:
    sim = _sim_or_503()
    from chat import answer_question, chat_config_present

    if not chat_config_present():
        raise HTTPException(
            status_code=503,
            detail="GEMINI_API_KEY not set. Copy .env.example to .env and add a key.",
        )
    result = answer_question(
        [m.model_dump() for m in body.messages],
        sim,
        dept_id=body.dept_id or "(all)",
    )
    if not result.get("ok"):
        raise HTTPException(status_code=502, detail=str(result.get("error") or "chat failed"))
    return result


def _prepare_orders_df(
    sim: pd.DataFrame,
    date: str | None,
    dept_id: str,
    search: str,
    status: str,
) -> tuple[str, list[str], pd.DataFrame]:
    unique_ts = sorted(pd.to_datetime(sim["date"].unique()), reverse=True)
    dates = [ts.strftime("%Y-%m-%d") for ts in unique_ts]
    if not dates:
        raise HTTPException(status_code=404, detail="No simulation dates available")
    target = date if date and date in dates else dates[0]
    day = sim[sim["date"] == pd.Timestamp(target)].copy()
    if dept_id and dept_id != "(all)" and "dept_id" in day.columns:
        day = day[day["dept_id"].astype(str) == dept_id]
    q = search.strip()
    if q:
        day = day[day["item_id"].astype(str).str.contains(q, case=False, regex=False)]

    day["recommended_order"] = day["stock_decision"].astype(int)
    day["baseline_order"] = (
        day["human_stock"].astype(int)
        if "human_stock" in day.columns
        else day["recommended_order"]
    )
    day["delta_vs_baseline"] = day["recommended_order"] - day["baseline_order"]
    day["action"] = np.where(
        day["delta_vs_baseline"] < 0,
        "Trim Overstock",
        np.where(day["delta_vs_baseline"] > 0, "Boost Stock", "Hold"),
    )
    if "sell_price" not in day.columns:
        day["sell_price"] = 0.0

    norm_status = status.strip().lower()
    if norm_status == "trim":
        day = day[day["delta_vs_baseline"] < 0]
    elif norm_status == "boost":
        day = day[day["delta_vs_baseline"] > 0]
    elif norm_status == "hold":
        day = day[day["delta_vs_baseline"] == 0]

    day = day.sort_values(["net_profit", "recommended_order"], ascending=[False, False])
    return target, dates, day


@app.get("/api/orders")
def orders(
    date: str | None = None,
    dept_id: str = "(all)",
    search: str = "",
    status: str = "all",
    limit: int = Query(150, ge=1, le=2000),
) -> dict[str, Any]:
    sim = _sim_or_503()
    target, dates, day = _prepare_orders_df(sim, date, dept_id, search, status)
    sliced = day.head(limit)
    rows = []
    for r in sliced.itertuples(index=False):
        rows.append(
            {
                "date": target,
                "item_id": str(r.item_id),
                "dept_id": str(getattr(r, "dept_id", "")),
                "sell_price": _money(getattr(r, "sell_price", 0.0)),
                "sales": int(r.sales),
                "pred_demand": round(float(r.pred_demand), 2),
                "pred_q75": round(float(r.pred_q75), 2),
                "pred_q90": round(float(r.pred_q90), 2),
                "recommended_order": int(r.recommended_order),
                "baseline_order": int(r.baseline_order),
                "delta_vs_baseline": int(r.delta_vs_baseline),
                "action": str(r.action),
                "waste_cost": _money(r.waste_cost),
                "net_profit": _money(r.net_profit),
                "sma_profit": _money(r.sma_profit),
            }
        )
    return {
        "selected_date": target,
        "dates": dates,
        "summary": {
            "sku_count": int(len(day)),
            "total_order_units": int(day["recommended_order"].sum()),
            "baseline_order_units": int(day["baseline_order"].sum()),
            "trim_count": int((day["delta_vs_baseline"] < 0).sum()),
            "boost_count": int((day["delta_vs_baseline"] > 0).sum()),
            "estimated_profit": _money(day["net_profit"].sum()),
            "baseline_profit": _money(day["sma_profit"].sum()),
            "estimated_waste_cost": _money(day["waste_cost"].sum()),
        },
        "rows": rows,
        "returned": len(rows),
    }


@app.get("/api/orders/export")
def export_orders(
    date: str | None = None,
    dept_id: str = "(all)",
    search: str = "",
    status: str = "all",
) -> Response:
    sim = _sim_or_503()
    target, _, day = _prepare_orders_df(sim, date, dept_id, search, status)
    export_cols = [
        c
        for c in [
            "item_id",
            "dept_id",
            "sell_price",
            "pred_demand",
            "pred_q75",
            "pred_q90",
            "recommended_order",
            "baseline_order",
            "delta_vs_baseline",
            "action",
            "waste_cost",
            "net_profit",
        ]
        if c in day.columns
    ]
    out = day[export_cols].copy()
    out.insert(0, "date", target)
    csv_text = out.to_csv(index=False)
    return Response(
        content=csv_text,
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="smartsupply-po-{target}.csv"'
        },
    )


@app.post("/api/simulate")
def simulate_policy(body: SimulateRequest) -> dict[str, Any]:
    sim = _sim_or_503()
    df = sim
    if body.dept_id and body.dept_id != "(all)" and "dept_id" in df.columns:
        df = df[df["dept_id"].astype(str) == body.dept_id]
    if df.empty:
        raise HTTPException(status_code=404, detail="No rows match selected department")

    depts = df["dept_id"].astype(str) if "dept_id" in df.columns else pd.Series(["FOODS_2"] * len(df), index=df.index)
    cogs_r = depts.map(DEPT_COGS_RATIO).fillna(0.60).to_numpy(dtype=float)
    base_spoil = depts.map(DEPT_SPOILAGE_RATE).fillna(0.10).to_numpy(dtype=float)
    spoil_r = np.clip(base_spoil * body.spoilage_multiplier, 0.0, 1.0)

    if "sell_price" in df.columns:
        price = df["sell_price"].to_numpy(dtype=float)
    else:
        # approximate price when compact test CSV omits sell_price
        price = np.full(len(df), 3.0, dtype=float)
    cost_p = price * cogs_r
    margin = price - cost_p
    sales = df["sales"].to_numpy(dtype=float)

    ml_stock = df["stock_decision"].to_numpy(dtype=float)
    hum_stock = (
        df["human_stock"].to_numpy(dtype=float)
        if "human_stock" in df.columns
        else ml_stock
    )

    def _score(stock: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        sold = np.minimum(stock, sales)
        unsold = np.maximum(0.0, stock - sales)
        missed = np.maximum(0.0, sales - stock)
        w_units = unsold * spoil_r
        w_cost = w_units * cost_p
        so_cost = missed * margin * body.stockout_penalty
        profit = sold * margin - w_cost - so_cost
        return profit, w_cost, w_units, so_cost

    ml_prof, ml_wc, ml_wu, ml_so = _score(ml_stock)
    hu_prof, hu_wc, hu_wu, hu_so = _score(hum_stock)

    dept_rows = []
    for d in sorted(depts.unique()):
        m = (depts == d).to_numpy()
        dept_rows.append(
            {
                "dept_id": d,
                "ml_profit": _money(ml_prof[m].sum()),
                "human_profit": _money(hu_prof[m].sum()),
                "profit_lift": _money(ml_prof[m].sum() - hu_prof[m].sum()),
                "ml_waste_cost": _money(ml_wc[m].sum()),
                "human_waste_cost": _money(hu_wc[m].sum()),
                "waste_saved": _money(hu_wc[m].sum() - ml_wc[m].sum()),
                "ml_stockout_cost": _money(ml_so[m].sum()),
            }
        )

    return {
        "spoilage_multiplier": body.spoilage_multiplier,
        "stockout_penalty": body.stockout_penalty,
        "dept_id": body.dept_id,
        "ml_profit": _money(ml_prof.sum()),
        "human_profit": _money(hu_prof.sum()),
        "profit_lift": _money(ml_prof.sum() - hu_prof.sum()),
        "ml_waste_cost": _money(ml_wc.sum()),
        "human_waste_cost": _money(hu_wc.sum()),
        "waste_saved": _money(hu_wc.sum() - ml_wc.sum()),
        "ml_waste_units": round(float(ml_wu.sum()), 1),
        "human_waste_units": round(float(hu_wu.sum()), 1),
        "ml_stockout_cost": _money(ml_so.sum()),
        "human_stockout_cost": _money(hu_so.sum()),
        "departments": dept_rows,
    }


@app.get("/api/item/{item_id}")
def item_detail(item_id: str) -> dict[str, Any]:
    sim = _sim_or_503()
    sub = sim[sim["item_id"].astype(str) == item_id].sort_values("date")
    if sub.empty:
        raise HTTPException(status_code=404, detail=f"Item not found: {item_id}")

    has_hwc = "human_waste_cost" in sub.columns
    has_hs = "human_stock" in sub.columns
    has_sp = "sell_price" in sub.columns
    series = []
    for r in sub.itertuples(index=False):
        series.append(
            {
                "date": r.date.strftime("%Y-%m-%d"),
                "sales": int(r.sales),
                "pred_demand": round(float(r.pred_demand), 2),
                "pred_q75": round(float(r.pred_q75), 2),
                "pred_q90": round(float(r.pred_q90), 2),
                "stock_decision": int(r.stock_decision),
                "human_stock": int(r.human_stock) if has_hs else int(r.stock_decision),
                "waste": round(float(r.waste), 2),
                "waste_cost": _money(r.waste_cost),
                "net_profit": _money(r.net_profit),
                "sma_profit": _money(r.sma_profit),
            }
        )

    return {
        "item_id": item_id,
        "dept_id": str(sub["dept_id"].iloc[0]) if "dept_id" in sub.columns else "",
        "avg_sell_price": _money(sub["sell_price"].mean()) if has_sp else 0.0,
        "days": int(len(sub)),
        "total_sales": int(sub["sales"].sum()),
        "total_ml_profit": _money(sub["net_profit"].sum()),
        "total_human_profit": _money(sub["sma_profit"].sum()),
        "total_waste_cost": _money(sub["waste_cost"].sum()),
        "total_human_waste_cost": _money(sub["human_waste_cost"].sum()) if has_hwc else None,
        "stockout_days": int((sub["missed_sales"] > 0).sum()),
        "series": series,
    }


@lru_cache(maxsize=1)
def _raw_cleaning_counts(n_items: int, lookback_days: int = 730) -> dict[str, int] | None:
    root = Path(__file__).resolve().parent.parent
    sales_path = root / "sales_train_validation.csv"
    cal_path = root / "calendar.csv"
    prices_path = root / "sell_prices.csv"
    if not (sales_path.exists() and cal_path.exists() and prices_path.exists()):
        return None
    cache_key = {
        "n_items": n_items,
        "lookback_days": lookback_days,
        "sales_mtime": sales_path.stat().st_mtime,
        "cal_mtime": cal_path.stat().st_mtime,
        "prices_mtime": prices_path.stat().st_mtime,
    }
    cache_file = Path(tempfile.gettempdir()) / "smartsupply_cleaning_cache.json"
    try:
        if cache_file.exists():
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            if cached.get("key") == cache_key:
                return cached["counts"]
    except Exception:
        pass
    try:
        with sales_path.open("r", encoding="utf-8") as fh:
            hdr = fh.readline().rstrip("\n").split(",")
            d_cols = [c for c in hdr if c.startswith("d_")][-lookback_days:]
            # match CA_1 FOODS item rows by prefix without parsing 1,919 numeric columns
            item_ids: set[str] = set()
            for line in fh:
                if line.startswith("FOODS_") and "_CA_1_" in line[:48]:
                    parts = line.split(",", 6)
                    if len(parts) >= 6 and parts[4] == "CA_1" and parts[3] == "FOODS":
                        item_ids.add(parts[1])
        if not item_ids or len(item_ids) != n_items:
            return None

        cal = pd.read_csv(cal_path, usecols=["d", "wm_yr_wk"])
        cal = cal[cal["d"].isin(d_cols)]
        post_warmup_days = set(d_cols[28:] if len(d_cols) > 28 else d_cols)
        warmup_rows = int(len(item_ids) * min(28, len(d_cols)))

        # count days per week post-warmup and stream matching CA_1,FOODS_ price keys
        wk_day_counts = cal[cal["d"].isin(post_warmup_days)]["wm_yr_wk"].value_counts().to_dict()
        priced_pairs: set[tuple[str, int]] = set()
        valid_wks = set(wk_day_counts)
        with prices_path.open("r", encoding="utf-8") as fh:
            next(fh, None)
            for line in fh:
                if line.startswith("CA_1,FOODS_"):
                    parts = line.split(",", 4)
                    if len(parts) >= 3 and parts[1] in item_ids:
                        wk = int(parts[2])
                        if wk in valid_wks:
                            priced_pairs.add((parts[1], wk))

        post_total = sum(wk_day_counts.values()) * len(item_ids)
        priced_post = sum(wk_day_counts[wk] for _, wk in priced_pairs)
        missing_price = max(0, int(post_total - priced_post))
        raw_rows = int(len(item_ids) * len(d_cols))
        kept_rows = max(0, raw_rows - warmup_rows - missing_price)
        counts = {
            "lookback_days": int(len(d_cols)),
            "raw_rows": raw_rows,
            "warmup_dropped": warmup_rows,
            "missing_price_dropped": missing_price,
            "kept_rows": kept_rows,
        }
        try:
            cache_file.write_text(
                json.dumps({"key": cache_key, "counts": counts}), encoding="utf-8"
            )
        except Exception:
            pass
        return counts
    except Exception:
        return None


def _categorize_feature(name: str) -> str:
    if name.startswith("sales_lag_") or name == "seasonal_naive" or name == "same_dow_mean_4w":
        return "Causal Lags & Same-Weekday"
    if name.startswith(("rolling_", "ewma_", "sales_trend")):
        return "Rolling & EWMA Momentum"
    if name in {"zero_rate_28", "days_since_zero"}:
        return "Intermittency & Zero Spells"
    if "price" in name:
        return "Price Elasticity"
    return "Calendar, SNAP & Payday"


@app.get("/api/pipeline")
def pipeline() -> dict[str, Any]:
    sim = _sim_or_503()
    n_items = int(sim["item_id"].nunique())
    n_days = int(sim["date"].nunique())
    bundle = load_bundle() or {}

    cleaning = _raw_cleaning_counts(n_items, 730)
    if cleaning is None:
        cleaning = {
            "lookback_days": n_days,
            "raw_rows": int(len(sim)),
            "warmup_dropped": 0,
            "missing_price_dropped": 0,
            "kept_rows": int(len(sim)),
        }

    dept_caps = {"FOODS_1": 0.63, "FOODS_2": 0.69, "FOODS_3": 0.93}
    dept_names = {
        "FOODS_1": "Produce & Fresh",
        "FOODS_2": "Dairy & Deli",
        "FOODS_3": "Packaged Foods",
    }
    depts_out = []
    if "dept_id" in sim.columns:
        for dept_id in ["FOODS_1", "FOODS_2", "FOODS_3"]:
            g = sim[sim["dept_id"].astype(str) == dept_id]
            if g.empty:
                continue
            avg_price = float(g["sell_price"].mean()) if "sell_price" in g.columns else 0.0
            cogs = float(DEPT_COGS_RATIO.get(dept_id, 0.60))
            spoil = float(DEPT_SPOILAGE_RATE.get(dept_id, 0.10))
            ml_p = float(g["net_profit"].sum())
            hum_p = float(g["sma_profit"].sum())
            ml_w = float(g["waste_cost"].sum())
            hum_w = float(g["human_waste_cost"].sum()) if "human_waste_cost" in g.columns else ml_w
            quantiles = {}
            for q_col in ["pred_q50", "pred_q62", "pred_q75", "pred_q83", "pred_q90", "pred_q95"]:
                if q_col in g.columns:
                    quantiles[q_col.replace("pred_", "")] = round(float(g[q_col].mean()), 2)
            depts_out.append(
                {
                    "dept_id": dept_id,
                    "label": dept_names.get(dept_id, dept_id),
                    "sku_count": int(g["item_id"].nunique()),
                    "cogs_ratio": cogs,
                    "spoilage_rate": spoil,
                    "stockout_penalty_fraction": STOCKOUT_PENALTY_FRACTION,
                    "theoretical_cf": round(float(critical_fractile(dept_id)), 3),
                    "waste_aware_cap": dept_caps.get(dept_id, 0.75),
                    "avg_sell_price": round(avg_price, 2),
                    "unit_margin": round(avg_price * (1.0 - cogs), 2),
                    "unit_spoilage_cost": round(avg_price * cogs * spoil, 2),
                    "unit_stockout_cost": round(
                        avg_price * (1.0 - cogs) * STOCKOUT_PENALTY_FRACTION, 2
                    ),
                    "zero_sale_pct": _pct((g["sales"] == 0).mean()),
                    "avg_daily_sales": round(float(g["sales"].mean()), 2),
                    "mean_pred_demand": round(float(g["pred_demand"].mean()), 2),
                    "mean_ml_stock": round(float(g["stock_decision"].mean()), 2),
                    "mean_human_stock": round(float(g["human_stock"].mean()), 2)
                    if "human_stock" in g.columns
                    else round(float(g["stock_decision"].mean()), 2),
                    "ml_profit": _money(ml_p),
                    "human_profit": _money(hum_p),
                    "profit_lift": _money(ml_p - hum_p),
                    "ml_waste_cost": _money(ml_w),
                    "human_waste_cost": _money(hum_w),
                    "waste_saved": _money(hum_w - ml_w),
                    "quantile_means": quantiles,
                }
            )

    features_list = [str(f) for f in (bundle.get("features") or [])]
    group_counts: dict[str, list[str]] = {}
    for f in features_list:
        grp = _categorize_feature(f)
        group_counts.setdefault(grp, []).append(f)

    cv_scores = [round(float(x), 3) for x in (bundle.get("cv_scores") or [])]
    holdout_mae = round(float((sim["pred_demand"] - sim["sales"]).abs().mean()), 3)
    best_params = bundle.get("best_params") or {}

    return {
        "store_id": "CA_1",
        "category": "FOODS",
        "skus": n_items,
        "holdout_days": n_days,
        "holdout_rows": int(len(sim)),
        "zero_sale_pct": _pct((sim["sales"] == 0).mean()),
        "holdout_mae": holdout_mae,
        "cleaning": cleaning,
        "feature_count": len(features_list),
        "feature_groups": [
            {"group": g, "count": len(cols), "examples": cols[:5]}
            for g, cols in group_counts.items()
        ],
        "cv_scores": cv_scores,
        "cv_mae_mean": round(float(np.mean(cv_scores)), 3) if cv_scores else None,
        "cv_mae_std": round(float(np.std(cv_scores)), 3) if cv_scores else None,
        "quantile_ladder": ["q50", "q62", "q75", "q83", "q90", "q95"],
        "best_params": {
            "learning_rate": round(float(best_params["learning_rate"]), 4)
            if "learning_rate" in best_params
            else None,
            "num_leaves": int(best_params["num_leaves"])
            if "num_leaves" in best_params
            else None,
            "min_data_in_leaf": int(best_params["min_data_in_leaf"])
            if "min_data_in_leaf" in best_params
            else None,
        },
        "departments": depts_out,
    }


_DIST_DIR = Path(__file__).resolve().parent.parent / "web" / "dist"
if _DIST_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(_DIST_DIR), html=True), name="web")
