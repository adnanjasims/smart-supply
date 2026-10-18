"""
FastAPI entrypoint.

  uvicorn server.main:app --reload --port 8000
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from server.data import DataNotReady, cv_caption, feature_importance, load_sim

app = FastAPI(title="Smart Supply", version="1.0.0")

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
