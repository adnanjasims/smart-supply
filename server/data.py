"""Load and cache the holdout CSV + LightGBM bundle for API handlers."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "m5_supply_chain_forecast.csv"
BUNDLE_PATH = ROOT / "lightgbm_bundle.pkl"
MODEL_PATH = ROOT / "lightgbm_model.pkl"


class DataNotReady(Exception):
    """CSV missing; run grocery.py first."""


@lru_cache(maxsize=4)
def _read_sim(mtime_ns: int) -> pd.DataFrame:
    return pd.read_csv(CSV_PATH, parse_dates=["date"])


def load_sim() -> pd.DataFrame:
    if not CSV_PATH.exists():
        raise DataNotReady(
            f"Missing {CSV_PATH.name}. Run `python3 grocery.py` from the project root first."
        )
    # key on mtime so resim_stock.py / grocery.py exports show up without restarting uvicorn
    return _read_sim(CSV_PATH.stat().st_mtime_ns)


def clear_sim_cache() -> None:
    _read_sim.cache_clear()



@lru_cache(maxsize=1)
def load_bundle() -> dict[str, Any] | None:
    if not BUNDLE_PATH.exists():
        return None
    import joblib

    return joblib.load(BUNDLE_PATH)


@lru_cache(maxsize=1)
def load_point_model() -> Any | None:
    bundle = load_bundle()
    if bundle is not None and bundle.get("point") is not None:
        return bundle["point"]
    if not MODEL_PATH.exists():
        return None
    import joblib

    return joblib.load(MODEL_PATH)


def feature_importance(top_n: int = 15) -> list[dict[str, Any]]:
    model = load_point_model()
    if model is None:
        return []
    gain = model.feature_importance(importance_type="gain")
    names = model.feature_name()
    rows = sorted(
        [{"feature": n, "gain": float(g)} for n, g in zip(names, gain)],
        key=lambda r: r["gain"],
        reverse=True,
    )
    return rows[:top_n]


def cv_caption() -> str | None:
    bundle = load_bundle()
    if not bundle or not bundle.get("cv_scores"):
        return None
    import numpy as np

    scores = bundle["cv_scores"]
    return (
        f"date-based cv mae: {float(np.mean(scores)):.3f} +/- {float(np.std(scores)):.3f} "
        f"across {len(scores)} folds"
    )
