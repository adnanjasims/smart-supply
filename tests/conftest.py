"""Shared fixtures: tiny M5-shaped CSVs for fast offline tests."""

from __future__ import annotations

import pytest

from grocery import SmartSupplyAI
from tests.m5_factory import write_mini_m5


@pytest.fixture
def mini_m5(tmp_path):
    return write_mini_m5(tmp_path, n_days=90, n_items=3)


@pytest.fixture
def trained_pipeline(mini_m5, tmp_path):
    out = tmp_path / "out"
    app = SmartSupplyAI(
        sales_path=mini_m5["sales"],
        calendar_path=mini_m5["calendar"],
        prices_path=mini_m5["prices"],
        store_id=mini_m5["store_id"],
        lookback_days=90,
        holdout_days=14,
        output_dir=out,
        n_cv_splits=2,
        num_boost_round=40,
        cv_boost_round=30,
        skip_tuning=True,
    )
    app.load_and_clean()
    app.feature_engineering()
    app.train_model()
    app.run_simulation()
    app.export_results()
    return app
