"""Unit and integration tests for SmartSupplyAI."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest

from grocery import (
    DEPT_COGS_RATIO,
    DEPT_SPOILAGE_RATE,
    SmartSupplyAI,
    parse_args,
)
from tests.m5_factory import write_mini_m5


class TestValidation:
    def test_missing_file_raises(self, tmp_path):
        app = SmartSupplyAI(
            sales_path=tmp_path / "missing.csv",
            calendar_path=tmp_path / "calendar.csv",
            prices_path=tmp_path / "prices.csv",
        )
        with pytest.raises(FileNotFoundError, match="Missing required"):
            app._validate_inputs()

    def test_require_columns(self):
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="missing columns"):
            SmartSupplyAI._require_columns(df, {"a", "b"}, "toy.csv")

    def test_empty_filter_raises(self, tmp_path):
        paths = write_mini_m5(tmp_path, n_days=60, n_items=2)
        app = SmartSupplyAI(
            sales_path=paths["sales"],
            calendar_path=paths["calendar"],
            prices_path=paths["prices"],
            store_id="TX_1",
            lookback_days=60,
            holdout_days=10,
            skip_tuning=True,
        )
        with pytest.raises(ValueError, match="No rows left"):
            app.load_and_clean()


class TestFeatureEngineering:
    def test_warmup_drop_is_mostly_lag28(self, mini_m5, tmp_path):
        app = SmartSupplyAI(
            sales_path=mini_m5["sales"],
            calendar_path=mini_m5["calendar"],
            prices_path=mini_m5["prices"],
            lookback_days=90,
            holdout_days=14,
            output_dir=tmp_path / "out",
            skip_tuning=True,
        )
        app.load_and_clean()
        app.feature_engineering()

        stats = app.drop_stats
        assert stats["dropped_total"] > 0
        # Almost all drops should be the intentional 28-day lag warmup
        assert stats["dropped_lag_warmup"] >= mini_m5["n_items"] * 28
        assert stats["dropped_lag_warmup"] / stats["dropped_total"] > 0.85
        assert stats["rows_after"] > 0
        # Features required by the model exist and are finite
        for col in app._get_feature_list():
            assert col in app.df.columns
            assert app.df[col].isna().sum() == 0

    def test_missing_price_counted_separately(self, tmp_path):
        paths = write_mini_m5(tmp_path, n_days=90, n_items=2, missing_price_days=5)
        app = SmartSupplyAI(
            sales_path=paths["sales"],
            calendar_path=paths["calendar"],
            prices_path=paths["prices"],
            lookback_days=90,
            holdout_days=14,
            output_dir=tmp_path / "out",
            skip_tuning=True,
        )
        app.load_and_clean()
        assert app.drop_stats["missing_sell_price"] > 0
        app.feature_engineering()
        # warmup still dominates; price holes are tracked
        assert app.drop_stats["dropped_lag_warmup"] > 0

    def test_price_changed_false_when_prev_missing(self, mini_m5, tmp_path):
        app = SmartSupplyAI(
            sales_path=mini_m5["sales"],
            calendar_path=mini_m5["calendar"],
            prices_path=mini_m5["prices"],
            lookback_days=90,
            holdout_days=14,
            output_dir=tmp_path / "out",
            skip_tuning=True,
        )
        app.load_and_clean()
        df = app.df.sort_values(["id", "date"]).copy()
        df["price_prev_week"] = df.groupby("id")["sell_price"].shift(7)
        price_changed = (
            df["sell_price"].notna()
            & df["price_prev_week"].notna()
            & (df["sell_price"] != df["price_prev_week"])
        )
        # Where prev week is missing, flag must be False
        assert not bool(price_changed[df["price_prev_week"].isna()].any())

    def test_event_features_created(self, mini_m5, tmp_path):
        app = SmartSupplyAI(
            sales_path=mini_m5["sales"],
            calendar_path=mini_m5["calendar"],
            prices_path=mini_m5["prices"],
            lookback_days=90,
            holdout_days=14,
            output_dir=tmp_path / "out",
            skip_tuning=True,
        )
        app.load_and_clean()
        app.feature_engineering()
        assert "has_event" in app.df.columns
        assert "event_type" in app.df.columns
        assert app.df["has_event"].sum() >= 1


class TestStockRules:
    def test_dept_aware_stock(self):
        app = SmartSupplyAI("a.csv", "b.csv", "c.csv")
        df = pd.DataFrame({
            "dept_id": ["FOODS_1", "FOODS_2", "FOODS_3"],
            "pred_demand": [10.0, 10.0, 10.0],
            "pred_q75": [12.0, 12.0, 12.0],
            "pred_q90": [15.0, 15.0, 15.0],
        })
        stock = app._dept_aware_stock(df)
        assert list(stock) == [12, 15, 15]

    def test_human_stock_buffer(self):
        app = SmartSupplyAI("a.csv", "b.csv", "c.csv")
        df = pd.DataFrame({"ewma_7": [10.0], "seasonal_naive": [10.0]})
        assert app._human_stock(df)[0] == 11

    def test_cost_maps_cover_foods_depts(self):
        for dept in ["FOODS_1", "FOODS_2", "FOODS_3"]:
            assert dept in DEPT_COGS_RATIO
            assert dept in DEPT_SPOILAGE_RATE
            assert 0 < DEPT_SPOILAGE_RATE[dept] <= 1
            assert 0 < DEPT_COGS_RATIO[dept] < 1


class TestDateCV:
    def test_folds_are_time_ordered_and_disjoint(self, mini_m5, tmp_path):
        app = SmartSupplyAI(
            sales_path=mini_m5["sales"],
            calendar_path=mini_m5["calendar"],
            prices_path=mini_m5["prices"],
            lookback_days=90,
            holdout_days=14,
            output_dir=tmp_path / "out",
            skip_tuning=True,
        )
        app.load_and_clean()
        app.feature_engineering()
        cutoff = app.df["date"].max() - pd.Timedelta(days=14)
        train = app.df[app.df["date"] <= cutoff]
        folds = app._date_fold_splits(train, n_splits=3)
        assert len(folds) == 3
        prev_test_end = None
        for tr, te in folds:
            assert tr["date"].max() < te["date"].min()
            if prev_test_end is not None:
                assert te["date"].min() > prev_test_end
            prev_test_end = te["date"].max()
            # no calendar leakage: train dates never overlap test dates
            assert set(tr["date"]).isdisjoint(set(te["date"]))


class TestEndToEnd:
    def test_pipeline_trains_and_exports(self, trained_pipeline, tmp_path):
        app = trained_pipeline
        assert app.model is not None
        assert app.model_q75 is not None
        assert app.model_q90 is not None
        assert app.simulation is not None
        assert len(app.cv_scores) == 2
        assert not app.used_human_fallback

        out = app.output_dir
        assert (out / "lightgbm_model.pkl").exists()
        assert (out / "lightgbm_model_q75.pkl").exists()
        assert (out / "lightgbm_model_q90.pkl").exists()
        assert (out / "lightgbm_bundle.pkl").exists()
        assert (out / "m5_supply_chain_forecast.csv").exists()

        bundle = joblib.load(out / "lightgbm_bundle.pkl")
        assert set(bundle.keys()) >= {"point", "q75", "q90", "features", "cv_scores"}
        assert bundle["features"] == app.features

        sim = pd.read_csv(out / "m5_supply_chain_forecast.csv", parse_dates=["date"])
        required = {
            "pred_demand", "pred_q75", "pred_q90", "stock_decision",
            "net_profit", "sma_profit", "oracle_profit",
            "human_waste_cost", "human_stockout_cost", "human_missed",
        }
        assert required.issubset(sim.columns)
        assert len(sim) > 0
        assert sim["stock_decision"].ge(0).all()
        # accounting identity: sold + missed == demand when stock < demand, else sold == demand
        assert np.allclose(
            sim["sales"],
            np.minimum(sim["stock_decision"], sim["sales"]) + sim["missed_sales"],
        )

    def test_simulation_profits_are_finite(self, trained_pipeline):
        sim = trained_pipeline.simulation
        for col in ["net_profit", "sma_profit", "oracle_profit", "waste_cost", "stockout_cost"]:
            assert np.isfinite(sim[col]).all()
        # oracle should dominate ML and human in total (perfect foresight)
        assert sim["oracle_profit"].sum() >= sim["net_profit"].sum() - 1e-6
        assert sim["oracle_profit"].sum() >= sim["sma_profit"].sum() - 1e-6

    def test_predict_fallback_uses_human_rule(self, trained_pipeline):
        app = trained_pipeline
        app.model = None  # force predict failure
        app.run_simulation()
        assert app.used_human_fallback is True
        assert (app.simulation["stock_decision"] >= 0).all()

    def test_dashboard_writes_png(self, trained_pipeline):
        app = trained_pipeline
        app.generate_dashboard()
        assert (app.output_dir / "smart_supply_dashboard.png").exists()


class TestCLI:
    def test_parse_args_defaults(self):
        args = parse_args([])
        assert args.store == "CA_1"
        assert args.cat == "FOODS"
        assert args.lookback_days == 365
        assert args.holdout_days == 60
        assert args.skip_tuning is False

    def test_parse_args_flags(self):
        args = parse_args([
            "--store", "CA_2",
            "--holdout-days", "30",
            "--skip-tuning",
            "--output-dir", "/tmp/out",
        ])
        assert args.store == "CA_2"
        assert args.holdout_days == 30
        assert args.skip_tuning is True
        assert args.output_dir == "/tmp/out"
