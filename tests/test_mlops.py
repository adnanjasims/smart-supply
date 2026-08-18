"""Tests for local MLOps tracking, gates, and registry."""

from __future__ import annotations

import json

import pytest
import yaml

from mlops.config import load_config
from mlops.pipeline import run_training_pipeline
from mlops.registry import ModelRegistry
from mlops.tracking import RunTracker
from mlops.validate import evaluate_gates
from tests.m5_factory import write_mini_m5


def _tiny_config(tmp_path, paths, artifacts_dir) -> str:
    cfg = {
        "data": {
            "sales": str(paths["sales"]),
            "calendar": str(paths["calendar"]),
            "prices": str(paths["prices"]),
            "store_id": paths["store_id"],
            "cat_id": "FOODS",
            "lookback_days": 90,
            "holdout_days": 14,
        },
        "train": {
            "n_cv_splits": 2,
            "num_boost_round": 40,
            "cv_boost_round": 30,
            "skip_tuning": True,
            "seed": 42,
        },
        "mlops": {
            "artifacts_dir": str(artifacts_dir),
            "experiment_name": "smartsupply-test",
            "auto_promote": False,
        },
        "gates": {
            "max_val_mae": 10.0,
            "min_profit_lift": -1e9,
            "min_q90_ge_q75_frac": 0.5,
            "require_no_fallback": True,
            "min_rows_after_features": 50,
        },
    }
    path = tmp_path / "test_config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return str(path)


class TestConfig:
    def test_load_default_config(self):
        cfg = load_config()
        assert "data" in cfg and "gates" in cfg
        assert cfg["data"]["store_id"] == "CA_1"

    def test_overrides_merge(self, tmp_path):
        cfg = load_config(overrides={"data": {"store_id": "CA_2"}, "train": {"skip_tuning": True}})
        assert cfg["data"]["store_id"] == "CA_2"
        assert cfg["data"]["cat_id"] == "FOODS"
        assert cfg["train"]["skip_tuning"] is True


class TestGates:
    def test_gates_pass(self):
        metrics = {
            "val_mae": 1.2,
            "profit_lift": 100.0,
            "q90_ge_q75_frac": 0.95,
            "used_human_fallback": False,
        }
        gates = {
            "max_val_mae": 3.0,
            "min_profit_lift": 0.0,
            "min_q90_ge_q75_frac": 0.8,
            "require_no_fallback": True,
            "min_rows_after_features": 10,
        }
        result = evaluate_gates(metrics, gates, drop_stats={"rows_after": 100})
        assert result.passed
        assert all(c["passed"] for c in result.checks)

    def test_gates_fail_on_mae(self):
        metrics = {"val_mae": 9.0, "used_human_fallback": False}
        gates = {"max_val_mae": 3.0, "require_no_fallback": True}
        result = evaluate_gates(metrics, gates)
        assert not result.passed
        assert any(c["name"] == "max_val_mae" and not c["passed"] for c in result.checks)


class TestTrackingAndRegistry:
    def test_tracker_writes_files(self, tmp_path):
        tracker = RunTracker(tmp_path / "artifacts", experiment_name="unit")
        tracker.log_params({"lr": 0.05})
        tracker.log_metrics({"val_mae": 1.1})
        sample = tmp_path / "note.txt"
        sample.write_text("ok")
        tracker.log_artifact(sample)
        tracker.finish("completed")

        assert (tracker.run_dir / "params.json").exists()
        assert (tracker.run_dir / "metrics.json").exists()
        assert (tracker.run_dir / "meta.json").exists()
        assert (tracker.run_dir / "note.txt").exists()
        rows = RunTracker.list_runs(tmp_path / "artifacts")
        assert rows and rows[0]["run_id"] == tracker.run_id

    def test_promote_and_load(self, tmp_path):
        artifacts = tmp_path / "artifacts"
        tracker = RunTracker(artifacts, experiment_name="unit")
        #minimal fake bundle
        import joblib
        bundle = {"point": "fake", "features": ["a"]}
        joblib.dump(bundle, tracker.run_dir / "lightgbm_bundle.pkl")
        tracker.log_metrics({"val_mae": 1.0})
        tracker.finish("completed")

        reg = ModelRegistry(artifacts)
        stage = reg.promote(tracker.run_id)
        assert (stage / "current.json").exists()
        assert reg.current()["run_id"] == tracker.run_id
        loaded = reg.load_bundle()
        assert loaded["features"] == ["a"]


class TestPipelineE2E:
    def test_tracked_train_promote(self, tmp_path):
        paths = write_mini_m5(tmp_path, n_days=90, n_items=3)
        artifacts = tmp_path / "artifacts"
        cfg_path = _tiny_config(tmp_path, paths, artifacts)

        result = run_training_pipeline(config_path=cfg_path, promote=True)
        assert result["status"] == "completed"
        assert result["promoted"] is True
        assert (artifacts / "runs" / result["run_id"] / "metrics.json").exists()
        assert (artifacts / "registry" / "production" / "current.json").exists()

        with (artifacts / "runs" / result["run_id"] / "metrics.json").open() as f:
            metrics = json.load(f)
        assert "val_mae" in metrics
        assert "profit_lift" in metrics

    def test_rejected_when_gates_strict(self, tmp_path):
        paths = write_mini_m5(tmp_path, n_days=90, n_items=3)
        artifacts = tmp_path / "artifacts"
        cfg_path = _tiny_config(tmp_path, paths, artifacts)
        #impossible mae gate
        result = run_training_pipeline(
            config_path=cfg_path,
            overrides={"gates": {"max_val_mae": 0.0001}},
            promote=False,
        )
        assert result["status"] == "rejected"
        info = RunTracker.load_run(artifacts, result["run_id"])
        assert info["meta"]["status"] == "rejected"
