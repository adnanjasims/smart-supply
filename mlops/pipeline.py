"""Train + track + gate + optionally promote."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from grocery import SmartSupplyAI
from mlops.config import load_config
from mlops.registry import ModelRegistry
from mlops.tracking import RunTracker
from mlops.validate import evaluate_gates


def run_training_pipeline(
    config_path: str | Path | None = None,
    overrides: dict | None = None,
    promote: bool | None = None,
) -> dict[str, Any]:
    cfg = load_config(config_path, overrides=overrides)
    data = cfg["data"]
    train = cfg["train"]
    mlops_cfg = cfg["mlops"]
    gates = cfg.get("gates", {})

    artifacts_dir = Path(mlops_cfg.get("artifacts_dir", "artifacts"))
    tracker = RunTracker(
        artifacts_dir=artifacts_dir,
        experiment_name=mlops_cfg.get("experiment_name", "smartsupply-demand"),
    )
    print(f"[mlops] started run {tracker.run_id}")
    print(f"[mlops] artifacts -> {tracker.run_dir}")

    tracker.log_params({
        "store_id": data["store_id"],
        "cat_id": data["cat_id"],
        "lookback_days": data["lookback_days"],
        "holdout_days": data["holdout_days"],
        "n_cv_splits": train["n_cv_splits"],
        "num_boost_round": train["num_boost_round"],
        "cv_boost_round": train["cv_boost_round"],
        "skip_tuning": train["skip_tuning"],
        "seed": train.get("seed", 42),
        "config_path": str(config_path or "config/default.yaml"),
    })
    tracker.log_data_hashes({
        "sales": data["sales"],
        "calendar": data["calendar"],
        "prices": data["prices"],
    })

    app = SmartSupplyAI(
        sales_path=data["sales"],
        calendar_path=data["calendar"],
        prices_path=data["prices"],
        store_id=data["store_id"],
        cat_id=data["cat_id"],
        lookback_days=int(data["lookback_days"]),
        holdout_days=int(data["holdout_days"]),
        output_dir=tracker.run_dir,
        n_cv_splits=int(train["n_cv_splits"]),
        num_boost_round=int(train["num_boost_round"]),
        cv_boost_round=int(train["cv_boost_round"]),
        skip_tuning=bool(train["skip_tuning"]),
    )

    try:
        app.load_and_clean()
        app.feature_engineering()
        app.train_model()
        app.run_simulation()
        app.generate_dashboard()
        app.export_results()
        app.print_report()

        metrics = app.collect_metrics()
        tracker.log_metrics(metrics)
        tracker.log_params({"best_params": app.best_params, "features": app.features})

        for name in (
            "lightgbm_bundle.pkl",
            "lightgbm_model.pkl",
            "lightgbm_model_q75.pkl",
            "lightgbm_model_q90.pkl",
            "m5_supply_chain_forecast.csv",
            "smart_supply_dashboard.png",
        ):
            path = tracker.run_dir / name
            if path.exists():
                tracker.log_artifact(path)

        gate = evaluate_gates(metrics, gates, drop_stats=app.drop_stats)
        print("[mlops] quality gates:")
        for line in gate.summary_lines():
            print(line)

        do_promote = mlops_cfg.get("auto_promote", False) if promote is None else promote
        if not gate.passed:
            tracker.finish(status="rejected")
            print(f"[mlops] run {tracker.run_id} rejected by gates")
            return {
                "run_id": tracker.run_id,
                "status": "rejected",
                "metrics": metrics,
                "gates": gate.checks,
                "run_dir": str(tracker.run_dir),
            }

        tracker.finish(status="completed")
        result = {
            "run_id": tracker.run_id,
            "status": "completed",
            "metrics": metrics,
            "gates": gate.checks,
            "run_dir": str(tracker.run_dir),
            "promoted": False,
        }
        if do_promote:
            reg = ModelRegistry(artifacts_dir)
            stage_dir = reg.promote(tracker.run_id)
            result["promoted"] = True
            result["production_dir"] = str(stage_dir)
            print(f"[mlops] promoted {tracker.run_id} -> {stage_dir}")
        return result

    except Exception as exc:
        tracker.finish(status="failed", error=str(exc))
        raise
