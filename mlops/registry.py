"""Simple model registry: promote a tracked run to production."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib


class ModelRegistry:
    def __init__(self, artifacts_dir: str | Path):
        self.artifacts_dir = Path(artifacts_dir)
        self.registry_dir = self.artifacts_dir / "registry"
        self.production_dir = self.registry_dir / "production"
        self.registry_dir.mkdir(parents=True, exist_ok=True)

    def promote(self, run_id: str, stage: str = "production") -> Path:
        run_dir = self.artifacts_dir / "runs" / run_id
        bundle = run_dir / "lightgbm_bundle.pkl"
        if not bundle.exists():
            raise FileNotFoundError(f"no lightgbm_bundle.pkl in run {run_id}")

        meta_path = run_dir / "meta.json"
        if meta_path.exists():
            with meta_path.open() as f:
                meta = json.load(f)
            if meta.get("status") == "rejected":
                raise ValueError(f"run {run_id} failed quality gates and cannot be promoted")

        stage_dir = self.registry_dir / stage
        if stage_dir.exists():
            shutil.rmtree(stage_dir)
        stage_dir.mkdir(parents=True, exist_ok=True)

        #copy core artifacts into the stage folder
        for name in (
            "lightgbm_bundle.pkl",
            "lightgbm_model.pkl",
            "lightgbm_model_q75.pkl",
            "lightgbm_model_q90.pkl",
            "params.json",
            "metrics.json",
            "meta.json",
            "m5_supply_chain_forecast.csv",
            "smart_supply_dashboard.png",
        ):
            src = run_dir / name
            if src.exists():
                shutil.copy2(src, stage_dir / name)

        pointer = {
            "stage": stage,
            "run_id": run_id,
            "promoted_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "bundle": str(stage_dir / "lightgbm_bundle.pkl"),
        }
        with (stage_dir / "current.json").open("w") as f:
            json.dump(pointer, f, indent=2)
        with (self.registry_dir / f"{stage}_history.jsonl").open("a") as f:
            f.write(json.dumps(pointer) + "\n")
        return stage_dir

    def current(self, stage: str = "production") -> dict[str, Any] | None:
        path = self.registry_dir / stage / "current.json"
        if not path.exists():
            return None
        with path.open() as f:
            return json.load(f)

    def load_bundle(self, stage: str = "production") -> dict[str, Any]:
        info = self.current(stage)
        if not info:
            raise FileNotFoundError(f"no model registered at stage={stage}")
        bundle_path = Path(info["bundle"])
        if not bundle_path.exists():
            raise FileNotFoundError(f"registered bundle missing: {bundle_path}")
        return joblib.load(bundle_path)
