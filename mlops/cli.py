"""CLI: python -m mlops <train|list|show|promote|current>"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mlops.pipeline import run_training_pipeline
from mlops.registry import ModelRegistry
from mlops.tracking import RunTracker


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m mlops", description="SmartSupply MLOps CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="run tracked training pipeline")
    train.add_argument("--config", default="config/default.yaml")
    train.add_argument("--sales")
    train.add_argument("--calendar")
    train.add_argument("--prices")
    train.add_argument("--store")
    train.add_argument("--cat")
    train.add_argument("--lookback-days", type=int)
    train.add_argument("--holdout-days", type=int)
    train.add_argument("--skip-tuning", action="store_true")
    train.add_argument("--promote", action="store_true", help="promote if gates pass")
    train.add_argument("--artifacts-dir")

    ls = sub.add_parser("list", help="list recent runs")
    ls.add_argument("--artifacts-dir", default="artifacts")
    ls.add_argument("--limit", type=int, default=10)

    show = sub.add_parser("show", help="show one run")
    show.add_argument("run_id")
    show.add_argument("--artifacts-dir", default="artifacts")

    promote = sub.add_parser("promote", help="promote a completed run to production")
    promote.add_argument("run_id")
    promote.add_argument("--artifacts-dir", default="artifacts")

    current = sub.add_parser("current", help="show production pointer")
    current.add_argument("--artifacts-dir", default="artifacts")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        overrides: dict = {}
        data_over = {}
        train_over = {}
        mlops_over = {}
        if args.sales:
            data_over["sales"] = args.sales
        if args.calendar:
            data_over["calendar"] = args.calendar
        if args.prices:
            data_over["prices"] = args.prices
        if args.store:
            data_over["store_id"] = args.store
        if args.cat:
            data_over["cat_id"] = args.cat
        if args.lookback_days is not None:
            data_over["lookback_days"] = args.lookback_days
        if args.holdout_days is not None:
            data_over["holdout_days"] = args.holdout_days
        if args.skip_tuning:
            train_over["skip_tuning"] = True
        if args.artifacts_dir:
            mlops_over["artifacts_dir"] = args.artifacts_dir
        if data_over:
            overrides["data"] = data_over
        if train_over:
            overrides["train"] = train_over
        if mlops_over:
            overrides["mlops"] = mlops_over

        #relax row gate when user clearly runs a tiny custom job
        if args.lookback_days is not None and args.lookback_days < 200:
            overrides.setdefault("gates", {})["min_rows_after_features"] = 50

        result = run_training_pipeline(
            config_path=args.config,
            overrides=overrides or None,
            promote=True if args.promote else None,
        )
        print(json.dumps({
            "run_id": result["run_id"],
            "status": result["status"],
            "promoted": result.get("promoted", False),
            "val_mae": result["metrics"].get("val_mae"),
            "profit_lift": result["metrics"].get("profit_lift"),
        }, indent=2))
        return 0 if result["status"] == "completed" else 2

    if args.command == "list":
        rows = RunTracker.list_runs(args.artifacts_dir, limit=args.limit)
        if not rows:
            print("no runs yet. try: python -m mlops train")
            return 0
        for row in rows:
            m = row.get("metrics") or {}
            print(
                f"{row['run_id']}  status={row.get('status')}  "
                f"mae={m.get('val_mae', 'n/a')}  lift={m.get('profit_lift', 'n/a')}"
            )
        return 0

    if args.command == "show":
        info = RunTracker.load_run(args.artifacts_dir, args.run_id)
        print(json.dumps(info, indent=2, default=str))
        return 0

    if args.command == "promote":
        stage_dir = ModelRegistry(args.artifacts_dir).promote(args.run_id)
        print(f"promoted {args.run_id} -> {stage_dir}")
        return 0

    if args.command == "current":
        info = ModelRegistry(args.artifacts_dir).current()
        if not info:
            print("no production model registered")
            return 1
        print(json.dumps(info, indent=2))
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
