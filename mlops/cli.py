"""CLI: python -m mlops <train|list|show|promote|current|sync-s3|pull-s3|publish-ddb>"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

from mlops.aws_store import DynamoDBForecastStore, S3ArtifactStore
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

    sync_s3 = sub.add_parser("sync-s3", help="upload a registry stage to Amazon S3")
    sync_s3.add_argument("--artifacts-dir", default="artifacts")
    sync_s3.add_argument("--stage", default="production")
    sync_s3.add_argument("--bucket", default="")

    pull_s3 = sub.add_parser("pull-s3", help="download a registry stage from Amazon S3")
    pull_s3.add_argument("--artifacts-dir", default="artifacts")
    pull_s3.add_argument("--stage", default="production")
    pull_s3.add_argument("--bucket", default="")

    pub_ddb = sub.add_parser(
        "publish-ddb", help="publish forecast CSV rows to Amazon DynamoDB"
    )
    pub_ddb.add_argument("--csv", default="m5_supply_chain_forecast.csv")
    pub_ddb.add_argument("--store", default="CA_1")
    pub_ddb.add_argument("--table", default="")
    pub_ddb.add_argument("--max-rows", type=int, default=5000)
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

    if args.command == "sync-s3":
        bucket = args.bucket or os.environ.get("SMARTSUPPLY_S3_BUCKET", "").strip()
        if not bucket:
            print("set --bucket or SMARTSUPPLY_S3_BUCKET")
            return 1
        reg = ModelRegistry(args.artifacts_dir, s3_store=S3ArtifactStore(bucket))
        uploaded = reg.sync_to_s3(stage=args.stage)
        print(json.dumps({"bucket": bucket, "uploaded": uploaded}, indent=2))
        return 0

    if args.command == "pull-s3":
        bucket = args.bucket or os.environ.get("SMARTSUPPLY_S3_BUCKET", "").strip()
        if not bucket:
            print("set --bucket or SMARTSUPPLY_S3_BUCKET")
            return 1
        reg = ModelRegistry(args.artifacts_dir, s3_store=S3ArtifactStore(bucket))
        saved = reg.pull_from_s3(stage=args.stage)
        print(json.dumps({"bucket": bucket, "saved": [str(p) for p in saved]}, indent=2))
        return 0

    if args.command == "publish-ddb":
        table = args.table or os.environ.get("SMARTSUPPLY_DDB_TABLE", "").strip()
        if not table:
            print("set --table or SMARTSUPPLY_DDB_TABLE")
            return 1
        df = pd.read_csv(args.csv)
        written = DynamoDBForecastStore(table).publish_dataframe(
            df, store_id=args.store, max_rows=args.max_rows
        )
        print(json.dumps({"table": table, "written": written}, indent=2))
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
