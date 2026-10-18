"""
Rebuild holdout results without Optuna retrain.

Loads lightgbm_bundle.pkl, rebuilds features from M5 CSVs, re-runs stocking
(ML newsvendor + stronger historical baseline), and rewrites the forecast CSV.
"""

from __future__ import annotations

from pathlib import Path

import joblib

from grocery import SmartSupplyAI

ROOT = Path(__file__).resolve().parent


def main() -> None:
    bundle_path = ROOT / "lightgbm_bundle.pkl"
    if not bundle_path.exists():
        raise SystemExit("Missing lightgbm_bundle.pkl. Run grocery.py once first.")

    bundle = joblib.load(bundle_path)
    app = SmartSupplyAI(
        sales_path=ROOT / "sales_train_validation.csv",
        calendar_path=ROOT / "calendar.csv",
        prices_path=ROOT / "sell_prices.csv",
        store_id="CA_1",
        cat_id="FOODS",
        lookback_days=365,
        holdout_days=60,
        output_dir=ROOT,
        skip_tuning=True,
    )
    app.load_and_clean()
    app.feature_engineering()

    cutoff = app.df["date"].max() - __import__("pandas").Timedelta(days=app.holdout_days)
    app.features = bundle.get("features") or app._get_feature_list()
    app.model = bundle["point"]
    app.model_q75 = bundle["q75"]
    app.model_q90 = bundle["q90"]
    app.cv_scores = bundle.get("cv_scores") or []
    app.best_params = bundle.get("best_params") or {}
    app.val_df = app.df[app.df["date"] > cutoff].copy()
    if app.val_df.empty:
        raise SystemExit("Empty holdout after feature build.")

    app.run_simulation()
    app.export_results()
    app.print_report()


if __name__ == "__main__":
    main()
