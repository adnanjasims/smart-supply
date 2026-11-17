import pandas as pd
import pytest
from fastapi.testclient import TestClient

from grocery import SmartSupplyAI
from server.main import app


def _sample_sim_df() -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=5, freq="D")
    rows = []
    for d in dates:
        rows.append(
            {
                "date": d,
                "item_id": "FOODS_1_001",
                "dept_id": "FOODS_1",
                "sell_price": 4.0,
                "sales": 3,
                "pred_demand": 3.1,
                "pred_q50": 2.6,
                "pred_q62": 3.1,
                "pred_q75": 3.8,
                "pred_q83": 4.4,
                "pred_q90": 5.0,
                "pred_q95": 5.8,
                "stock_decision": 3,
                "human_stock": 5,
                "waste": 0.0,
                "human_waste": 0.4,
                "missed_sales": 0,
                "waste_cost": 0.0,
                "stockout_cost": 0.0,
                "human_waste_cost": 1.12,
                "human_stockout_cost": 0.0,
                "human_missed": 0,
                "net_profit": 3.6,
                "sma_profit": 2.48,
                "oracle_profit": 3.6,
            }
        )
        rows.append(
            {
                "date": d,
                "item_id": "FOODS_3_009",
                "dept_id": "FOODS_3",
                "sell_price": 5.0,
                "sales": 6,
                "pred_demand": 5.2,
                "pred_q50": 4.8,
                "pred_q62": 5.5,
                "pred_q75": 6.2,
                "pred_q83": 6.8,
                "pred_q90": 7.5,
                "pred_q95": 8.4,
                "stock_decision": 6,
                "human_stock": 4,
                "waste": 0.0,
                "human_waste": 0.0,
                "missed_sales": 0,
                "waste_cost": 0.0,
                "stockout_cost": 0.0,
                "human_waste_cost": 0.0,
                "human_stockout_cost": 3.75,
                "human_missed": 2,
                "net_profit": 15.0,
                "sma_profit": 6.25,
                "oracle_profit": 15.0,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture()
def api_client(monkeypatch):
    sim_df = _sample_sim_df()
    monkeypatch.setattr("server.main.load_sim", lambda: sim_df)
    monkeypatch.setattr("server.data.load_sim", lambda: sim_df)
    return TestClient(app)


def test_orders_filters_and_csv_export(api_client: TestClient):
    res = api_client.get("/api/orders?status=trim")
    assert res.status_code == 200
    body = res.json()
    assert body["selected_date"] == "2026-01-05"
    assert len(body["rows"]) == 1
    assert body["rows"][0]["item_id"] == "FOODS_1_001"
    assert body["rows"][0]["action"] == "Trim Overstock"
    assert body["summary"]["trim_count"] == 1

    csv_res = api_client.get("/api/orders/export?dept_id=FOODS_3")
    assert csv_res.status_code == 200
    assert "text/csv" in csv_res.headers["content-type"]
    lines = csv_res.text.strip().splitlines()
    assert len(lines) == 2
    assert "FOODS_3_009" in lines[1]


def test_simulate_endpoint_scales_spoilage_and_stockout(api_client: TestClient):
    base = api_client.post(
        "/api/simulate",
        json={"spoilage_multiplier": 1.0, "stockout_penalty": 0.75, "dept_id": "(all)"},
    )
    assert base.status_code == 200
    b = base.json()
    assert b["profit_lift"] > 0
    assert b["waste_saved"] > 0

    harsher = api_client.post(
        "/api/simulate",
        json={"spoilage_multiplier": 2.0, "stockout_penalty": 0.75, "dept_id": "(all)"},
    )
    assert harsher.status_code == 200
    h = harsher.json()
    assert h["human_waste_cost"] > b["human_waste_cost"]
    assert h["waste_saved"] > b["waste_saved"]


def test_item_detail_returns_series_and_404_for_unknown(api_client: TestClient):
    ok = api_client.get("/api/item/FOODS_1_001")
    assert ok.status_code == 200
    body = ok.json()
    assert body["days"] == 5
    assert body["total_sales"] == 15
    assert len(body["series"]) == 5

    missing = api_client.get("/api/item/NO_SUCH_SKU")
    assert missing.status_code == 404


def test_pipeline_endpoint_returns_real_structure(api_client: TestClient, monkeypatch):
    monkeypatch.setattr(
        "server.main.load_bundle",
        lambda: {
            "features": ["sales_lag_1", "zero_rate_28", "sell_price", "day_of_week"],
            "cv_scores": [1.3, 1.2],
            "best_params": {"learning_rate": 0.05, "num_leaves": 64, "min_data_in_leaf": 20},
        },
    )
    res = api_client.get("/api/pipeline")
    assert res.status_code == 200
    body = res.json()
    assert body["skus"] == 2
    assert body["feature_count"] == 4
    assert len(body["departments"]) == 2
    assert body["cv_scores"] == [1.3, 1.2]
    assert body["quantile_ladder"] == ["q50", "q62", "q75", "q83", "q90", "q95"]


def test_full_quantile_ladder_reduces_perishable_stock_vs_shelf_stable():
    app_inst = SmartSupplyAI("a.csv", "b.csv", "c.csv")
    df = pd.DataFrame(
        {
            "dept_id": ["FOODS_1", "FOODS_2", "FOODS_3"],
            "pred_demand": [2.0, 2.0, 2.0],
            "pred_q50": [1.5, 1.5, 1.5],
            "pred_q62": [2.0, 2.0, 2.0],
            "pred_q75": [3.0, 3.0, 3.0],
            "pred_q83": [3.8, 3.8, 3.8],
            "pred_q90": [4.6, 4.6, 4.6],
            "pred_q95": [5.5, 5.5, 5.5],
        }
    )
    stock = app_inst._dept_aware_stock(df)
    assert stock[0] <= stock[1] <= stock[2]
    assert stock[0] == 2
    assert stock[2] >= 5
