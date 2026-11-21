"""Unit tests for Smart Supply AWS S3 model registry and DynamoDB forecast store."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi.testclient import TestClient

from mlops.aws_store import DynamoDBForecastStore, S3ArtifactStore
from mlops.registry import ModelRegistry
from mlops.tracking import RunTracker
from server.main import app


class _FakeS3:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}
        self.put_calls: list[dict[str, Any]] = []

    def put_object(self, **kwargs: Any) -> dict[str, Any]:
        self.put_calls.append(kwargs)
        body = kwargs["Body"]
        raw = body if isinstance(body, bytes) else str(body).encode("utf-8")
        self.objects[(str(kwargs["Bucket"]), str(kwargs["Key"]))] = raw
        return {}

    def list_objects_v2(self, **kwargs: Any) -> dict[str, Any]:
        bucket = str(kwargs["Bucket"])
        prefix = str(kwargs.get("Prefix", ""))
        contents = [
            {"Key": k}
            for (b, k) in sorted(self.objects.keys())
            if b == bucket and k.startswith(prefix)
        ]
        return {"Contents": contents}

    def get_object(self, **kwargs: Any) -> dict[str, Any]:
        data = self.objects[(str(kwargs["Bucket"]), str(kwargs["Key"]))]
        return {"Body": io.BytesIO(data)}


class _FakeBatchWriter:
    def __init__(self, target: list[dict[str, Any]]) -> None:
        self._target = target

    def __enter__(self) -> "_FakeBatchWriter":
        return self

    def __exit__(self, *_args: Any) -> None:
        return None

    def put_item(self, Item: dict[str, Any]) -> None:
        self._target.append(Item)


class _FakeDynamoTable:
    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    def batch_writer(self) -> _FakeBatchWriter:
        return _FakeBatchWriter(self.items)

    def query(self, **kwargs: Any) -> dict[str, Any]:
        values = kwargs["ExpressionAttributeValues"]
        pk = values[":pk"]
        sk_prefix = values.get(":sk_prefix", "")
        limit = int(kwargs.get("Limit", 50))
        matched = [
            item
            for item in self.items
            if item["pk"] == pk and (not sk_prefix or item["sk"].startswith(sk_prefix))
        ]
        return {"Items": matched[:limit]}


def test_registry_promote_syncs_and_pulls_from_s3(tmp_path: Path) -> None:
    fake_s3 = _FakeS3()
    s3_store = S3ArtifactStore("smartsupply-test-bucket", client=fake_s3)

    artifacts = tmp_path / "artifacts"
    tracker = RunTracker(artifacts, experiment_name="aws-test")
    joblib.dump({"point": "demo-model", "features": ["lag_7"]}, tracker.run_dir / "lightgbm_bundle.pkl")
    tracker.log_metrics({"val_mae": 1.05, "profit_lift": 420.0})
    tracker.finish("completed")

    reg = ModelRegistry(artifacts, s3_store=s3_store)
    reg.promote(tracker.run_id)

    keys = [k for (_, k) in fake_s3.objects]
    assert "smartsupply/registry/production/lightgbm_bundle.pkl" in keys
    assert "smartsupply/registry/production/current.json" in keys
    assert all(c["ServerSideEncryption"] == "AES256" for c in fake_s3.put_calls)

    pointer = s3_store.read_current_pointer("production")
    assert pointer is not None and pointer["run_id"] == tracker.run_id

    fresh_artifacts = tmp_path / "fresh_artifacts"
    fresh_reg = ModelRegistry(fresh_artifacts, s3_store=s3_store)
    saved = fresh_reg.pull_from_s3("production")
    assert any(p.name == "lightgbm_bundle.pkl" for p in saved)
    loaded = fresh_reg.load_bundle("production")
    assert loaded["point"] == "demo-model"


def test_dynamodb_publish_and_query_recommendations() -> None:
    fake_table = _FakeDynamoTable()
    store = DynamoDBForecastStore("SmartSupplyForecasts-dev", table=fake_table)

    df = pd.DataFrame(
        [
            {
                "date": "2016-05-01",
                "dept_id": "FOODS_1",
                "item_id": "FOODS_1_001",
                "pred_demand": 14.2,
                "stock_decision": 18.0,
                "net_profit": 24.5,
                "waste_cost": 1.2,
            },
            {
                "date": "2016-05-02",
                "dept_id": "FOODS_1",
                "item_id": "FOODS_1_002",
                "pred_demand": 7.5,
                "stock_decision": 10.0,
                "net_profit": 12.0,
                "waste_cost": 0.4,
            },
            {
                "date": "2016-05-01",
                "dept_id": "FOODS_2",
                "item_id": "FOODS_2_001",
                "pred_demand": 9.0,
                "stock_decision": 12.0,
                "net_profit": 15.0,
                "waste_cost": 0.0,
            },
        ]
    )

    written = store.publish_dataframe(df, store_id="CA_1", run_id="run-123")
    assert written == 3

    may1 = store.query_recommendations("CA_1", "FOODS_1", date_prefix="2016-05-01")
    assert len(may1) == 1
    assert may1[0]["item_id"] == "FOODS_1_001"
    assert may1[0]["stock_decision"] == 18.0


def test_cloud_status_endpoint(monkeypatch) -> None:
    monkeypatch.setenv("SMARTSUPPLY_S3_BUCKET", "my-models")
    monkeypatch.delenv("SMARTSUPPLY_DDB_TABLE", raising=False)
    client = TestClient(app)
    resp = client.get("/api/cloud/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["s3_enabled"] is True
    assert body["s3_bucket"] == "my-models"
    assert body["dynamodb_enabled"] is False
