"""AWS cloud storage for Smart Supply (Amazon S3 and DynamoDB)."""

from __future__ import annotations

import json
import os
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd


def aws_region() -> str:
    return (
        os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
        or "us-east-1"
    )


def aws_endpoint_url() -> str | None:
    raw = os.environ.get("AWS_ENDPOINT_URL", "").strip()
    return raw or None


def aws_config_summary() -> dict[str, Any]:
    bucket = os.environ.get("SMARTSUPPLY_S3_BUCKET", "").strip()
    table = os.environ.get("SMARTSUPPLY_DDB_TABLE", "").strip()
    return {
        "region": aws_region(),
        "s3_enabled": bool(bucket),
        "s3_bucket": bucket or None,
        "dynamodb_enabled": bool(table),
        "dynamodb_table": table or None,
    }


def _to_decimal(val: Any) -> Decimal:
    return Decimal(str(round(float(val), 4)))


def _from_decimal(val: Any) -> Any:
    if isinstance(val, Decimal):
        return float(val)
    if isinstance(val, dict):
        return {k: _from_decimal(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_from_decimal(v) for v in val]
    return val


class S3ArtifactStore:
    """Syncs promoted model bundles, metrics, and forecast CSVs with Amazon S3."""

    def __init__(
        self,
        bucket: str,
        *,
        prefix: str = "smartsupply",
        region: str | None = None,
        endpoint_url: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.bucket = bucket.strip()
        self.prefix = prefix.strip("/")
        self.region = region or aws_region()
        self.endpoint_url = endpoint_url if endpoint_url is not None else aws_endpoint_url()
        self._client = client

    def _s3(self) -> Any:
        if self._client is not None:
            return self._client
        import boto3

        self._client = boto3.client(
            "s3",
            region_name=self.region,
            endpoint_url=self.endpoint_url,
        )
        return self._client

    def stage_prefix(self, stage: str = "production") -> str:
        return f"{self.prefix}/registry/{stage}"

    def sync_stage_dir(self, stage_dir: str | Path, stage: str = "production") -> list[str]:
        """Upload every file in a local registry stage folder to S3."""
        folder = Path(stage_dir)
        if not folder.is_dir():
            raise FileNotFoundError(f"stage directory not found: {folder}")
        uploaded: list[str] = []
        s3 = self._s3()
        base_prefix = self.stage_prefix(stage)
        for path in sorted(folder.iterdir()):
            if not path.is_file():
                continue
            key = f"{base_prefix}/{path.name}"
            s3.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=path.read_bytes(),
                ServerSideEncryption="AES256",
            )
            uploaded.append(f"s3://{self.bucket}/{key}")
        return uploaded

    def download_stage(self, dest_dir: str | Path, stage: str = "production") -> list[Path]:
        """Download all objects for a registry stage from S3 into dest_dir."""
        target = Path(dest_dir)
        target.mkdir(parents=True, exist_ok=True)
        s3 = self._s3()
        prefix = f"{self.stage_prefix(stage)}/"
        resp = s3.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
        saved: list[Path] = []
        for item in resp.get("Contents", []):
            key = str(item["Key"])
            name = Path(key).name
            if not name:
                continue
            obj = s3.get_object(Bucket=self.bucket, Key=key)
            body = obj["Body"].read()
            out = target / name
            out.write_bytes(body if isinstance(body, bytes) else str(body).encode("utf-8"))
            saved.append(out)
        return saved

    def read_current_pointer(self, stage: str = "production") -> dict[str, Any] | None:
        key = f"{self.stage_prefix(stage)}/current.json"
        try:
            obj = self._s3().get_object(Bucket=self.bucket, Key=key)
            raw = obj["Body"].read()
            payload = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else str(raw))
            return payload if isinstance(payload, dict) else None
        except Exception:
            return None


class DynamoDBForecastStore:
    """Stores and queries daily SKU stock recommendations in Amazon DynamoDB."""

    def __init__(
        self,
        table_name: str,
        *,
        region: str | None = None,
        endpoint_url: str | None = None,
        table: Any | None = None,
    ) -> None:
        self.table_name = table_name.strip()
        self.region = region or aws_region()
        self.endpoint_url = endpoint_url if endpoint_url is not None else aws_endpoint_url()
        self._table = table

    def _ddb_table(self) -> Any:
        if self._table is not None:
            return self._table
        import boto3

        resource = boto3.resource(
            "dynamodb",
            region_name=self.region,
            endpoint_url=self.endpoint_url,
        )
        self._table = resource.Table(self.table_name)
        return self._table

    @staticmethod
    def partition_key(store_id: str, dept_id: str) -> str:
        return f"STORE#{store_id}#DEPT#{dept_id}"

    @staticmethod
    def sort_key(date_str: str, item_id: str) -> str:
        return f"DATE#{date_str}#ITEM#{item_id}"

    def publish_dataframe(
        self,
        df: pd.DataFrame,
        *,
        store_id: str = "CA_1",
        run_id: str = "local",
        max_rows: int = 5000,
    ) -> int:
        """Batch-write daily order recommendations into DynamoDB."""
        subset = df.head(max_rows)
        table = self._ddb_table()
        written = 0
        with table.batch_writer() as batch:
            for row in subset.itertuples(index=False):
                date_val = str(getattr(row, "date"))[:10]
                dept_id = str(getattr(row, "dept_id", "FOODS_1"))
                item_id = str(getattr(row, "item_id"))
                item = {
                    "pk": self.partition_key(store_id, dept_id),
                    "sk": self.sort_key(date_val, item_id),
                    "store_id": store_id,
                    "dept_id": dept_id,
                    "date": date_val,
                    "item_id": item_id,
                    "run_id": run_id,
                    "pred_demand": _to_decimal(getattr(row, "pred_demand", 0.0)),
                    "stock_decision": _to_decimal(getattr(row, "stock_decision", 0.0)),
                    "net_profit": _to_decimal(getattr(row, "net_profit", 0.0)),
                    "waste_cost": _to_decimal(getattr(row, "waste_cost", 0.0)),
                }
                batch.put_item(Item=item)
                written += 1
        return written

    def query_recommendations(
        self,
        store_id: str,
        dept_id: str,
        *,
        date_prefix: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Query recommendations by partition key and optional date prefix."""
        pk = self.partition_key(store_id, dept_id)
        table = self._ddb_table()

        expr_values: dict[str, Any] = {":pk": pk}
        key_expr = "pk = :pk"
        if date_prefix:
            key_expr += " AND begins_with(sk, :sk_prefix)"
            expr_values[":sk_prefix"] = f"DATE#{date_prefix}"

        resp = table.query(
            KeyConditionExpression=key_expr,
            ExpressionAttributeValues=expr_values,
            Limit=limit,
        )
        return [_from_decimal(item) for item in resp.get("Items", [])]
