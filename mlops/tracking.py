"""Local experiment tracking (params, metrics, artifacts per run)."""

from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _file_sha256(path: Path, chunk: int = 1 << 20) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


class RunTracker:
    """File-based run store under artifacts/runs/<run_id>/."""

    def __init__(self, artifacts_dir: str | Path, experiment_name: str = "smartsupply"):
        self.artifacts_dir = Path(artifacts_dir)
        self.experiment_name = experiment_name
        self.runs_dir = self.artifacts_dir / "runs"
        self.index_path = self.artifacts_dir / "runs_index.jsonl"
        self.runs_dir.mkdir(parents=True, exist_ok=True)

        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{stamp}_{uuid.uuid4().hex[:6]}"
        self.run_dir = self.runs_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.params: dict[str, Any] = {}
        self.metrics: dict[str, Any] = {}
        self.meta: dict[str, Any] = {
            "run_id": self.run_id,
            "experiment": experiment_name,
            "status": "running",
            "started_at": _utc_now(),
            "ended_at": None,
            "artifacts": [],
            "data_hashes": {},
        }
        self._flush_meta()

    def log_params(self, params: dict[str, Any]) -> None:
        self.params.update(params)
        self._write_json(self.run_dir / "params.json", self.params)

    def log_metrics(self, metrics: dict[str, Any]) -> None:
        #coerce numpy scalars to plain python for json
        clean = {}
        for k, v in metrics.items():
            if hasattr(v, "item"):
                try:
                    v = v.item()
                except Exception:
                    pass
            clean[k] = v
        self.metrics.update(clean)
        self._write_json(self.run_dir / "metrics.json", self.metrics)

    def log_data_hashes(self, paths: dict[str, str | Path]) -> None:
        hashes = {}
        for label, path in paths.items():
            digest = _file_sha256(Path(path))
            if digest:
                hashes[label] = digest
        self.meta["data_hashes"] = hashes
        self._flush_meta()

    def log_artifact(self, src: str | Path, name: str | None = None) -> Path:
        src_path = Path(src)
        if not src_path.exists():
            raise FileNotFoundError(f"artifact missing: {src_path}")
        dest_name = name or src_path.name
        dest = self.run_dir / dest_name
        if src_path.resolve() != dest.resolve():
            shutil.copy2(src_path, dest)
        if dest_name not in self.meta["artifacts"]:
            self.meta["artifacts"].append(dest_name)
            self._flush_meta()
        return dest

    def set_status(self, status: str, error: str | None = None) -> None:
        self.meta["status"] = status
        if error:
            self.meta["error"] = error
        if status in {"completed", "failed", "rejected"}:
            self.meta["ended_at"] = _utc_now()
        self._flush_meta()

    def finish(self, status: str = "completed", error: str | None = None) -> None:
        self.set_status(status, error=error)
        #append one line to the experiment index for easy listing
        record = {
            "run_id": self.run_id,
            "experiment": self.experiment_name,
            "status": self.meta["status"],
            "started_at": self.meta["started_at"],
            "ended_at": self.meta["ended_at"],
            "metrics": self.metrics,
            "run_dir": str(self.run_dir),
        }
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        with self.index_path.open("a") as f:
            f.write(json.dumps(record) + "\n")

    def _flush_meta(self) -> None:
        self._write_json(self.run_dir / "meta.json", self.meta)

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(payload, f, indent=2, default=str)

    @staticmethod
    def list_runs(artifacts_dir: str | Path, limit: int = 20) -> list[dict]:
        index = Path(artifacts_dir) / "runs_index.jsonl"
        if not index.exists():
            return []
        rows = []
        with index.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return list(reversed(rows[-limit:]))

    @staticmethod
    def load_run(artifacts_dir: str | Path, run_id: str) -> dict[str, Any]:
        run_dir = Path(artifacts_dir) / "runs" / run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"run not found: {run_id}")
        out: dict[str, Any] = {"run_dir": str(run_dir)}
        for name in ("meta.json", "params.json", "metrics.json"):
            path = run_dir / name
            if path.exists():
                with path.open() as f:
                    out[name.replace(".json", "")] = json.load(f)
        return out
