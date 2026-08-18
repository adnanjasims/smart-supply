"""Quality gates before a run can be promoted."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GateResult:
    passed: bool
    checks: list[dict[str, Any]] = field(default_factory=list)

    def summary_lines(self) -> list[str]:
        lines = []
        for c in self.checks:
            mark = "PASS" if c["passed"] else "FAIL"
            lines.append(f"  [{mark}] {c['name']}: {c['detail']}")
        return lines


def evaluate_gates(metrics: dict[str, Any], gates: dict[str, Any], drop_stats: dict | None = None) -> GateResult:
    """Compare logged metrics against config thresholds."""
    checks = []
    drop_stats = drop_stats or {}

    def _add(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    if "max_val_mae" in gates and "val_mae" in metrics:
        lim = float(gates["max_val_mae"])
        val = float(metrics["val_mae"])
        _add("max_val_mae", val <= lim, f"val_mae={val:.4f} limit={lim:.4f}")

    if "min_profit_lift" in gates and "profit_lift" in metrics:
        lim = float(gates["min_profit_lift"])
        val = float(metrics["profit_lift"])
        _add("min_profit_lift", val >= lim, f"profit_lift={val:.2f} min={lim:.2f}")

    if "min_q90_ge_q75_frac" in gates and "q90_ge_q75_frac" in metrics:
        lim = float(gates["min_q90_ge_q75_frac"])
        val = float(metrics["q90_ge_q75_frac"])
        _add("min_q90_ge_q75_frac", val >= lim, f"frac={val:.3f} min={lim:.3f}")

    if gates.get("require_no_fallback") and "used_human_fallback" in metrics:
        fb = bool(metrics["used_human_fallback"])
        _add("require_no_fallback", not fb, f"used_human_fallback={fb}")

    if "min_rows_after_features" in gates:
        lim = int(gates["min_rows_after_features"])
        rows = int(drop_stats.get("rows_after", metrics.get("rows_after", 0)))
        _add("min_rows_after_features", rows >= lim, f"rows_after={rows} min={lim}")

    if "cv_mae_mean" in metrics and "max_cv_mae" in gates:
        lim = float(gates["max_cv_mae"])
        val = float(metrics["cv_mae_mean"])
        _add("max_cv_mae", val <= lim, f"cv_mae_mean={val:.4f} limit={lim:.4f}")

    passed = all(c["passed"] for c in checks) if checks else True
    return GateResult(passed=passed, checks=checks)
