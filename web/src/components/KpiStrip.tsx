import type { Summary } from "../api/types";
import { money, pct } from "../lib/format";

export function KpiStrip({ summary }: { summary: Summary }) {
  const cells = [
    { label: "Baseline profit", value: money(summary.human_profit) },
    { label: "ML profit", value: money(summary.ml_profit) },
    { label: "Profit lift", value: money(summary.profit_lift), tone: " lift" },
    { label: "Perfect-foresight profit", value: money(summary.oracle_profit) },
    { label: "ML stockout rate", value: pct(summary.ml_stockout_rate_pct) },
    { label: "ML stockout cost", value: money(summary.ml_stockout_cost) },
  ];

  return (
    <div className="kpi-grid kpi-grid-six">
      {cells.map((c) => (
        <div className="kpi" key={c.label}>
          <span className="kpi-label">{c.label}</span>
          <span className={`kpi-value${c.tone || ""}`}>{c.value}</span>
        </div>
      ))}
    </div>
  );
}
