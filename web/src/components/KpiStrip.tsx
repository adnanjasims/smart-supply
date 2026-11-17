import type { Summary } from "../api/types";
import { money, pct } from "../lib/format";

export function KpiStrip({ s }: { s: Summary }) {
  const capturePct = s.oracle_profit > 0 ? (s.ml_profit / s.oracle_profit) * 100 : 0;
  const wasteSaved = s.waste_saved ?? 0;
  const wastePctCut =
    s.human_waste_cost && s.human_waste_cost > 0
      ? (wasteSaved / s.human_waste_cost) * 100
      : 0;

  return (
    <section className="kpi-ledger" aria-label="Executive metrics">
      <div className="kpi-cell">
        <div className="kpi-label">Extra net profit vs. clerk rule</div>
        <div className="kpi-val pos">+{money(s.profit_lift)}</div>
        <div className="kpi-sub">
          {money(s.ml_profit)} ML vs {money(s.human_profit)} clerk trend (EWMA)
        </div>
      </div>

      <div className="kpi-cell">
        <div className="kpi-label">Food spoilage cost saved</div>
        <div className="kpi-val pos">{money(wasteSaved)}</div>
        <div className="kpi-sub">
          {money(s.ml_waste_cost)} ML vs {money(s.human_waste_cost ?? 0)} clerk (
          {pct(wastePctCut)} less shrink)
        </div>
      </div>

      <div className="kpi-cell">
        <div className="kpi-label">Wasted units prevented</div>
        <div className="kpi-val">
          {s.waste_units_saved != null ? `${Math.round(s.waste_units_saved).toLocaleString()} units` : "—"}
        </div>
        <div className="kpi-sub">
          {s.ml_waste_units != null
            ? `${Math.round(s.ml_waste_units).toLocaleString()} vs ${Math.round(
                s.human_waste_units ?? 0,
              ).toLocaleString()} spoiled`
            : "Perishable shrink cut"}
        </div>
      </div>

      <div className="kpi-cell">
        <div className="kpi-label">Missed-sale stockout cost</div>
        <div className="kpi-val">{money(s.ml_stockout_cost)}</div>
        <div className="kpi-sub">Empty shelf on {pct(s.ml_stockout_rate_pct)} of SKU-days</div>
      </div>

      <div className="kpi-cell">
        <div className="kpi-label">Max theoretical capture</div>
        <div className="kpi-val">{pct(capturePct)}</div>
        <div className="kpi-sub">Of {money(s.oracle_profit)} perfect-foresight ceiling</div>
      </div>

      <div className="kpi-cell">
        <div className="kpi-label">Store footprint</div>
        <div className="kpi-val">{s.items.toLocaleString()} SKUs</div>
        <div className="kpi-sub">
          {s.date_start} → {s.date_end}
        </div>
      </div>
    </section>
  );
}
