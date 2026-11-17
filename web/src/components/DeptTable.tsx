import type { DeptRow } from "../api/types";
import { money, pct } from "../lib/format";

const DEPT_LABELS: Record<string, string> = {
  FOODS_1: "Produce & Fresh (20% spoilage)",
  FOODS_2: "Dairy & Deli (10% spoilage)",
  FOODS_3: "Packaged Foods (1% spoilage)",
};

export function DeptTable({ rows }: { rows: DeptRow[] }) {
  return (
    <div className="desk-block">
      <div className="panel-head">
        <div>
          <h3>Department Profit &amp; Spoilage Breakdown</h3>
          <p className="hint">
            Fresh produce and dairy prioritize spoilage control, while shelf-stable packaged goods
            stock deeper to prevent lost high-margin sales.
          </p>
        </div>
      </div>

      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Department</th>
              <th className="num">ML Profit</th>
              <th className="num">Clerk Rule (EWMA)</th>
              <th className="num">Profit Lift</th>
              <th className="num">ML Spoilage</th>
              <th className="num">Spoilage Saved</th>
              <th className="num">Stockout Cost</th>
              <th className="num">Stockout Rate</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => {
              const wasteSaved = r.waste_saved ?? 0;
              return (
                <tr key={r.dept_id}>
                  <td>
                    <strong>{r.dept_id}</strong>{" "}
                    <span className="muted">({DEPT_LABELS[r.dept_id] ?? "Grocery"})</span>
                  </td>
                  <td className="num">{money(r.ml_profit)}</td>
                  <td className="num">{money(r.human_profit)}</td>
                  <td className={`num ${r.profit_lift >= 0 ? "pos" : "neg"}`}>
                    {r.profit_lift >= 0 ? "+" : ""}
                    {money(r.profit_lift)}
                  </td>
                  <td className="num">{money(r.waste_cost)}</td>
                  <td className={`num ${wasteSaved >= 0 ? "pos" : "neg"}`}>
                    {r.waste_saved != null ? `${wasteSaved >= 0 ? "+" : ""}${money(wasteSaved)}` : "—"}
                  </td>
                  <td className="num">{money(r.stockout_cost)}</td>
                  <td className="num">{pct(r.stockout_rate_pct)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
