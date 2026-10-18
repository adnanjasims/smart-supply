import type { DeptRow } from "../api/types";
import { money, pct } from "../lib/format";

export function DeptTable({ rows }: { rows: DeptRow[] }) {
  if (!rows.length) {
    return <p className="muted">No department column in the export.</p>;
  }

  return (
    <div className="table-wrap">
      <table className="data">
        <thead>
          <tr>
            <th>Dept</th>
            <th>ML profit</th>
            <th>Baseline profit</th>
            <th>Lift</th>
            <th>Stockout $</th>
            <th>Stockout %</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.dept_id}>
              <td>{r.dept_id}</td>
              <td>{money(r.ml_profit)}</td>
              <td>{money(r.human_profit)}</td>
              <td>{money(r.profit_lift)}</td>
              <td>{money(r.stockout_cost)}</td>
              <td>{pct(r.stockout_rate_pct)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
