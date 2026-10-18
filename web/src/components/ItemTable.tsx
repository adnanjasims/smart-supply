import { useEffect, useState } from "react";
import { fetchItems } from "../api/client";
import type { ItemRow } from "../api/types";
import { money } from "../lib/format";

export function ItemTable({ dateStart, dateEnd }: { dateStart: string; dateEnd: string }) {
  const [itemId, setItemId] = useState("");
  const [start, setStart] = useState(dateStart);
  const [end, setEnd] = useState(dateEnd);
  const [rows, setRows] = useState<ItemRow[]>([]);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setStart(dateStart);
    setEnd(dateEnd);
  }, [dateStart, dateEnd]);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError("");
    fetchItems({ item_id: itemId.trim() || undefined, start, end, limit: 300 })
      .then((res) => {
        if (!cancelled) setRows(res.rows);
      })
      .catch((e) => {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [itemId, start, end]);

  return (
    <>
      <div className="filters">
        <div className="field">
          <label htmlFor="item-filter">Item id</label>
          <input
            id="item-filter"
            value={itemId}
            onChange={(e) => setItemId(e.target.value)}
            placeholder="FOODS_1_001"
          />
        </div>
        <div className="field">
          <label htmlFor="start">Start</label>
          <input id="start" type="date" value={start} onChange={(e) => setStart(e.target.value)} />
        </div>
        <div className="field">
          <label htmlFor="end">End</label>
          <input id="end" type="date" value={end} onChange={(e) => setEnd(e.target.value)} />
        </div>
      </div>
      {error ? <p className="banner error">{error}</p> : null}
      {loading ? <p className="muted">Loading rows...</p> : null}
      <div className="table-wrap">
        <table className="data">
          <thead>
            <tr>
              <th>Date</th>
              <th>Item</th>
              <th>Dept</th>
              <th>Sales</th>
              <th>Pred</th>
              <th>Stock</th>
              <th>Missed</th>
              <th>ML $</th>
              <th>Baseline $</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={`${r.date}-${r.item_id}-${i}`}>
                <td>{r.date}</td>
                <td>{r.item_id}</td>
                <td>{r.dept_id}</td>
                <td>{r.sales}</td>
                <td>{r.pred_demand}</td>
                <td>{r.stock_decision}</td>
                <td>{r.missed_sales}</td>
                <td>{money(r.net_profit)}</td>
                <td>{money(r.sma_profit)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}
