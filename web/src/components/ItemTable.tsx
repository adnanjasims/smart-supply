import { useEffect, useState } from "react";
import { fetchItems } from "../api/client";
import type { ItemRow } from "../api/types";
import { unitMoney } from "../lib/format";
import { IconRefresh } from "./Icons";

const DEPT_SHORT: Record<string, string> = {
  FOODS_1: "Produce (FOODS_1)",
  FOODS_2: "Dairy & Deli (FOODS_2)",
  FOODS_3: "Packaged (FOODS_3)",
};

export function ItemTable({
  dateStart,
  dateEnd,
  onSelectItem,
}: {
  dateStart: string;
  dateEnd: string;
  onSelectItem: (itemId: string) => void;
}) {
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
    if (start && end && start > end) {
      setError("Start date cannot be later than End date.");
      setRows([]);
      return;
    }
    let cancelled = false;
    setLoading(true);
    setError("");
    fetchItems({ item_id: itemId.trim() || undefined, start, end, limit: 200 })
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
    <div className="desk-block">
      <div className="panel-head">
        <div>
          <h3>Historical SKU Row Explorer</h3>
          <p className="hint">
            Search any item ID across holdout dates, or click any SKU to inspect its 60-day
            demand trajectory.
          </p>
        </div>
      </div>

      <div className="filters">
        <div className="field">
          <label htmlFor="item-filter">SKU Filter</label>
          <input
            id="item-filter"
            value={itemId}
            onChange={(e) => setItemId(e.target.value)}
            placeholder="e.g. FOODS_1_001"
          />
        </div>
        <div className="field">
          <label htmlFor="start">Start Date</label>
          <input id="start" type="date" value={start} onChange={(e) => setStart(e.target.value)} />
        </div>
        <div className="field">
          <label htmlFor="end">End Date</label>
          <input id="end" type="date" value={end} onChange={(e) => setEnd(e.target.value)} />
        </div>
        <div className="field-actions">
          <button
            type="button"
            className="btn btn-secondary"
            onClick={() => {
              setItemId("");
              setStart(dateStart);
              setEnd(dateEnd);
            }}
          >
            <IconRefresh size={14} />
            <span>Reset</span>
          </button>
        </div>
      </div>

      {error ? <div className="banner error">{error}</div> : null}

      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Date</th>
              <th>SKU</th>
              <th>Aisle</th>
              <th className="num">Sold</th>
              <th className="num">Expected</th>
              <th className="num">ML Order</th>
              <th className="num">Missed</th>
              <th className="num">ML Profit</th>
              <th className="num">Clerk Profit</th>
            </tr>
          </thead>
          <tbody>
            {rows.length === 0 ? (
              <tr>
                <td colSpan={9} className="empty-cell">
                  {loading
                    ? "Loading historical SKU records..."
                    : error
                      ? "Adjust the date range above to load SKU rows."
                      : "No SKU records matched your filter."}
                </td>
              </tr>
            ) : (
              rows.map((r, i) => (
                <tr
                  key={`${r.date}-${r.item_id}-${i}`}
                  className="clickable"
                  onClick={() => onSelectItem(r.item_id)}
                >
                  <td>{r.date}</td>
                  <td>
                    <button
                      type="button"
                      className="sku-link"
                      onClick={(e) => {
                        e.stopPropagation();
                        onSelectItem(r.item_id);
                      }}
                    >
                      {r.item_id}
                    </button>
                  </td>
                  <td>{DEPT_SHORT[r.dept_id ?? ""] ?? r.dept_id}</td>
                  <td className="num">{r.sales}</td>
                  <td className="num">{r.pred_demand.toFixed(2)}</td>
                  <td className="num">{r.stock_decision}</td>
                  <td className="num">{r.missed_sales}</td>
                  <td className="num">{unitMoney(r.net_profit)}</td>
                  <td className="num">{unitMoney(r.sma_profit)}</td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
