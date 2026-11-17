import { useCallback, useEffect, useState } from "react";
import { fetchOrders, ordersExportUrl } from "../api/client";
import type { OrderRow, OrdersResponse } from "../api/types";
import { money, unitMoney } from "../lib/format";
import { IconDownload, IconRefresh } from "./Icons";

const DEPT_SHORT: Record<string, string> = {
  FOODS_1: "Produce (FOODS_1)",
  FOODS_2: "Dairy & Deli (FOODS_2)",
  FOODS_3: "Packaged (FOODS_3)",
};

function badgeClass(action: string): string {
  if (action === "Trim Overstock") return "badge badge-trim";
  if (action === "Boost Stock") return "badge badge-boost";
  return "badge badge-hold";
}

export function OrderSheet({ onSelectItem }: { onSelectItem: (itemId: string) => void }) {
  const [data, setData] = useState<OrdersResponse | null>(null);
  const [date, setDate] = useState<string>("");
  const [deptId, setDeptId] = useState<string>("(all)");
  const [status, setStatus] = useState<string>("all");
  const [search, setSearch] = useState<string>("");
  const [limit, setLimit] = useState<number>(50);
  const [error, setError] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(false);

  const load = useCallback(
    async (nextDate: string, dept: string, st: string, q: string, rowLimit: number) => {
      setLoading(true);
      setError("");
      try {
        const res = await fetchOrders({
          date: nextDate || undefined,
          dept_id: dept,
          status: st,
          search: q,
          limit: rowLimit,
        });
        setData(res);
        if (!nextDate && res.selected_date) {
          setDate(res.selected_date);
        }
      } catch (e: unknown) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setLoading(false);
      }
    },
    [],
  );

  useEffect(() => {
    void load("", "(all)", "all", "", 50);
  }, [load]);

  function resetAllFilters() {
    const firstDate = data?.dates[0] ?? "";
    setDate(firstDate);
    setDeptId("(all)");
    setStatus("all");
    setSearch("");
    setLimit(50);
    void load(firstDate, "(all)", "all", "", 50);
  }

  const rows: OrderRow[] = data?.rows ?? [];
  const summary = data?.summary;
  const exportHref = ordersExportUrl({
    date: date || data?.selected_date,
    dept_id: deptId,
    status,
    search,
  });

  return (
    <div className="desk-block">
      <div className="panel-head">
        <div>
          <h3>
            Holdout Order Sheet &amp; PO Replay{" "}
            {data?.selected_date ? `(${data.selected_date})` : ""}
          </h3>
          <p className="hint">
            Replays store replenishment orders against M5 holdout outcomes. Compare Recommended
            Order vs. Clerk Rule alongside realized Actual Sales and Net Profit, or export a PO CSV.
          </p>
        </div>
        <a className="btn btn-primary" href={exportHref} download>
          <IconDownload size={15} />
          <span>Export Purchase Order (CSV)</span>
        </a>
      </div>

      <div className="filters">
        <div className="field">
          <label htmlFor="po-date">Holdout Replay Date</label>
          <select
            id="po-date"
            value={date}
            onChange={(e) => {
              const v = e.target.value;
              setDate(v);
              setLimit(50);
              void load(v, deptId, status, search, 50);
            }}
          >
            {(data?.dates ?? []).map((d) => (
              <option key={d} value={d}>
                {d}
              </option>
            ))}
          </select>
        </div>

        <div className="field">
          <label htmlFor="po-dept">Department</label>
          <select
            id="po-dept"
            value={deptId}
            onChange={(e) => {
              const v = e.target.value;
              setDeptId(v);
              setLimit(50);
              void load(date, v, status, search, 50);
            }}
          >
            <option value="(all)">Whole store</option>
            <option value="FOODS_1">Produce (FOODS_1)</option>
            <option value="FOODS_2">Dairy &amp; Deli (FOODS_2)</option>
            <option value="FOODS_3">Packaged Foods (FOODS_3)</option>
          </select>
        </div>

        <div className="field">
          <label htmlFor="po-status">Adjustment vs. Clerk</label>
          <select
            id="po-status"
            value={status}
            onChange={(e) => {
              const v = e.target.value;
              setStatus(v);
              setLimit(50);
              void load(date, deptId, v, search, 50);
            }}
          >
            <option value="all">All SKUs</option>
            <option value="trim">Trim Overstock (Cut Waste)</option>
            <option value="boost">Boost Stock (Stop Stockouts)</option>
            <option value="hold">Hold (Matches Clerk)</option>
          </select>
        </div>

        <div className="field">
          <label htmlFor="po-search">SKU Search</label>
          <input
            id="po-search"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                setLimit(50);
                void load(date, deptId, status, search, 50);
              }
            }}
            placeholder="e.g. FOODS_3_090"
          />
        </div>

        <div className="field-actions">
          <button
            type="button"
            className="btn btn-secondary"
            onClick={() => {
              setLimit(50);
              void load(date, deptId, status, search, 50);
            }}
            disabled={loading}
          >
            {loading ? "Updating..." : "Apply"}
          </button>
          <button
            type="button"
            className="btn btn-secondary"
            onClick={resetAllFilters}
            disabled={loading}
          >
            <IconRefresh size={14} />
            <span>Reset</span>
          </button>
        </div>
      </div>

      {summary ? (
        <div className="ruled-metrics">
          <div className="ruled-item">
            <span>Planned Order Units</span>
            <strong className="num">
              {summary.total_order_units.toLocaleString()}{" "}
              <small className="muted">(clerk {summary.baseline_order_units.toLocaleString()})</small>
            </strong>
          </div>
          <div className="ruled-item">
            <span>Waste-Cutting Trims</span>
            <strong className="num pos">{summary.trim_count.toLocaleString()} SKUs</strong>
          </div>
          <div className="ruled-item">
            <span>Availability Boosts</span>
            <strong className="num boost">{summary.boost_count.toLocaleString()} SKUs</strong>
          </div>
          <div className="ruled-item">
            <span>Realized Day Net Profit</span>
            <strong className="num">
              {money(summary.estimated_profit)}{" "}
              <small className="muted">(clerk {money(summary.baseline_profit)})</small>
            </strong>
          </div>
        </div>
      ) : null}

      {error ? <div className="banner error">{error}</div> : null}

      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>SKU (Click to Inspect)</th>
              <th>Aisle</th>
              <th className="num">Price</th>
              <th className="num">Expected</th>
              <th className="num">Safety (q75 / q90)</th>
              <th className="num">ML Order</th>
              <th className="num">Clerk</th>
              <th className="num">Delta</th>
              <th>Action</th>
              <th className="num">Actual Sold</th>
              <th className="num">Net Profit</th>
            </tr>
          </thead>
          <tbody>
            {rows.length === 0 ? (
              <tr>
                <td colSpan={11} className="empty-cell">
                  {loading
                    ? "Loading purchase order replay..."
                    : "No SKUs match your current filters. Use Reset to view the full store sheet."}
                </td>
              </tr>
            ) : (
              rows.map((r) => (
                <tr key={r.item_id} className="clickable" onClick={() => onSelectItem(r.item_id)}>
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
                  <td>{DEPT_SHORT[r.dept_id] ?? r.dept_id}</td>
                  <td className="num">{unitMoney(r.sell_price)}</td>
                  <td className="num">{r.pred_demand.toFixed(1)}</td>
                  <td className="num">
                    {r.pred_q75.toFixed(1)} / {r.pred_q90.toFixed(1)}
                  </td>
                  <td className="num">
                    <strong>{r.recommended_order}</strong>
                  </td>
                  <td className="num">{r.baseline_order}</td>
                  <td
                    className={`num ${
                      r.delta_vs_baseline < 0 ? "pos" : r.delta_vs_baseline > 0 ? "boost" : ""
                    }`}
                  >
                    {r.delta_vs_baseline > 0 ? `+${r.delta_vs_baseline}` : r.delta_vs_baseline}
                  </td>
                  <td>
                    <span className={badgeClass(r.action)}>{r.action}</span>
                  </td>
                  <td className="num">{r.sales}</td>
                  <td className="num">{unitMoney(r.net_profit)}</td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
      {data && summary && summary.sku_count > data.returned ? (
        <div className="order-pager">
          <p className="muted">
            Showing {data.returned} of {summary.sku_count} matching SKUs for {data.selected_date}.
          </p>
          <button
            type="button"
            className="btn btn-secondary"
            disabled={loading}
            onClick={() => {
              const nextLimit = limit + 100;
              setLimit(nextLimit);
              void load(date, deptId, status, search, nextLimit);
            }}
          >
            Load 100 more SKUs
          </button>
        </div>
      ) : null}
    </div>
  );
}
