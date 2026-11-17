import { useEffect, useRef, useState } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { fetchItemDetail } from "../api/client";
import type { ItemDetailResponse } from "../api/types";
import { unitMoney } from "../lib/format";
import { IconClose } from "./Icons";

export function ItemDrawer({
  itemId,
  onClose,
}: {
  itemId: string | null;
  onClose: () => void;
}) {
  const [detail, setDetail] = useState<ItemDetailResponse | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const drawerRef = useRef<HTMLDivElement>(null);
  const closeBtnRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (!itemId) {
      setDetail(null);
      return;
    }
    const prevFocus = document.activeElement as HTMLElement | null;
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    closeBtnRef.current?.focus();

    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        onClose();
        return;
      }
      if (e.key === "Tab" && drawerRef.current) {
        const focusables = drawerRef.current.querySelectorAll<HTMLElement>(
          'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])',
        );
        if (!focusables.length) return;
        const first = focusables[0];
        const last = focusables[focusables.length - 1];
        if (e.shiftKey && document.activeElement === first) {
          e.preventDefault();
          last.focus();
        } else if (!e.shiftKey && document.activeElement === last) {
          e.preventDefault();
          first.focus();
        }
      }
    };
    window.addEventListener("keydown", onKeyDown);

    let alive = true;
    setLoading(true);
    setError("");
    fetchItemDetail(itemId)
      .then((res) => {
        if (alive) setDetail(res);
      })
      .catch((e: unknown) => {
        if (alive) setError(e instanceof Error ? e.message : String(e));
      })
      .finally(() => {
        if (alive) setLoading(false);
      });

    return () => {
      alive = false;
      window.removeEventListener("keydown", onKeyDown);
      document.body.style.overflow = prevOverflow;
      prevFocus?.focus?.();
    };
  }, [itemId, onClose]);

  if (!itemId) return null;

  return (
    <div
      className="drawer-backdrop"
      role="dialog"
      aria-modal="true"
      aria-label={`SKU detail ${itemId}`}
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="drawer" ref={drawerRef}>
        <div className="drawer-head">
          <div>
            <h2 className="drawer-title">{itemId}</h2>
            {detail ? (
              <p className="muted drawer-sub">
                {detail.dept_id} · Avg price {unitMoney(detail.avg_sell_price)} across{" "}
                {detail.days} holdout days
              </p>
            ) : null}
          </div>
          <button
            ref={closeBtnRef}
            type="button"
            className="btn btn-secondary"
            onClick={onClose}
            aria-label="Close SKU detail drawer"
          >
            <IconClose size={15} />
            <span>Close</span>
          </button>
        </div>

        {loading ? <p className="muted">Loading SKU forecast history...</p> : null}
        {error ? <div className="banner error">{error}</div> : null}

        {detail ? (
          <>
            <div className="ruled-metrics two">
              <div className="ruled-item">
                <span>ML vs. Clerk Profit</span>
                <strong className="num">
                  {unitMoney(detail.total_ml_profit)}{" "}
                  <small className="muted">vs {unitMoney(detail.total_human_profit)}</small>
                </strong>
              </div>
              <div className="ruled-item">
                <span>60-Day Spoilage Cost</span>
                <strong className="num">
                  {unitMoney(detail.total_waste_cost)}{" "}
                  {detail.total_human_waste_cost != null ? (
                    <small className="muted">(clerk {unitMoney(detail.total_human_waste_cost)})</small>
                  ) : null}
                </strong>
              </div>
              <div className="ruled-item">
                <span>Total Units Sold</span>
                <strong className="num">{detail.total_sales.toLocaleString()}</strong>
              </div>
              <div className="ruled-item">
                <span>Stockout Days</span>
                <strong className="num">
                  {detail.stockout_days} / {detail.days}
                </strong>
              </div>
            </div>

            <div className="drawer-chart">
              <h3>Daily Sales vs. Recommended Order</h3>
              <p className="hint">
                Actual store sales compared against SmartSupply order targets and the clerk rule.
              </p>
              <div className="chart-wrap">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={detail.series} margin={{ top: 6, right: 10, left: 0, bottom: 0 }}>
                    <CartesianGrid stroke="#e2e6e3" strokeDasharray="3 3" />
                    <XAxis dataKey="date" tick={{ fontSize: 11, fill: "#4b5550" }} minTickGap={18} />
                    <YAxis tick={{ fontSize: 11, fill: "#4b5550" }} width={48} />
                    <Tooltip />
                    <Legend />
                    <Line
                      type="monotone"
                      dataKey="sales"
                      name="Actual Sales"
                      stroke="#18201b"
                      strokeWidth={2}
                      dot={false}
                      isAnimationActive={false}
                    />
                    <Line
                      type="monotone"
                      dataKey="stock_decision"
                      name="SmartSupply Order"
                      stroke="#166534"
                      strokeWidth={2}
                      dot={false}
                      isAnimationActive={false}
                    />
                    <Line
                      type="monotone"
                      dataKey="human_stock"
                      name="Clerk Order"
                      stroke="#b42318"
                      strokeDasharray="4 3"
                      dot={false}
                      isAnimationActive={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          </>
        ) : null}
      </div>
    </div>
  );
}
