import { useEffect, useState } from "react";
import { postSimulate } from "../api/client";
import type { SimulateResponse } from "../api/types";
import { money } from "../lib/format";

type PresetId = "default" | "summer" | "weekend" | null;

export function SimulatorPanel({ totalRows }: { totalRows: number }) {
  const [spoilMult, setSpoilMult] = useState<number>(1.0);
  const [stockoutPen, setStockoutPen] = useState<number>(0.75);
  const [deptId, setDeptId] = useState<string>("(all)");
  const [activePreset, setActivePreset] = useState<PresetId>("default");
  const [res, setRes] = useState<SimulateResponse | null>(null);
  const [busy, setBusy] = useState<boolean>(false);
  const [error, setError] = useState<string>("");

  useEffect(() => {
    let alive = true;
    setBusy(true);
    const handle = window.setTimeout(() => {
      postSimulate({
        spoilage_multiplier: spoilMult,
        stockout_penalty: stockoutPen,
        dept_id: deptId,
      })
        .then((d) => {
          if (alive) {
            setRes(d);
            setError("");
          }
        })
        .catch((e: unknown) => {
          if (alive) setError(e instanceof Error ? e.message : String(e));
        })
        .finally(() => {
          if (alive) setBusy(false);
        });
    }, 120);
    return () => {
      alive = false;
      window.clearTimeout(handle);
    };
  }, [spoilMult, stockoutPen, deptId]);

  return (
    <div className="desk-block" aria-busy={busy}>
      <div className="panel-head">
        <div>
          <h3>What-If Spoilage &amp; Stockout Stress Simulator</h3>
          <p className="hint">
            Re-score all {totalRows.toLocaleString()} holdout SKU-days under changing refrigeration
            spoilage risk or customer goodwill stockout penalties.
          </p>
        </div>
        {busy ? <span className="muted">Recalculating...</span> : null}
      </div>

      <div className="pill-row" role="group" aria-label="Simulation presets">
        <button
          type="button"
          className={`chip ${activePreset === "default" ? "active" : ""}`}
          onClick={() => {
            setSpoilMult(1.0);
            setStockoutPen(0.75);
            setDeptId("(all)");
            setActivePreset("default");
          }}
        >
          Standard Store Defaults
        </button>
        <button
          type="button"
          className={`chip ${activePreset === "summer" ? "active" : ""}`}
          onClick={() => {
            setSpoilMult(1.8);
            setStockoutPen(0.75);
            setActivePreset("summer");
          }}
        >
          Summer Spoilage Spike (1.8x)
        </button>
        <button
          type="button"
          className={`chip ${activePreset === "weekend" ? "active" : ""}`}
          onClick={() => {
            setSpoilMult(1.0);
            setStockoutPen(1.4);
            setActivePreset("weekend");
          }}
        >
          Peak Weekend Stockout Risk
        </button>
      </div>

      <div className="sim-band">
        <div className="sim-control">
          <div className="slider-head">
            <label htmlFor="sim-spoil">Spoilage Intensity</label>
            <output htmlFor="sim-spoil">{spoilMult.toFixed(2)}x</output>
          </div>
          <input
            id="sim-spoil"
            type="range"
            min={0.25}
            max={2.5}
            step={0.05}
            value={spoilMult}
            onChange={(e) => {
              setSpoilMult(Number(e.target.value));
              setActivePreset(null);
            }}
          />
          <p className="muted sim-note">
            Scales baseline daily perishability across Produce (20%), Dairy (10%), Packaged (1%).
          </p>
        </div>

        <div className="sim-control">
          <div className="slider-head">
            <label htmlFor="sim-so">Stockout Goodwill Penalty</label>
            <output htmlFor="sim-so">{(stockoutPen * 100).toFixed(0)}% margin</output>
          </div>
          <input
            id="sim-so"
            type="range"
            min={0}
            max={2.0}
            step={0.05}
            value={stockoutPen}
            onChange={(e) => {
              setStockoutPen(Number(e.target.value));
              setActivePreset(null);
            }}
          />
          <p className="muted sim-note">
            Extra customer churn cost per missed unit beyond lost immediate margin.
          </p>
        </div>

        <div className="field">
          <label htmlFor="sim-dept">Department Scope</label>
          <select id="sim-dept" value={deptId} onChange={(e) => setDeptId(e.target.value)}>
            <option value="(all)">Whole store</option>
            <option value="FOODS_1">Produce (FOODS_1)</option>
            <option value="FOODS_2">Dairy &amp; Deli (FOODS_2)</option>
            <option value="FOODS_3">Packaged Foods (FOODS_3)</option>
          </select>
        </div>
      </div>

      {error ? <div className="banner error">{error}</div> : null}

      {res ? (
        <>
          <div className="ruled-metrics">
            <div className="ruled-item">
              <span>Simulated Profit Lift</span>
              <strong className={`num ${res.profit_lift >= 0 ? "pos" : "neg"}`}>
                {res.profit_lift >= 0 ? "+" : ""}
                {money(res.profit_lift)}
              </strong>
            </div>
            <div className="ruled-item">
              <span>ML vs. Clerk Net Profit</span>
              <strong className="num">
                {money(res.ml_profit)}{" "}
                <small className="muted">vs {money(res.human_profit)}</small>
              </strong>
            </div>
            <div className="ruled-item">
              <span>Food Spoilage Saved</span>
              <strong className={`num ${res.waste_saved >= 0 ? "pos" : "neg"}`}>
                {res.waste_saved >= 0 ? "+" : ""}
                {money(res.waste_saved)}
              </strong>
            </div>
            <div className="ruled-item">
              <span>Units Spoiled (ML vs. Clerk)</span>
              <strong className="num">
                {Math.round(res.ml_waste_units).toLocaleString()} vs{" "}
                {Math.round(res.human_waste_units).toLocaleString()}
              </strong>
            </div>
          </div>

          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Department</th>
                  <th className="num">ML Profit</th>
                  <th className="num">Clerk Profit</th>
                  <th className="num">Profit Lift</th>
                  <th className="num">ML Spoilage</th>
                  <th className="num">Clerk Spoilage</th>
                  <th className="num">Waste Saved</th>
                </tr>
              </thead>
              <tbody>
                {res.departments.map((d) => (
                  <tr key={d.dept_id}>
                    <td>
                      <strong>{d.dept_id}</strong>
                    </td>
                    <td className="num">{money(d.ml_profit)}</td>
                    <td className="num">{money(d.human_profit)}</td>
                    <td className={`num ${d.profit_lift >= 0 ? "pos" : "neg"}`}>
                      {d.profit_lift >= 0 ? "+" : ""}
                      {money(d.profit_lift)}
                    </td>
                    <td className="num">{money(d.ml_waste_cost)}</td>
                    <td className="num">{money(d.human_waste_cost)}</td>
                    <td className={`num ${d.waste_saved >= 0 ? "pos" : "neg"}`}>
                      {d.waste_saved >= 0 ? "+" : ""}
                      {money(d.waste_saved)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      ) : null}
    </div>
  );
}
