import { Suspense, lazy, useEffect, useState } from "react";
import {
  fetchDaily,
  fetchDepartments,
  fetchFeatures,
  fetchHealth,
  fetchPipeline,
  fetchSummary,
} from "./api/client";
import type {
  DailyResponse,
  DeptRow,
  FeaturesResponse,
  Health,
  PipelineResponse,
  Summary,
} from "./api/types";
import { DemandChart } from "./components/DemandChart";
import { DeptTable } from "./components/DeptTable";
import { IconArrowRight, IconLedger } from "./components/Icons";
import { KpiStrip } from "./components/KpiStrip";
import { PipelineLab } from "./components/PipelineLab";
import { ProfitChart } from "./components/ProfitChart";
import { SiteNav, type TopView } from "./components/SiteNav";

const OrderSheet = lazy(() =>
  import("./components/OrderSheet").then((m) => ({ default: m.OrderSheet })),
);
const SimulatorPanel = lazy(() =>
  import("./components/SimulatorPanel").then((m) => ({ default: m.SimulatorPanel })),
);
const ItemTable = lazy(() =>
  import("./components/ItemTable").then((m) => ({ default: m.ItemTable })),
);
const ItemDrawer = lazy(() =>
  import("./components/ItemDrawer").then((m) => ({ default: m.ItemDrawer })),
);
const ChatPanel = lazy(() =>
  import("./components/ChatPanel").then((m) => ({ default: m.ChatPanel })),
);

type DeskTab = "overview" | "orders" | "simulator" | "explorer";

const DESK_TABS: { id: DeskTab; label: string }[] = [
  { id: "orders", label: "Daily Order Sheet" },
  { id: "overview", label: "P&L Overview" },
  { id: "simulator", label: "What-If Simulator" },
  { id: "explorer", label: "SKU Explorer" },
];

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [summary, setSummary] = useState<Summary | null>(null);
  const [pipeline, setPipeline] = useState<PipelineResponse | null>(null);
  const [daily, setDaily] = useState<DailyResponse | null>(null);
  const [depts, setDepts] = useState<DeptRow[]>([]);
  const [features, setFeatures] = useState<FeaturesResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [topView, setTopView] = useState<TopView>("desk");
  const [deskTab, setDeskTab] = useState<DeskTab>("orders");
  const [chatOpen, setChatOpen] = useState(false);
  const [selectedSku, setSelectedSku] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    async function load() {
      try {
        const [hRes, sRes, pipRes, dRes, depRes, featRes] = await Promise.allSettled([
          fetchHealth(),
          fetchSummary(),
          fetchPipeline(),
          fetchDaily(),
          fetchDepartments(),
          fetchFeatures(),
        ]);
        if (!alive) return;

        if (hRes.status === "fulfilled") {
          setHealth(hRes.value);
          if (!hRes.value.data_ready) {
            setError(hRes.value.detail);
            return;
          }
        }
        if (sRes.status === "fulfilled") setSummary(sRes.value);
        else setError(sRes.reason instanceof Error ? sRes.reason.message : String(sRes.reason));

        if (pipRes.status === "fulfilled") setPipeline(pipRes.value);
        if (dRes.status === "fulfilled") setDaily(dRes.value);
        if (depRes.status === "fulfilled") setDepts(depRes.value);
        if (featRes.status === "fulfilled") setFeatures(featRes.value);
      } finally {
        if (alive) setLoading(false);
      }
    }
    void load();
    return () => {
      alive = false;
    };
  }, []);

  function onDeskTabKeyDown(e: React.KeyboardEvent<HTMLButtonElement>, idx: number) {
    if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
      e.preventDefault();
      const next =
        e.key === "ArrowRight"
          ? (idx + 1) % DESK_TABS.length
          : (idx - 1 + DESK_TABS.length) % DESK_TABS.length;
      setDeskTab(DESK_TABS[next].id);
    }
  }

  return (
    <div id="top" className="shell">
      <SiteNav
        activeView={topView}
        onSelectView={setTopView}
        chatOpen={chatOpen}
        chatEnabled={Boolean(health?.chat_enabled)}
        onToggleChat={() => setChatOpen((o) => !o)}
      />

      <main className="page">
        <header className="hero-stage">
          <div>
            <h1>Smarter Grocery Ordering That Raises Profit and Cuts Food Waste</h1>
            <p className="sub">
              Built on Walmart store <code>CA_1</code> receipts: forecast daily SKU demand ranges (
              <code>q50–q95</code>) to trim perishable rot in Produce &amp; Dairy while keeping
              high-margin Packaged favorites in stock.
            </p>
            <div className="hero-actions">
              <button
                type="button"
                className="btn btn-primary"
                onClick={() => {
                  setTopView("desk");
                  setDeskTab("orders");
                }}
              >
                <IconLedger size={16} />
                <span>Daily Order Sheet</span>
              </button>
              <button
                type="button"
                className="btn btn-secondary"
                onClick={() => setTopView(topView === "pipeline" ? "desk" : "pipeline")}
              >
                <span>
                  {topView === "pipeline" ? "Back to Buyer Desk" : "Inspect 4-Step Model Pipeline"}
                </span>
                <IconArrowRight size={15} />
              </button>
            </div>
          </div>

          <figure className="photo-frame">
            <img
              src="/intro/produce-aisle.jpg"
              alt="Fresh whole produce, dairy, bread, legumes, and meat display"
              width={640}
              height={425}
              decoding="async"
            />
            <figcaption className="photo-credit">
              Photo:{" "}
              <a
                href="https://commons.wikimedia.org/wiki/File:Good_Food_Display_-_NCI_Visuals_Online.jpg"
                target="_blank"
                rel="noreferrer"
              >
                Good Food Display — NCI Visuals Online (Wikimedia Commons, Public Domain)
              </a>
            </figcaption>
          </figure>
        </header>

        {loading && !error ? (
          <div className="banner" aria-busy="true">
            Loading store replenishment metrics and holdout ledger...
          </div>
        ) : null}

        {error ? (
          <div className="banner error">
            <strong>Simulation ledger unavailable.</strong> {error} Generate the holdout forecast
            dataset to populate store orders and charts.
          </div>
        ) : null}

        {/* Single authoritative Executive KPI Ledger */}
        {summary ? <KpiStrip s={summary} /> : null}

        {topView === "desk" && summary ? (
          <section
            id="top-panel-desk"
            role="tabpanel"
            aria-labelledby="top-tab-desk"
            className="chapter"
          >
            <div className="desk-header">
              <div>
                <h2>Buyer Replenishment Desk</h2>
                <p className="lead-text">
                  Review holdout order decisions ({summary.date_start} to {summary.date_end}),
                  export PO CSVs, or stress-test spoilage assumptions.
                </p>
              </div>
              <div className="desk-tabs" role="tablist" aria-label="Buyer desk tools">
                {DESK_TABS.map((t, idx) => {
                  const isSelected = deskTab === t.id;
                  return (
                    <button
                      key={t.id}
                      id={`desk-tab-${t.id}`}
                      type="button"
                      role="tab"
                      aria-selected={isSelected}
                      aria-controls={`desk-panel-${t.id}`}
                      tabIndex={isSelected ? 0 : -1}
                      className={`nav-tab ${isSelected ? "active" : ""}`}
                      onClick={() => setDeskTab(t.id)}
                      onKeyDown={(e) => onDeskTabKeyDown(e, idx)}
                    >
                      {t.label}
                    </button>
                  );
                })}
              </div>
            </div>

            {deskTab === "overview" && daily ? (
              <div
                id="desk-panel-overview"
                role="tabpanel"
                aria-labelledby="desk-tab-overview"
                className="desk-block"
              >
                <div className="ledger-split">
                  <div className="ledger-col">
                    <h3>Daily Demand vs. Safety Cushions</h3>
                    <p className="hint">
                      Actual units sold vs. expected forecast and upper safety bands (q75, q90).
                    </p>
                    <DemandChart data={daily.demand} />
                  </div>
                  <div className="ledger-col">
                    <h3>Daily Net Profit: ML vs. Clerk Rule (EWMA)</h3>
                    <p className="hint">
                      Net profit after wholesale cost, perishable spoilage, and missed-sale penalty.
                    </p>
                    <ProfitChart data={daily.profit} />
                  </div>
                </div>
                {depts.length ? <DeptTable rows={depts} /> : null}
              </div>
            ) : null}

            <Suspense fallback={<div className="banner">Loading workspace view...</div>}>
              {deskTab === "orders" ? (
                <div id="desk-panel-orders" role="tabpanel" aria-labelledby="desk-tab-orders">
                  <OrderSheet onSelectItem={(id) => setSelectedSku(id)} />
                </div>
              ) : null}

              {deskTab === "simulator" ? (
                <div id="desk-panel-simulator" role="tabpanel" aria-labelledby="desk-tab-simulator">
                  <SimulatorPanel totalRows={summary.rows} />
                </div>
              ) : null}

              {deskTab === "explorer" ? (
                <div id="desk-panel-explorer" role="tabpanel" aria-labelledby="desk-tab-explorer">
                  <ItemTable
                    dateStart={summary.date_start}
                    dateEnd={summary.date_end}
                    onSelectItem={(id) => setSelectedSku(id)}
                  />
                </div>
              ) : null}
            </Suspense>
          </section>
        ) : null}

        {topView === "pipeline" && pipeline && summary && features ? (
          <div id="top-panel-pipeline" role="tabpanel" aria-labelledby="top-tab-pipeline">
            <PipelineLab
              pipeline={pipeline}
              summary={summary}
              importance={features.importance}
              cvCaption={features.cv_caption}
              onOpenDeskTab={(tab) => {
                setTopView("desk");
                setDeskTab(tab);
                requestAnimationFrame(() => {
                  document.getElementById("top")?.scrollIntoView({ behavior: "smooth" });
                });
              }}
            />
          </div>
        ) : null}
      </main>

      <Suspense fallback={null}>
        {selectedSku ? (
          <ItemDrawer itemId={selectedSku} onClose={() => setSelectedSku(null)} />
        ) : null}
        {chatOpen ? (
          <ChatPanel
            open={chatOpen}
            onClose={() => setChatOpen(false)}
            chatEnabled={Boolean(health?.chat_enabled)}
          />
        ) : null}
      </Suspense>
    </div>
  );
}
