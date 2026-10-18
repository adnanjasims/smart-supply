import { useEffect, useState } from "react";
import {
  fetchDaily,
  fetchDepartments,
  fetchFeatures,
  fetchHealth,
  fetchSummary,
} from "./api/client";
import type {
  DailyPayload,
  DeptRow,
  FeatureRow,
  Health,
  Summary,
} from "./api/types";
import { ChatPanel } from "./components/ChatPanel";
import { DatasetSection } from "./components/DatasetSection";
import { DemandChart } from "./components/DemandChart";
import { DeptTable } from "./components/DeptTable";
import { FeatureChart } from "./components/FeatureChart";
import { ItemTable } from "./components/ItemTable";
import { KpiStrip } from "./components/KpiStrip";
import { MethodSection } from "./components/MethodSection";
import { MetricNotes } from "./components/MetricNotes";
import { ProfitChart } from "./components/ProfitChart";
import { ProjectIntro } from "./components/ProjectIntro";
import { SiteNav } from "./components/SiteNav";

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [summary, setSummary] = useState<Summary | null>(null);
  const [daily, setDaily] = useState<DailyPayload | null>(null);
  const [depts, setDepts] = useState<DeptRow[]>([]);
  const [features, setFeatures] = useState<FeatureRow[]>([]);
  const [cvCaption, setCvCaption] = useState<string | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [chatOpen, setChatOpen] = useState(false);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      setLoading(true);
      setError("");
      try {
        const h = await fetchHealth();
        if (cancelled) return;
        setHealth(h);
        if (!h.data_ready) {
          setError(h.detail);
          return;
        }
        const [s, d, deptRes, featRes] = await Promise.all([
          fetchSummary(),
          fetchDaily(),
          fetchDepartments(),
          fetchFeatures(),
        ]);
        if (cancelled) return;
        setSummary(s);
        setDaily(d);
        setDepts(deptRes.departments);
        setFeatures(featRes.importance);
        setCvCaption(featRes.cv_caption);
      } catch (e) {
        if (!cancelled) {
          setError(
            e instanceof Error
              ? e.message
              : "Could not reach the API. Start uvicorn on port 8000."
          );
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    void load();
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className="app">
      <SiteNav onAsk={() => setChatOpen(true)} />

      <header className="hero-head">
        <h1 className="brand">Smart Supply</h1>
        <p className="tagline">
          Demand forecasting and stocking for grocery, measured on profit against a
          historical baseline.
        </p>
      </header>

      <ProjectIntro />
      <DatasetSection />
      <MethodSection />

      {error ? <div className="banner error">{error}</div> : null}
      {loading ? <p className="muted">Loading holdout run...</p> : null}

      <div id="results" className="section-anchor">
        {summary ? (
          <>
            <p className="muted results-label">
              Holdout results | {summary.date_start} to {summary.date_end} |{" "}
              {summary.items.toLocaleString()} items | {summary.rows.toLocaleString()} rows
            </p>
            <KpiStrip summary={summary} />
            <MetricNotes />
          </>
        ) : null}

        {daily ? (
          <>
            <section className="panel">
              <h2>Daily profit</h2>
              <p className="hint">ML policy versus the EWMA + seasonal-naive baseline.</p>
              <ProfitChart data={daily.profit} />
            </section>
            <section className="panel">
              <h2>Demand and intervals</h2>
              <p className="hint">
                Store-level actuals against the point forecast and q75 / q90 bounds.
              </p>
              <DemandChart data={daily.demand} />
            </section>
          </>
        ) : null}

        <section className="panel">
          <h2>Departments</h2>
          <p className="hint">
            FOODS_1 perishables, FOODS_2 dairy/deli, FOODS_3 packaged. Stock levels follow
            each dept&apos;s newsvendor critical fractile.
          </p>
          <DeptTable rows={depts} />
        </section>

        <section className="panel">
          <h2>Feature importance</h2>
          <p className="hint">LightGBM gain from the trained point model.</p>
          <FeatureChart rows={features} caption={cvCaption} />
        </section>
      </div>

      {summary ? (
        <section id="explore" className="panel section-block">
          <h2>Explore</h2>
          <p className="hint">Filter the holdout export. Capped for the browser.</p>
          <ItemTable dateStart={summary.date_start} dateEnd={summary.date_end} />
        </section>
      ) : null}

      <ChatPanel
        open={chatOpen}
        onClose={() => setChatOpen(false)}
        chatEnabled={!!health?.chat_enabled}
      />
    </div>
  );
}
