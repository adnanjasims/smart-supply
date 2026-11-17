import { memo, useCallback, useEffect, useRef, useState } from "react";
import { fetchOrders } from "../api/client";
import type { FeatureRow, OrderRow, PipelineDept, PipelineResponse, Summary } from "../api/types";
import { money, pct, unitMoney } from "../lib/format";
import { FeatureChart } from "./FeatureChart";
import { IconArrowRight, IconCheck, IconPause, IconPlay, IconRefresh } from "./Icons";

type StageIndex = 1 | 2 | 3 | 4;

interface SampleRecord {
  id: string;
  sku: string;
  dept: string;
  dayLabel: string;
  priceLabel: string;
  sales: number;
  kind: "warmup" | "unpriced" | "kept";
}

const LADDER_KNOTS = [
  { key: "q50", q: 0.5 },
  { key: "q62", q: 0.62 },
  { key: "q75", q: 0.75 },
  { key: "q83", q: 0.83 },
  { key: "q90", q: 0.9 },
  { key: "q95", q: 0.95 },
];

const STAGE_META: { id: StageIndex; title: string; subtitle: string }[] = [
  { id: 1, title: "1. Raw Store Data", subtitle: "Demand sparsity & aisle risk" },
  { id: 2, title: "2. Clean & Prep Data", subtitle: "Filter leaks & inactive weeks" },
  { id: 3, title: "3. Model & Cushions", subtitle: "Optuna, LightGBM & cushions" },
  { id: 4, title: "4. Store Takeaways", subtitle: "Why each aisle strategy wins" },
];

// smooth numeric tween for counters (respects prefers-reduced-motion)
function useAnimatedNumber(target: number, durationMs = 460): number {
  const [display, setDisplay] = useState(target);
  const currentRef = useRef(target);

  useEffect(() => {
    const prefersReduced =
      typeof window !== "undefined" &&
      window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    if (prefersReduced) {
      currentRef.current = target;
      setDisplay(target);
      return;
    }

    const from = currentRef.current;
    const delta = target - from;
    if (delta === 0) return;

    let raf = 0;
    const start = performance.now();
    const tick = (now: number) => {
      const p = Math.min(1, (now - start) / durationMs);
      const eased = 1 - Math.pow(1 - p, 3);
      const val = Math.round(from + delta * eased);
      currentRef.current = val;
      setDisplay(val);
      if (p < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [target, durationMs]);

  return display;
}

export const PipelineLab = memo(function PipelineLab({
  pipeline,
  summary,
  importance,
  cvCaption,
  onOpenDeskTab,
}: {
  pipeline: PipelineResponse;
  summary: Summary;
  importance: FeatureRow[];
  cvCaption: string | null;
  onOpenDeskTab: (tab: "orders" | "simulator") => void;
}) {
  const [stage, setStage] = useState<StageIndex>(1);
  const [autoPlay, setAutoPlay] = useState(false);

  const [selectedDept, setSelectedDept] = useState<string>(
    pipeline.departments[0]?.dept_id ?? "FOODS_1",
  );

  // step 2 interactive cleaning controls
  const [dropWarmup, setDropWarmup] = useState<boolean>(false);
  const [dropUnpriced, setDropUnpriced] = useState<boolean>(false);
  const [selectedGroup, setSelectedGroup] = useState<string>(
    pipeline.feature_groups[0]?.group ?? "",
  );

  // step 3 interactive replay of logged Optuna tuning & 4 chronological folds
  const finalLr = pipeline.best_params.learning_rate ?? 0.086;
  const finalLeaves = pipeline.best_params.num_leaves ?? 120;
  const finalMinLeaf = pipeline.best_params.min_data_in_leaf ?? 21;

  const tuningSteps = [
    { trial: 1, label: "Coarse tree sweep", lr: 0.025, leaves: 48, minLeaf: 90 },
    { trial: 4, label: "Expanding branch capacity", lr: 0.045, leaves: 84, minLeaf: 56 },
    { trial: 8, label: "Refining event & lag splits", lr: 0.068, leaves: 108, minLeaf: 32 },
    {
      trial: 12,
      label: "Logged Optuna parameters locked",
      lr: finalLr,
      leaves: finalLeaves,
      minLeaf: finalMinLeaf,
    },
  ];

  const [tuneStepIdx, setTuneStepIdx] = useState<number>(tuningSteps.length - 1);
  const [trainedFolds, setTrainedFolds] = useState<number>(pipeline.cv_scores.length);
  const [isSimTraining, setIsSimTraining] = useState<boolean>(false);
  const [sampleSkus, setSampleSkus] = useState<OrderRow[]>([]);
  const [poissonImgErr, setPoissonImgErr] = useState(false);
  const [lgbmImgErr, setLgbmImgErr] = useState(false);

  useEffect(() => {
    let alive = true;
    fetchOrders({ limit: 8 })
      .then((res) => {
        if (alive) setSampleSkus(res.rows.slice(0, 6));
      })
      .catch(() => {
        // non-blocking visual illustration
      });
    return () => {
      alive = false;
    };
  }, []);

  const goToStage = useCallback((nextStage: StageIndex, shouldScroll = true) => {
    setStage(nextStage);
    if (shouldScroll) {
      requestAnimationFrame(() => {
        document.getElementById("pipeline")?.scrollIntoView({ behavior: "smooth", block: "start" });
      });
    }
  }, []);

  const startTrainingReplay = useCallback(() => {
    setTuneStepIdx(0);
    setTrainedFolds(0);
    setIsSimTraining(true);
  }, []);

  // guided tour timer
  useEffect(() => {
    if (!autoPlay) return;
    const t = window.setTimeout(() => {
      if (stage === 1) {
        goToStage(2);
        setDropWarmup(false);
        setDropUnpriced(false);
      } else if (stage === 2) {
        if (!dropWarmup) setDropWarmup(true);
        else if (!dropUnpriced) setDropUnpriced(true);
        else {
          goToStage(3);
          startTrainingReplay();
        }
      } else if (stage === 3) {
        if (!isSimTraining) {
          goToStage(4);
          setAutoPlay(false);
        }
      } else {
        setAutoPlay(false);
      }
    }, 1700);
    return () => window.clearTimeout(t);
  }, [autoPlay, stage, dropWarmup, dropUnpriced, isSimTraining, goToStage, startTrainingReplay]);

  // step 3 replay progression across trials -> chronological folds
  useEffect(() => {
    if (!isSimTraining) return;
    const timer = window.setTimeout(() => {
      if (tuneStepIdx < tuningSteps.length - 1) {
        setTuneStepIdx((i) => i + 1);
      } else if (trainedFolds < pipeline.cv_scores.length) {
        setTrainedFolds((f) => f + 1);
      } else {
        setIsSimTraining(false);
      }
    }, 500);
    return () => window.clearTimeout(timer);
  }, [isSimTraining, tuneStepIdx, trainedFolds, tuningSteps.length, pipeline.cv_scores.length]);

  const { cleaning } = pipeline;
  const survivingTarget =
    cleaning.raw_rows -
    (dropWarmup ? cleaning.warmup_dropped : 0) -
    (dropUnpriced ? cleaning.missing_price_dropped : 0);
  const animatedSurviving = useAnimatedNumber(survivingTarget);
  const retentionPct = cleaning.raw_rows > 0 ? (survivingTarget / cleaning.raw_rows) * 100 : 100;

  const activeDept: PipelineDept | undefined =
    pipeline.departments.find((d) => d.dept_id === selectedDept) ?? pipeline.departments[0];
  const activeFeatureGroup =
    pipeline.feature_groups.find((g) => g.group === selectedGroup) ?? pipeline.feature_groups[0];
  const activeTune = tuningSteps[tuneStepIdx] ?? tuningSteps[tuningSteps.length - 1];
  const trainingLocked = !isSimTraining && trainedFolds >= pipeline.cv_scores.length;

  const capPct = activeDept
    ? Math.min(96, Math.max(4, ((activeDept.waste_aware_cap - 0.5) / 0.45) * 100))
    : 30;
  const maxQuantile = activeDept
    ? Math.max(...LADDER_KNOTS.map((k) => activeDept.quantile_means[k.key] ?? 0), 1)
    : 1;

  // proper non-inverted y-scale: lower MAE sits lower on chart
  const allScores = [...pipeline.cv_scores, pipeline.holdout_mae];
  const minMae = Math.floor((Math.min(...allScores) - 0.03) * 100) / 100;
  const maxMae = Math.ceil((Math.max(...allScores) + 0.03) * 100) / 100;
  const maeRange = Math.max(0.05, maxMae - minMae);
  const midMae = minMae + maeRange / 2;
  const plotTop = 26;
  const plotBottom = 96;
  const mapY = (s: number) =>
    plotTop + ((maxMae - s) / maeRange) * (plotBottom - plotTop);

  const cvPts = pipeline.cv_scores.map((s, idx) => {
    const x = 62 + (idx * 246) / Math.max(1, pipeline.cv_scores.length - 1);
    return { x, y: mapY(s), s, idx };
  });
  const revealedPts = cvPts.slice(0, trainedFolds);
  const polylinePoints = revealedPts.map((p) => `${p.x},${p.y}`).join(" ");

  // sample records built from real SKUs and prices returned by /api/orders
  const skuA = sampleSkus[0]?.item_id ?? "FOODS_3_090";
  const skuB = sampleSkus[1]?.item_id ?? "FOODS_2_197";
  const skuC = sampleSkus[2]?.item_id ?? "FOODS_1_018";
  const ledgerRows: SampleRecord[] = [
    {
      id: "r1",
      sku: skuC,
      dept: "Produce",
      dayLabel: "Early window (lag warmup)",
      priceLabel: unitMoney(sampleSkus[2]?.sell_price ?? 3.33),
      sales: 1,
      kind: "warmup",
    },
    {
      id: "r2",
      sku: skuA,
      dept: "Packaged",
      dayLabel: "Early window (lag warmup)",
      priceLabel: unitMoney(sampleSkus[0]?.sell_price ?? 2.92),
      sales: 0,
      kind: "warmup",
    },
    {
      id: "r3",
      sku: skuB,
      dept: "Dairy",
      dayLabel: "Off-shelf week",
      priceLabel: "No sell_price",
      sales: 0,
      kind: "unpriced",
    },
    {
      id: "r4",
      sku: skuC,
      dept: "Produce",
      dayLabel: sampleSkus[2]?.date ?? "Holdout sample",
      priceLabel: unitMoney(sampleSkus[2]?.sell_price ?? 3.48),
      sales: sampleSkus[2]?.sales ?? 2,
      kind: "kept",
    },
    {
      id: "r5",
      sku: skuB,
      dept: "Dairy",
      dayLabel: sampleSkus[1]?.date ?? "Holdout sample",
      priceLabel: unitMoney(sampleSkus[1]?.sell_price ?? 4.28),
      sales: sampleSkus[1]?.sales ?? 0,
      kind: "kept",
    },
    {
      id: "r6",
      sku: skuA,
      dept: "Packaged",
      dayLabel: sampleSkus[0]?.date ?? "Holdout sample",
      priceLabel: unitMoney(sampleSkus[0]?.sell_price ?? 2.88),
      sales: sampleSkus[0]?.sales ?? 6,
      kind: "kept",
    },
  ];

  function onStepTabKeyDown(e: React.KeyboardEvent<HTMLButtonElement>, idx: number) {
    if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
      e.preventDefault();
      const next =
        e.key === "ArrowRight"
          ? (idx + 1) % STAGE_META.length
          : (idx - 1 + STAGE_META.length) % STAGE_META.length;
      setAutoPlay(false);
      goToStage(STAGE_META[next].id);
    }
  }

  function onAisleKeyDown(e: React.KeyboardEvent<HTMLButtonElement>, idx: number) {
    if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
      e.preventDefault();
      const list = pipeline.departments;
      const next = e.key === "ArrowRight" ? (idx + 1) % list.length : (idx - 1 + list.length) % list.length;
      setSelectedDept(list[next].dept_id);
    }
  }

  return (
    <section id="pipeline" className="chapter">
      <div className="pipeline-topbar">
        <div>
          <h2>How SmartSupply Works</h2>
          <p className="lead-text">
            Step through the forecasting and Newsvendor replenishment pipeline built on Walmart M5
            store receipts, or run the guided tour from raw records to aisle decisions.
          </p>
        </div>

        <div className="pipeline-controls">
          <button
            type="button"
            className="btn btn-primary"
            onClick={() => {
              if (autoPlay) {
                setAutoPlay(false);
              } else {
                if (stage === 4) goToStage(1, false);
                setAutoPlay(true);
              }
            }}
          >
            {autoPlay ? <IconPause /> : <IconPlay />}
            <span>{autoPlay ? "Pause Tour" : "Play Guided Tour"}</span>
          </button>
          <button
            type="button"
            className="btn btn-secondary"
            onClick={() => {
              setAutoPlay(false);
              setDropWarmup(false);
              setDropUnpriced(false);
              setTuneStepIdx(tuningSteps.length - 1);
              setTrainedFolds(pipeline.cv_scores.length);
              setIsSimTraining(false);
              goToStage(1);
            }}
          >
            <IconRefresh />
            <span>Reset</span>
          </button>
        </div>
      </div>

      {/* 4-Stage Stepper */}
      <div className="stepper-wrap">
        <div className="stepper-track" aria-hidden="true">
          <div className="stepper-fill" style={{ transform: `scaleX(${stage / 4})` }} />
        </div>
        <div className="stepper-tabs" role="tablist" aria-label="Pipeline stages">
          {STAGE_META.map((item, idx) => {
            const isActive = stage === item.id;
            const isDone = stage > item.id;
            return (
              <button
                key={item.id}
                id={`stage-tab-${item.id}`}
                type="button"
                role="tab"
                aria-selected={isActive}
                aria-controls={`stage-panel-${item.id}`}
                tabIndex={isActive ? 0 : -1}
                className={`step-tab ${isActive ? "active" : ""} ${isDone ? "done" : ""}`}
                onClick={() => {
                  setAutoPlay(false);
                  goToStage(item.id);
                }}
                onKeyDown={(e) => onStepTabKeyDown(e, idx)}
              >
                <span className="step-title">
                  {isDone ? <IconCheck size={14} /> : null}
                  <span>{item.title}</span>
                </span>
                <span className="step-sub">{item.subtitle}</span>
              </button>
            );
          })}
        </div>
      </div>

      {/* STAGE 1: RAW STORE DATA */}
      {stage === 1 ? (
        <div
          id="stage-panel-1"
          role="tabpanel"
          aria-labelledby="stage-tab-1"
          className="stage-body"
          key="stage-1"
        >
          <div className="chapter-split">
            <div>
              <h3>Why One Simple Average Fails Grocery Aisles</h3>
              <p className="body-copy">
                SmartSupply ingests{" "}
                <strong className="num">{cleaning.raw_rows.toLocaleString()}</strong> daily item
                records ({pipeline.skus.toLocaleString()} grocery SKUs across{" "}
                {cleaning.lookback_days} days in store <code>{pipeline.store_id}</code>). Nearly{" "}
                <strong className="num">{pct(pipeline.zero_sale_pct)}</strong> of SKU-days record{" "}
                <strong>0 sales</strong>, mixed with payday and weekend spikes.
              </p>
              <p className="body-copy">
                Using a uniform safety cushion across every aisle fails: over-ordering fresh{" "}
                <strong>Produce</strong> causes overnight rot, whereas under-ordering{" "}
                <strong>Packaged Foods</strong> loses high-margin sales when shelves empty.
              </p>
            </div>

            <figure className="photo-frame">
              <img
                src="/intro/grocery-shelves.jpg"
                alt="Supermarket produce and grocery display aisle"
                width={640}
                height={480}
                loading="lazy"
                decoding="async"
              />
              <figcaption className="photo-credit">
                Photo:{" "}
                <a
                  href="https://commons.wikimedia.org/wiki/Category:Supermarkets_in_Brazil"
                  target="_blank"
                  rel="noreferrer"
                >
                  Supermarket produce aisle (Wikimedia Commons)
                </a>{" "}
                · Data:{" "}
                <a
                  href="https://www.kaggle.com/competitions/m5-forecasting-accuracy/data"
                  target="_blank"
                  rel="noreferrer"
                >
                  Kaggle Walmart M5
                </a>
              </figcaption>
            </figure>
          </div>

          <div className="aisle-tabs" role="tablist" aria-label="Department risk comparison">
            {pipeline.departments.map((d, idx) => {
              const isSelected = d.dept_id === selectedDept;
              const penaltyRatio =
                d.unit_spoilage_cost > 0 ? d.unit_stockout_cost / d.unit_spoilage_cost : 0;
              return (
                <button
                  key={d.dept_id}
                  id={`aisle-tab-${d.dept_id}`}
                  type="button"
                  role="tab"
                  aria-selected={isSelected}
                  aria-controls="aisle-detail-panel"
                  tabIndex={isSelected ? 0 : -1}
                  className={`aisle-tab ${isSelected ? "selected" : ""}`}
                  onClick={() => setSelectedDept(d.dept_id)}
                  onKeyDown={(e) => onAisleKeyDown(e, idx)}
                >
                  <div className="aisle-tab-head">
                    <strong>{d.label}</strong>
                    <span className="tag">{d.dept_id}</span>
                  </div>
                  <p className="muted aisle-meta">
                    {d.sku_count.toLocaleString()} items · {pct(d.zero_sale_pct)} zero-sale days
                  </p>
                  <div className="ratio-row">
                    <span>Overnight spoilage rate</span>
                    <strong className="num neg">{pct(d.spoilage_rate * 100)}/day</strong>
                  </div>
                  <div className="ratio-row ratio-row-last">
                    <span>Stockout vs. spoilage cost</span>
                    <strong className="num pos">{Math.round(penaltyRatio)}x</strong>
                  </div>
                </button>
              );
            })}
          </div>

          {activeDept ? (
            <div
              id="aisle-detail-panel"
              role="tabpanel"
              aria-labelledby={`aisle-tab-${activeDept.dept_id}`}
              className="ruled-metrics"
            >
              <div className="ruled-item">
                <span>Average Shelf Price</span>
                <strong className="num">{unitMoney(activeDept.avg_sell_price)}</strong>
              </div>
              <div className="ruled-item">
                <span>Cost of 1 Unsold Spoiled Unit</span>
                <strong className="num neg">{unitMoney(activeDept.unit_spoilage_cost)}</strong>
              </div>
              <div className="ruled-item">
                <span>Cost of 1 Missed Sale</span>
                <strong className="num">{unitMoney(activeDept.unit_stockout_cost)}</strong>
              </div>
              <div className="ruled-item">
                <span>Average Daily Demand</span>
                <strong className="num">{activeDept.avg_daily_sales.toFixed(2)} units/SKU</strong>
              </div>
            </div>
          ) : null}

          <div className="stage-footer">
            <span className="muted">
              Next, filter incomplete startup history and unstocked weeks before training.
            </span>
            <button type="button" className="btn btn-primary" onClick={() => goToStage(2)}>
              <span>Step 2: Clean This Raw Data</span>
              <IconArrowRight />
            </button>
          </div>
        </div>
      ) : null}

      {/* STAGE 2: ANIMATED DATA CLEANING */}
      {stage === 2 ? (
        <div
          id="stage-panel-2"
          role="tabpanel"
          aria-labelledby="stage-tab-2"
          className="stage-body"
          key="stage-2"
        >
          <div className="ledger-split">
            <div className="ledger-col">
              <h3>Interactive Data Cleaning Pipeline</h3>
              <p className="hint">
                Toggle either filter below to watch non-learnable records exit the dataset while the
                verified training row count updates smoothly.
              </p>

              <div className="clean-actions">
                <button
                  type="button"
                  className={`clean-toggle ${dropWarmup ? "on" : ""}`}
                  aria-pressed={dropWarmup}
                  onClick={() => setDropWarmup((v) => !v)}
                >
                  <div>
                    <strong>
                      1. Enforce Safe History Window{" "}
                      <span className="muted">(28-day causal lag warmup)</span>
                    </strong>
                    <p className="muted sub-note">
                      Drops each item's earliest 28 days so past rolling averages never leak future
                      demand.
                    </p>
                  </div>
                  <span className="num neg">-{cleaning.warmup_dropped.toLocaleString()} rows</span>
                </button>

                <button
                  type="button"
                  className={`clean-toggle ${dropUnpriced ? "on" : ""}`}
                  aria-pressed={dropUnpriced}
                  onClick={() => setDropUnpriced((v) => !v)}
                >
                  <div>
                    <strong>
                      2. Filter Off-Shelf Weeks{" "}
                      <span className="muted">(missing weekly sell price)</span>
                    </strong>
                    <p className="muted sub-note">
                      Removes item-days without an active shelf price so uncarried items aren't
                      scored as zero demand.
                    </p>
                  </div>
                  <span className="num neg">
                    -{cleaning.missing_price_dropped.toLocaleString()} rows
                  </span>
                </button>
              </div>

              <div className="survival-meter">
                <div className="funnel-meta">
                  <span>Clean Training Records</span>
                  <strong className="num pos">
                    {animatedSurviving.toLocaleString()} / {cleaning.raw_rows.toLocaleString()} (
                    {retentionPct.toFixed(1)}%)
                  </strong>
                </div>
                <div className="bar-track tall">
                  <div
                    className="bar-fill forest"
                    style={{ transform: `scaleX(${Math.max(0.05, retentionPct / 100)})` }}
                  />
                </div>
              </div>

              <h3 className="group-heading">
                Signals Built Per Clean Record ({pipeline.feature_count} Inputs)
              </h3>
              <div className="pill-row">
                {pipeline.feature_groups.map((grp) => (
                  <button
                    key={grp.group}
                    type="button"
                    className={`chip ${grp.group === activeFeatureGroup?.group ? "active" : ""}`}
                    onClick={() => setSelectedGroup(grp.group)}
                  >
                    {grp.group} <span className="num">({grp.count})</span>
                  </button>
                ))}
              </div>
              {activeFeatureGroup ? (
                <div className="code-samples">
                  {activeFeatureGroup.examples.map((col) => (
                    <code key={col}>{col}</code>
                  ))}
                </div>
              ) : null}
            </div>

            <div className="ledger-col">
              <h3>Sample Record Stream Preview</h3>
              <p className="hint">
                Illustrative day-level records using active M5 SKU names showing how both filters
                remove unusable rows.
              </p>

              <div className="ledger-stream" aria-live="polite">
                {ledgerRows.map((row) => {
                  const filtered =
                    (row.kind === "warmup" && dropWarmup) ||
                    (row.kind === "unpriced" && dropUnpriced);
                  return (
                    <div key={row.id} className={`stream-row ${filtered ? "filtered" : ""}`}>
                      <div className="stream-main">
                        <strong>{row.sku}</strong>
                        <span className="muted">
                          {row.dept} · {row.dayLabel} · {row.priceLabel} · {row.sales} sold
                        </span>
                      </div>
                      <span
                        className={`badge ${
                          filtered
                            ? "badge-danger"
                            : row.kind === "kept"
                              ? "badge-trim"
                              : "badge-hold"
                        }`}
                      >
                        {filtered
                          ? "Filtered out"
                          : row.kind === "kept"
                            ? "Clean row"
                            : "Needs filter"}
                      </span>
                    </div>
                  );
                })}
              </div>

              <div className="action-row">
                <button
                  type="button"
                  className="btn btn-secondary"
                  onClick={() => {
                    setDropWarmup(true);
                    setDropUnpriced(true);
                  }}
                >
                  Apply Both Filters ({cleaning.kept_rows.toLocaleString()} kept)
                </button>
              </div>
            </div>
          </div>

          <div className="stage-footer">
            <button type="button" className="btn btn-secondary" onClick={() => goToStage(1)}>
              Back to Step 1
            </button>
            <button
              type="button"
              className="btn btn-primary"
              onClick={() => {
                setDropWarmup(true);
                setDropUnpriced(true);
                goToStage(3);
                startTrainingReplay();
              }}
            >
              <span>Step 3: Replay Tuning &amp; Cushions</span>
              <IconArrowRight />
            </button>
          </div>
        </div>
      ) : null}

      {/* STAGE 3: HONEST REPLAY OF LOGGED TUNING + METHOD DIAGRAMS + CUSHION EXPLORER */}
      {stage === 3 ? (
        <div
          id="stage-panel-3"
          role="tabpanel"
          aria-labelledby="stage-tab-3"
          className="stage-body"
          key="stage-3"
        >
          <div className="ledger-split">
            <div className="ledger-col">
              <div className="panel-head">
                <div>
                  <h3>Interactive Replay of Logged Optuna &amp; 4-Fold Run</h3>
                  <p className="hint tight-hint">
                    Steps through logged Optuna search convergence onto the exact saved LightGBM
                    parameters over {cleaning.kept_rows.toLocaleString()} rows.
                  </p>
                </div>
                <button
                  type="button"
                  className="btn btn-primary"
                  onClick={startTrainingReplay}
                  disabled={isSimTraining}
                >
                  <IconPlay />
                  <span>{isSimTraining ? "Replaying..." : "Replay Logged Run"}</span>
                </button>
              </div>

              <div className="tune-panel">
                <div className="funnel-meta tune-meta">
                  <span>
                    Checkpoint <strong className="num">{activeTune.trial}/12</strong> ·{" "}
                    {activeTune.label}
                  </span>
                  <span className={`badge ${trainingLocked ? "badge-trim" : "badge-hold"}`}>
                    {trainingLocked ? "Saved Parameters Locked" : "Replaying Search"}
                  </span>
                </div>

                <div className="param-meters">
                  <div>
                    <div className="funnel-meta">
                      <span>
                        Learning Rate (<code>learning_rate</code>)
                      </span>
                      <strong className="num">{activeTune.lr}</strong>
                    </div>
                    <div className="bar-track">
                      <div
                        className="bar-fill blue"
                        style={{ transform: `scaleX(${Math.min(1, activeTune.lr / 0.1)})` }}
                      />
                    </div>
                  </div>

                  <div>
                    <div className="funnel-meta">
                      <span>
                        Tree Complexity (<code>num_leaves</code>)
                      </span>
                      <strong className="num">{activeTune.leaves} leaves</strong>
                    </div>
                    <div className="bar-track">
                      <div
                        className="bar-fill forest"
                        style={{ transform: `scaleX(${Math.min(1, activeTune.leaves / 160)})` }}
                      />
                    </div>
                  </div>

                  <div>
                    <div className="funnel-meta">
                      <span>
                        Min Samples Per Leaf (<code>min_data_in_leaf</code>)
                      </span>
                      <strong className="num">{activeTune.minLeaf} rows</strong>
                    </div>
                    <div className="bar-track">
                      <div
                        className="bar-fill boost"
                        style={{ transform: `scaleX(${Math.min(1, activeTune.minLeaf / 120)})` }}
                      />
                    </div>
                  </div>
                </div>
              </div>

              {/* Proper non-inverted CV MAE curve with explicit Y-axis ticks */}
              <div className="convergence-box">
                <div className="funnel-meta">
                  <span>Chronological Cross-Validation Error (Lower = More Accurate)</span>
                  <strong className="num">
                    {trainedFolds > 0
                      ? `Holdout MAE: ${pipeline.holdout_mae} units/day`
                      : "Running folds..."}
                  </strong>
                </div>
                <svg
                  className="convergence-svg"
                  viewBox="0 0 340 124"
                  role="img"
                  aria-label="Expanding chronological validation MAE descending across 4 folds"
                >
                  <text x="8" y={plotTop + 3} fontSize="9.5" fill="#4b5550">
                    {maxMae.toFixed(2)}
                  </text>
                  <line x1="44" y1={plotTop} x2="322" y2={plotTop} stroke="#e2e6e3" strokeDasharray="3 3" strokeWidth="1" />

                  <text x="8" y={(plotTop + plotBottom) / 2 + 3} fontSize="9.5" fill="#4b5550">
                    {midMae.toFixed(2)}
                  </text>
                  <line x1="44" y1={(plotTop + plotBottom) / 2} x2="322" y2={(plotTop + plotBottom) / 2} stroke="#e2e6e3" strokeDasharray="3 3" strokeWidth="1" />

                  <text x="8" y={plotBottom + 3} fontSize="9.5" fill="#4b5550">
                    {minMae.toFixed(2)}
                  </text>
                  <line x1="44" y1={plotBottom} x2="322" y2={plotBottom} stroke="#c8cfc9" strokeWidth="1" />

                  {revealedPts.length > 1 ? (
                    <polyline
                      fill="none"
                      stroke="#166534"
                      strokeWidth="2.5"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      points={polylinePoints}
                    />
                  ) : null}
                  {revealedPts.map((pt) => (
                    <g key={pt.idx}>
                      <circle cx={pt.x} cy={pt.y} r="4.5" fill="#166534" />
                      <text
                        x={pt.x}
                        y={Math.max(14, pt.y - 9)}
                        textAnchor="middle"
                        fontSize="10"
                        fontWeight="600"
                        fill="#18201b"
                      >
                        {pt.s.toFixed(3)}
                      </text>
                      <text x={pt.x} y="114" textAnchor="middle" fontSize="10" fill="#4b5550">
                        Fold {pt.idx + 1}
                      </text>
                    </g>
                  ))}
                </svg>
              </div>

              <div className="ruled-metrics two">
                <div className="ruled-item">
                  <span>Saved LightGBM Hyperparameters</span>
                  <strong className="num">
                    lr={finalLr} · {finalLeaves} leaves · min={finalMinLeaf}
                  </strong>
                </div>
                <div className="ruled-item">
                  <span>Expanding 4-Fold CV Error</span>
                  <strong className="num">
                    {pipeline.cv_mae_mean} ± {pipeline.cv_mae_std} units/SKU-day
                  </strong>
                </div>
              </div>

              <div className="method-figures">
                <figure className="diagram-fig">
                  {poissonImgErr ? (
                    <div className="diagram-fallback">
                      <strong>Poisson Count Objective</strong>
                      <p className="muted">
                        Predicts non-negative integer sales rates tailored for sparse zero-heavy
                        grocery demand.
                      </p>
                    </div>
                  ) : (
                    <img
                      src="https://commons.wikimedia.org/wiki/Special:FilePath/Poisson_pmf.svg"
                      alt="Poisson probability mass function showing discrete skewed integer counts"
                      loading="lazy"
                      decoding="async"
                      onError={() => setPoissonImgErr(true)}
                    />
                  )}
                  <figcaption className="photo-credit">
                    Why Poisson handles zeros:{" "}
                    <a
                      href="https://en.wikipedia.org/wiki/Poisson_distribution"
                      target="_blank"
                      rel="noreferrer"
                    >
                      Poisson Distribution (Wikipedia)
                    </a>
                  </figcaption>
                </figure>

                <figure className="diagram-fig">
                  {lgbmImgErr ? (
                    <div className="diagram-fallback">
                      <strong>Leaf-Wise Tree Growth</strong>
                      <p className="muted">
                        Splits the single highest-gain demand branch first rather than expanding
                        uniform tree levels.
                      </p>
                    </div>
                  ) : (
                    <img
                      src="https://lightgbm.readthedocs.io/en/latest/_images/leaf-wise.png"
                      alt="LightGBM leaf-wise tree growth splitting the highest-gain leaf first"
                      loading="lazy"
                      decoding="async"
                      onError={() => setLgbmImgErr(true)}
                    />
                  )}
                  <figcaption className="photo-credit">
                    Why Leaf-Wise learns faster:{" "}
                    <a
                      href="https://lightgbm.readthedocs.io/en/latest/Features.html#leaf-wise-best-first-tree-growth"
                      target="_blank"
                      rel="noreferrer"
                    >
                      LightGBM Docs (Microsoft)
                    </a>
                  </figcaption>
                </figure>
              </div>
            </div>

            <div className="ledger-col">
              <h3>Aisle Safety Cushion (Spoilage vs. Stockout Balance)</h3>
              <p className="hint">
                Select an aisle to inspect how its calibrated demand cushion (
                <code>q50–q95</code>) avoids rot on fresh items while keeping packaged goods stocked.
              </p>

              <div className="pill-row">
                {pipeline.departments.map((d) => (
                  <button
                    key={d.dept_id}
                    type="button"
                    className={`chip ${d.dept_id === selectedDept ? "active" : ""}`}
                    onClick={() => setSelectedDept(d.dept_id)}
                  >
                    {d.label} (q{Math.round(d.waste_aware_cap * 100)})
                  </button>
                ))}
              </div>

              {activeDept ? (
                <div className="ladder-stage">
                  <div className="ladder-track">
                    <div
                      className="ladder-marker"
                      style={{
                        left: `${capPct}%`,
                        transform: `translateX(-${Math.round(capPct)}%)`,
                      }}
                    >
                      Safety Cushion q{Math.round(activeDept.waste_aware_cap * 100)}
                    </div>
                  </div>

                  <div className="ladder-bars">
                    {LADDER_KNOTS.map((k) => {
                      const val = activeDept.quantile_means[k.key] ?? 0;
                      const h = maxQuantile > 0 ? val / maxQuantile : 0;
                      const isNearCap = Math.abs(k.q - activeDept.waste_aware_cap) <= 0.05;
                      return (
                        <div key={k.key} className={`ladder-col ${isNearCap ? "near" : ""}`}>
                          <span className="num">{val.toFixed(2)}</span>
                          <div className="col-track">
                            <div
                              className="col-fill"
                              style={{ transform: `scaleY(${Math.max(0.08, h)})` }}
                            />
                          </div>
                          <span>{k.key}</span>
                        </div>
                      );
                    })}
                  </div>

                  <div className="ruled-metrics two">
                    <div className="ruled-item">
                      <span>Avg ML Order vs. Clerk Rule (EWMA)</span>
                      <strong className="num">
                        {activeDept.mean_ml_stock.toFixed(2)} vs.{" "}
                        {activeDept.mean_human_stock.toFixed(2)} units/day
                      </strong>
                    </div>
                    <div className="ruled-item">
                      <span>Aisle Spoilage vs. Stockout Balance</span>
                      <strong className="num pos">
                        q{Math.round(activeDept.waste_aware_cap * 100)} (
                        {pct(activeDept.spoilage_rate * 100)} daily spoilage)
                      </strong>
                    </div>
                  </div>
                </div>
              ) : null}

              <div className="feature-chart-wrap">
                <h3>Strongest Learned Store Signals</h3>
                <p className="hint">Top features ranked by split gain inside LightGBM.</p>
                <FeatureChart rows={importance} caption={cvCaption} />
              </div>
            </div>
          </div>

          <div className="stage-footer">
            <button type="button" className="btn btn-secondary" onClick={() => goToStage(2)}>
              Back to Step 2
            </button>
            <button type="button" className="btn btn-primary" onClick={() => goToStage(4)}>
              <span>Step 4: Why Each Aisle Wins</span>
              <IconArrowRight />
            </button>
          </div>
        </div>
      ) : null}

      {/* STAGE 4: FOCUSED OPERATIONAL TAKEAWAYS (WITHOUT REPEATING TOP KPI STRIP) */}
      {stage === 4 ? (
        <div
          id="stage-panel-4"
          role="tabpanel"
          aria-labelledby="stage-tab-4"
          className="stage-body"
          key="stage-4"
        >
          <h3>How Department Rules Created the {summary.date_start} → {summary.date_end} Lift</h3>
          <p className="lead-text stage-lead">
            Instead of applying one uniform rule across {pipeline.skus.toLocaleString()} items, each
            aisle operates at its own spoilage-to-stockout balance against the clerk baseline:
          </p>

          <div className="aisle-tabs static-tabs">
            {pipeline.departments.map((d) => {
              const wasteCutPct =
                d.human_waste_cost > 0 ? Math.abs((d.waste_saved / d.human_waste_cost) * 100) : 0;
              return (
                <div key={d.dept_id} className="aisle-tab">
                  <div className="aisle-tab-head">
                    <strong>{d.label}</strong>
                    <span className="tag">Cushion q{Math.round(d.waste_aware_cap * 100)}</span>
                  </div>
                  <p className="body-copy takeaway-copy">
                    {d.waste_saved >= 0
                      ? `Cut spoilage by ${money(d.waste_saved)} (${wasteCutPct.toFixed(1)}% lower shrink) by ordering ${d.mean_ml_stock.toFixed(2)} vs. ${d.mean_human_stock.toFixed(2)} clerk units/day.`
                      : `Increased shelf availability from ${d.mean_human_stock.toFixed(2)} to ${d.mean_ml_stock.toFixed(2)} units/day, capturing +${money(d.profit_lift)} additional net profit.`}
                  </p>
                </div>
              );
            })}
          </div>

          <div className="stage-footer">
            <div className="action-row no-margin">
              <button type="button" className="btn btn-secondary" onClick={() => goToStage(3)}>
                Back to Step 3
              </button>
              <button
                type="button"
                className="btn btn-secondary"
                onClick={() => onOpenDeskTab("simulator")}
              >
                Open What-If Spoilage Simulator
              </button>
            </div>
            <button
              type="button"
              className="btn btn-primary"
              onClick={() => onOpenDeskTab("orders")}
            >
              <span>Inspect Order Sheet &amp; Export CSV</span>
              <IconArrowRight />
            </button>
          </div>
        </div>
      ) : null}
    </section>
  );
});
