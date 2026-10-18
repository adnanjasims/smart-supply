export function MethodSection() {
  return (
    <section id="method" className="panel section-block">
      <h2>Method</h2>
      <p className="hint">
        LightGBM demand forecasts drive stocking decisions; profit is scored against a
        simple historical baseline on a held-out window.
      </p>
      <ul className="plain-list">
        <li>
          <strong>Model.</strong> Poisson LightGBM point forecast, plus q75 / q90
          quantile models for uncertainty. Hyperparameters tuned with Optuna when
          available.
        </li>
        <li>
          <strong>Features.</strong> Lag and rolling sales, sell price, SNAP flags,
          holiday/event markers, and Fourier seasonality.
        </li>
        <li>
          <strong>Stocking.</strong> Per-department newsvendor critical fractile on the
          point → q75 → q90 ladder (tighter for perishables, wider for shelf-stable).
        </li>
        <li>
          <strong>Baseline.</strong> A stronger clerk heuristic: EWMA, 7/28-day
          rolling means, last-week seasonal, and recent peak, with aisle-specific
          buffers. Same cost model so the comparison is fair.
        </li>
        <li>
          <strong>Evaluation.</strong> Expanding-window CV by calendar date, then a
          ~60-day holdout simulation for the numbers below.
        </li>
      </ul>
    </section>
  );
}