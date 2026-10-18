export function MetricNotes() {
  return (
    <section className="panel metric-notes">
      <h2>How to read these numbers</h2>
      <ul>
        <li>
          <strong>Baseline</strong> is a stronger historical heuristic (EWMA, rolling
          averages, seasonal lag, recent peak, aisle buffers). It is still rule-based,
          not a learned model.
        </li>
        <li>
          <strong>Profit lift</strong> is ML profit minus baseline profit on the same
          holdout days, after revenue, COGS, and stockout penalties.
        </li>
        <li>
          <strong>Perfect-foresight profit</strong> is a ceiling: stock exactly
          tomorrow&apos;s demand. The gap from ML to that number is residual forecast
          error.
        </li>
      </ul>
    </section>
  );
}
