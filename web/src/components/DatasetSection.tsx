export function DatasetSection() {
  return (
    <section id="dataset" className="panel section-block">
      <h2>Dataset</h2>
      <p className="hint">
        Public Walmart M5 Forecasting Accuracy competition data, filtered to one store
        and the FOODS category for this run.
      </p>
      <ul className="plain-list">
        <li>
          <strong>sales_train_validation.csv</strong>: daily unit sales by item, store,
          and department over the competition calendar.
        </li>
        <li>
          <strong>calendar.csv</strong>: retail dates, holidays/events, and SNAP
          (food-assistance) flags that shift demand.
        </li>
        <li>
          <strong>sell_prices.csv</strong>: weekly store-level prices so the model can
          learn price sensitivity.
        </li>
      </ul>
      <p className="muted">
        Source:{" "}
        <a
          href="https://www.kaggle.com/competitions/m5-forecasting-accuracy/data"
          target="_blank"
          rel="noreferrer"
        >
          Kaggle M5 Forecasting Accuracy
        </a>
        .
      </p>
    </section>
  );
}
