export function ProjectIntro() {
  return (
    <section id="about" className="intro" aria-label="About Smart Supply">
      <div
        className="intro-visual"
        style={{ backgroundImage: "url(/intro/produce-aisle.jpg)" }}
        role="img"
        aria-label="Fresh produce display"
      />
      <div className="intro-copy">
        <p className="intro-lede">
          Millions of tons of surplus food leave the U.S. retail channel every year.
          ReFED puts 2024 retail surplus near 4.6 million tons (about $30 billion in
          product value). Overstock and empty shelves both hit the P&amp;L: one burns
          margin on product that never sells, the other loses the sale.
        </p>
        <p>
          Smart Supply tests whether a learned demand model can order better than a
          simple historical average. It trains a LightGBM forecaster on Walmart&apos;s
          public M5 competition data (one store, FOODS category), then simulates sixty
          days of stocking decisions and scores them on profit after cost of goods and
          missed-sale penalties.
        </p>
        <p>
          Use the nav to jump to the dataset, method, holdout results, and item-level
          explore table. Ask the data opens a Gemini Q&amp;A panel over the same
          holdout CSV.
        </p>
        <p className="intro-credit muted">
          Photo: NCI Visuals Online via Wikimedia Commons. Surplus figures: ReFED
          retail sector estimates.
        </p>
      </div>
    </section>
  );
}
