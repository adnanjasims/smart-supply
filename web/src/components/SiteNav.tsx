const LINKS = [
  { id: "about", label: "About" },
  { id: "dataset", label: "Dataset" },
  { id: "method", label: "Method" },
  { id: "results", label: "Results" },
  { id: "explore", label: "Explore" },
] as const;

export function SiteNav({ onAsk }: { onAsk: () => void }) {
  function jump(id: string) {
    const el = document.getElementById(id);
    if (el) el.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  return (
    <nav className="site-nav" aria-label="Primary">
      <button type="button" className="nav-brand" onClick={() => jump("about")}>
        Smart Supply
      </button>
      <ul className="nav-links">
        {LINKS.map((link) => (
          <li key={link.id}>
            <button type="button" onClick={() => jump(link.id)}>
              {link.label}
            </button>
          </li>
        ))}
      </ul>
      <button type="button" className="btn btn-primary nav-ask" onClick={onAsk}>
        Ask the data
      </button>
    </nav>
  );
}
