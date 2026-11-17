import { IconMessage } from "./Icons";

export type TopView = "desk" | "pipeline";

const VIEWS: { id: TopView; label: string }[] = [
  { id: "desk", label: "Buyer Desk" },
  { id: "pipeline", label: "Model Walkthrough" },
];

export function SiteNav({
  activeView,
  onSelectView,
  onToggleChat,
  chatOpen,
  chatEnabled,
}: {
  activeView: TopView;
  onSelectView: (view: TopView) => void;
  onToggleChat: () => void;
  chatOpen: boolean;
  chatEnabled: boolean;
}) {
  function onTabKeyDown(e: React.KeyboardEvent<HTMLButtonElement>, idx: number) {
    if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
      e.preventDefault();
      const next = e.key === "ArrowRight" ? (idx + 1) % VIEWS.length : (idx - 1 + VIEWS.length) % VIEWS.length;
      onSelectView(VIEWS[next].id);
    }
  }

  return (
    <header className="site-nav">
      <a
        className="brand"
        href="#top"
        onClick={(e) => {
          e.preventDefault();
          window.scrollTo({ top: 0, behavior: "smooth" });
        }}
      >
        <span className="brand-mark">SmartSupply</span>
        <span className="brand-sub">Store CA_1 Replenishment</span>
      </a>

      <nav className="nav-tabs" role="tablist" aria-label="Workspace views">
        {VIEWS.map((view, idx) => {
          const isSelected = activeView === view.id;
          return (
            <button
              key={view.id}
              id={`top-tab-${view.id}`}
              type="button"
              role="tab"
              aria-selected={isSelected}
              aria-controls={`top-panel-${view.id}`}
              tabIndex={isSelected ? 0 : -1}
              className={`nav-tab ${isSelected ? "active" : ""}`}
              onClick={() => onSelectView(view.id)}
              onKeyDown={(e) => onTabKeyDown(e, idx)}
            >
              {view.label}
            </button>
          );
        })}
      </nav>

      <button
        type="button"
        className={`btn ${chatOpen ? "btn-secondary" : "btn-primary"}`}
        onClick={onToggleChat}
        title={chatEnabled ? "Open ledger Q&A assistant" : "Assistant status"}
      >
        <IconMessage size={15} />
        <span>{chatOpen ? "Close Assistant" : "Ask Assistant"}</span>
      </button>
    </header>
  );
}
