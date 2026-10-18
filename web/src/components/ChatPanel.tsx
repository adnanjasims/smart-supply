import { useState } from "react";
import type { FormEvent } from "react";
import { postChat } from "../api/client";
import type { ChatMessage } from "../api/types";

const SUGGESTIONS = [
  "How much profit did the ML model make vs the human baseline?",
  "Which department benefited most from ML ordering?",
  "Show the worst stockout items.",
];

export function ChatPanel({
  open,
  onClose,
  chatEnabled,
}: {
  open: boolean;
  onClose: () => void;
  chatEnabled: boolean;
}) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [dept, setDept] = useState("(all)");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [meta, setMeta] = useState("");

  if (!open) return null;

  async function ask(text: string) {
    const trimmed = text.trim();
    if (!trimmed || busy) return;
    if (!chatEnabled) {
      setError("Set GEMINI_API_KEY in .env on the API server, then restart uvicorn.");
      return;
    }

    const next: ChatMessage[] = [...messages, { role: "user", content: trimmed }];
    setMessages(next);
    setInput("");
    setBusy(true);
    setError("");
    setMeta("");
    try {
      const res = await postChat(next, dept);
      setMessages([...next, { role: "assistant", content: res.reply }]);
      const bits = [res.model, (res.tools_used || []).join(", ")].filter(Boolean);
      setMeta(bits.join(" | "));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  function onSubmit(e: FormEvent) {
    e.preventDefault();
    void ask(input);
  }

  return (
    <aside className="chat-shell" aria-label="Ask the data">
      <div className="chat-head">
        <strong>Ask the data</strong>
        <button type="button" onClick={onClose} aria-label="Close chat">
          close
        </button>
      </div>
      <div className="chat-body">
        <div className="filters" style={{ padding: "0.65rem 0.85rem 0", marginBottom: 0 }}>
          <div className="field">
            <label htmlFor="chat-dept">Filter by aisle</label>
            <select id="chat-dept" value={dept} onChange={(e) => setDept(e.target.value)}>
              <option value="(all)">Whole store</option>
              <option value="FOODS_1">Produce (fresh)</option>
              <option value="FOODS_2">Dairy and deli</option>
              <option value="FOODS_3">Packaged foods</option>
            </select>
          </div>
        </div>
        <div className="chat-messages">
          {messages.length === 0 ? (
            <p className="muted">
              Ask about profit, stockouts, or a product id. Answers come from the holdout
              run, not guesses. Use the aisle filter to narrow the question.
            </p>
          ) : null}
          {messages.map((m, i) => (
            <div key={`${m.role}-${i}`} className={`bubble ${m.role}`}>
              {m.content}
            </div>
          ))}
          {meta ? <div className="chat-meta">{meta}</div> : null}
          {error ? <div className="banner error">{error}</div> : null}
        </div>
        {messages.length === 0 ? (
          <div className="suggests">
            {SUGGESTIONS.map((s) => (
              <button key={s} type="button" onClick={() => void ask(s)} disabled={busy}>
                {s}
              </button>
            ))}
          </div>
        ) : null}
        <form className="chat-form" onSubmit={onSubmit}>
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="e.g. which dept had the biggest profit lift?"
            disabled={busy}
          />
          <button type="submit" disabled={busy || !input.trim()}>
            {busy ? "..." : "Ask"}
          </button>
        </form>
      </div>
    </aside>
  );
}
