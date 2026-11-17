import { useEffect, useState } from "react";
import type { FormEvent } from "react";
import { postChat } from "../api/client";
import type { ChatMessage } from "../api/types";
import { IconClose } from "./Icons";

const SUGGESTIONS = [
  "How much profit did the ML model make vs the clerk baseline?",
  "Which department saved the most food spoilage?",
  "Show the top stockout SKUs in Packaged Foods.",
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

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;

  async function ask(text: string) {
    const trimmed = text.trim();
    if (!trimmed || busy || !chatEnabled) return;

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
    <aside className="chat-shell" aria-label="Ask Assistant">
      <div className="chat-head">
        <strong>Ask Assistant</strong>
        <button type="button" onClick={onClose} aria-label="Close assistant">
          <IconClose size={16} />
        </button>
      </div>
      <div className="chat-body">
        <div className="chat-filters">
          <div className="field">
            <label htmlFor="chat-dept">Filter by aisle</label>
            <select id="chat-dept" value={dept} onChange={(e) => setDept(e.target.value)}>
              <option value="(all)">Whole store</option>
              <option value="FOODS_1">Produce (FOODS_1)</option>
              <option value="FOODS_2">Dairy &amp; Deli (FOODS_2)</option>
              <option value="FOODS_3">Packaged Foods (FOODS_3)</option>
            </select>
          </div>
        </div>
        <div className="chat-messages">
          {!chatEnabled ? (
            <div className="banner">
              <strong>Assistant unavailable.</strong> Natural-language Q&amp;A is currently
              disabled because no Gemini API key is configured on the host. All dashboard tables
              and simulators remain active.
            </div>
          ) : messages.length === 0 ? (
            <p className="muted">
              Ask about store profit, spoilage savings, or any SKU. Replies query the verified
              60-day simulation ledger directly.
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
        {chatEnabled && messages.length === 0 ? (
          <div className="suggests">
            {SUGGESTIONS.map((s) => (
              <button key={s} type="button" onClick={() => void ask(s)} disabled={busy}>
                {s}
              </button>
            ))}
          </div>
        ) : null}
        <form className="chat-form" onSubmit={onSubmit}>
          <label htmlFor="chat-input" className="sr-only">
            Ask a store inventory question
          </label>
          <input
            id="chat-input"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={chatEnabled ? "e.g. which dept saved the most spoilage?" : "Chat disabled"}
            disabled={busy || !chatEnabled}
          />
          <button type="submit" disabled={busy || !chatEnabled || !input.trim()}>
            {busy ? "..." : "Ask"}
          </button>
        </form>
      </div>
    </aside>
  );
}
