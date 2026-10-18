"""
Thin Gemini wrapper: question in, tool calls against the sim CSV, answer out.

Set GEMINI_API_KEY (or GOOGLE_API_KEY) in .env. No key => chat tab stays quiet,
dashboard still runs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .tools import TOOL_SCHEMAS, ChatScope, execute_tool, tool_run_summary

ROOT = Path(__file__).resolve().parent.parent
MAX_TOOL_ROUNDS = 4
MAX_HISTORY = 12

SUGGESTIONS = [
    "How much profit did the ML model make vs the human baseline?",
    "Which department benefited most from ML ordering?",
    "Show the worst stockout items.",
    "What was the gap to perfect-foresight profit?",
]


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(ROOT / ".env", override=True)


def gemini_api_key() -> str:
    _load_env()
    return (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip()


def gemini_model() -> str:
    return (os.environ.get("GEMINI_MODEL") or "gemini-2.5-flash").strip()


def gemini_model_fallbacks() -> list[str]:
    # free tier sometimes has limit:0 on one model; walk the list
    primary = gemini_model()
    defaults = [primary, "gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash", "gemini-1.5-flash"]
    seen: set[str] = set()
    out: list[str] = []
    for m in defaults:
        if m and m not in seen:
            seen.add(m)
            out.append(m)
    return out


def chat_config_present() -> bool:
    return bool(gemini_api_key())


def _sanitize_history(messages: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for m in list(messages)[-MAX_HISTORY:]:
        role = str(m.get("role", "")).lower()
        content = str(m.get("content", "")).strip()
        if role not in {"user", "assistant"} or not content:
            continue
        if len(content) > 4000:
            content = content[:4000]
        out.append({"role": role, "content": content})
    return out


def _system_prompt(scope: ChatScope, preload: str) -> str:
    return (
        "You answer questions about a Smart Supply holdout run on Walmart M5 data. "
        "LightGBM forecasts demand; stocking uses a per-aisle newsvendor rule on the "
        "point / q75 / q90 ladder. Profit includes COGS and stockout penalties.\n\n"
        "How to answer:\n"
        "1. Use PRELOADED_SUMMARY for store-wide totals. Call tools for aisle splits, "
        "item ranks, a specific SKU, or the daily series.\n"
        "2. Stick to tool numbers. If a tool fails or comes back empty, say that.\n"
        "3. Write the full answer in your reply text; the UI only shows that, not tool JSON.\n"
        "4. Finish every answer completely. Do not stop mid-sentence or mid-bullet.\n"
        "5. Lead with profit, stockouts, and lift. Mention spoilage only if the user asks.\n"
        "6. FOODS_1 = produce, FOODS_2 = dairy/deli, FOODS_3 = packaged.\n"
        "7. sma_profit is the historical baseline; net_profit is the ML policy.\n\n"
        f"Active filters: dept={scope.dept_id}, "
        f"start={scope.start_date or 'holdout start'}, end={scope.end_date or 'holdout end'}.\n\n"
        "PRELOADED_SUMMARY (JSON):\n"
        + preload
    )


def _function_declarations():
    from google.genai import types

    decls = []
    for schema in TOOL_SCHEMAS:
        fn = schema["function"]
        decls.append(
            types.FunctionDeclaration(
                name=fn["name"],
                description=fn.get("description") or "",
                parameters=fn.get("parameters") or {"type": "object", "properties": {}},
            )
        )
    return types.Tool(function_declarations=decls)


def _history_to_contents(history: list[dict[str, str]]):
    from google.genai import types

    contents: list[Any] = []
    for m in history:
        role = "user" if m["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part(text=m["content"])]))
    return contents


def _retryable(err: Exception) -> bool:
    msg = str(err)
    return "429" in msg or "RESOURCE_EXHAUSTED" in msg or "limit: 0" in msg


def _generate(client, models: list[str], contents, config):
    last_err: Exception | None = None
    for model in models:
        try:
            return model, client.models.generate_content(model=model, contents=contents, config=config)
        except Exception as e:
            last_err = e
            if _retryable(e):
                continue
            raise
    if last_err:
        raise last_err
    raise RuntimeError("No Gemini models configured.")


def answer_question(
    messages: Iterable[dict[str, Any]],
    sim: pd.DataFrame,
    *,
    dept_id: str = "(all)",
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return {
            "ok": False,
            "error": (
                "google-genai is missing in this Python. From smartsupply run: "
                "source .venv/bin/activate && pip install google-genai "
                "then restart uvicorn with that same venv."
            ),
        }

    api_key = gemini_api_key()
    if not api_key:
        return {
            "ok": False,
            "error": "GEMINI_API_KEY is not set. Add it to .env (see .env.example).",
        }

    history = _sanitize_history(messages)
    if not history or history[-1]["role"] != "user":
        return {"ok": False, "error": "No user message provided."}

    scope = ChatScope(
        dept_id=str(dept_id or "(all)"),
        start_date=start_date,
        end_date=end_date,
    )
    # stash a summary in the system prompt so simple KPI questions skip a tool round
    preload = json.dumps(tool_run_summary(sim, scope), default=str)[:8000]

    client = genai.Client(api_key=api_key)
    model_candidates = gemini_model_fallbacks()
    active_model: str | None = None
    config = types.GenerateContentConfig(
        system_instruction=_system_prompt(scope, preload),
        tools=[_function_declarations()],
        temperature=0.15,
        max_output_tokens=4096,
    )

    contents = _history_to_contents(history)
    tools_used: list[str] = ["preload_summary"]

    for _ in range(MAX_TOOL_ROUNDS):
        try:
            models = [active_model] if active_model else model_candidates
            active_model, response = _generate(client, models, contents, config)
        except Exception as e:
            hint = ""
            if _retryable(e):
                hint = (
                    " Quota looks toasted for this model; "
                    "try GEMINI_MODEL=gemini-2.5-flash-lite in .env."
                )
            return {"ok": False, "error": f"Gemini request failed: {e}.{hint}"}

        if not response.candidates:
            return {"ok": False, "error": "Gemini returned no candidates."}

        candidate = response.candidates[0]
        parts = (candidate.content.parts if candidate.content else None) or []
        fn_calls = [p for p in parts if getattr(p, "function_call", None)]

        if not fn_calls:
            reply = (response.text or "").strip() or "I could not generate a reply."
            return {
                "ok": True,
                "reply": reply,
                "model": active_model or gemini_model(),
                "tools_used": list(dict.fromkeys(tools_used)),
                "suggested_followups": SUGGESTIONS[:3],
            }

        if candidate.content:
            contents.append(candidate.content)

        response_parts: list[Any] = []
        for part in fn_calls:
            fc = part.function_call
            name = fc.name or ""
            raw_args = fc.args
            if isinstance(raw_args, dict):
                args = raw_args
            else:
                try:
                    args = json.loads(str(raw_args)) if raw_args else {}
                except json.JSONDecodeError:
                    args = {}
            tools_used.append(name)
            payload_raw = execute_tool(name, args, sim, scope)
            try:
                payload = json.loads(payload_raw)
            except json.JSONDecodeError:
                payload = {"result": payload_raw}
            response_parts.append(
                types.Part(
                    function_response=types.FunctionResponse(name=name, response=payload)
                )
            )

        contents.append(types.Content(role="user", parts=response_parts))

    return {
        "ok": False,
        "error": "Hit the tool-round limit. Try a narrower question.",
        "tools_used": tools_used,
    }
