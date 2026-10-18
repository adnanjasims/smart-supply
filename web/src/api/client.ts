import type {
  ChatMessage,
  ChatResponse,
  DailyPayload,
  DeptRow,
  FeatureRow,
  Health,
  ItemRow,
  Summary,
} from "./types";

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(path);
  if (!res.ok) {
    const body = await res.text();
    throw new Error(body || `${res.status} ${res.statusText}`);
  }
  return res.json() as Promise<T>;
}

export function fetchHealth() {
  return getJson<Health>("/api/health");
}

export function fetchSummary() {
  return getJson<Summary>("/api/summary");
}

export function fetchDaily() {
  return getJson<DailyPayload>("/api/daily");
}

export function fetchDepartments() {
  return getJson<{ departments: DeptRow[] }>("/api/departments");
}

export function fetchFeatures() {
  return getJson<{ importance: FeatureRow[]; cv_caption: string | null }>("/api/features");
}

export function fetchItems(params: { item_id?: string; start?: string; end?: string; limit?: number }) {
  const q = new URLSearchParams();
  if (params.item_id) q.set("item_id", params.item_id);
  if (params.start) q.set("start", params.start);
  if (params.end) q.set("end", params.end);
  if (params.limit) q.set("limit", String(params.limit));
  const qs = q.toString();
  return getJson<{ rows: ItemRow[]; returned: number }>(`/api/items${qs ? `?${qs}` : ""}`);
}

export async function postChat(messages: ChatMessage[], deptId: string): Promise<ChatResponse> {
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages, dept_id: deptId }),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail || data);
    throw new Error(detail || `${res.status} ${res.statusText}`);
  }
  return data as ChatResponse;
}
