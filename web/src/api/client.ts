import type {
  ChatMessage,
  ChatResponse,
  DailyResponse,
  DeptRow,
  FeaturesResponse,
  Health,
  ItemDetailResponse,
  ItemRow,
  OrdersResponse,
  SimulateResponse,
  Summary,
} from "./types";

async function getJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = (await res.json()) as { detail?: string };
      if (body.detail) detail = body.detail;
    } catch {
      // plain text fallback
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export const fetchHealth = () => getJson<Health>("/api/health");
export const fetchSummary = () => getJson<Summary>("/api/summary");
export const fetchPipeline = () => getJson<import("./types").PipelineResponse>("/api/pipeline");
export const fetchDaily = () => getJson<DailyResponse>("/api/daily");
export const fetchDepartments = () =>
  getJson<{ departments: DeptRow[] }>("/api/departments").then((r) => r.departments);
export const fetchFeatures = () => getJson<FeaturesResponse>("/api/features");

export function fetchItems(params: {
  item_id?: string;
  start?: string;
  end?: string;
  limit?: number;
}) {
  const sp = new URLSearchParams();
  if (params.item_id) sp.set("item_id", params.item_id);
  if (params.start) sp.set("start", params.start);
  if (params.end) sp.set("end", params.end);
  if (params.limit) sp.set("limit", String(params.limit));
  return getJson<{ rows: ItemRow[]; returned: number }>(`/api/items?${sp.toString()}`);
}

export function fetchOrders(params: {
  date?: string;
  dept_id?: string;
  search?: string;
  status?: string;
  limit?: number;
}) {
  const sp = new URLSearchParams();
  if (params.date) sp.set("date", params.date);
  if (params.dept_id && params.dept_id !== "(all)") sp.set("dept_id", params.dept_id);
  if (params.search) sp.set("search", params.search);
  if (params.status && params.status !== "all") sp.set("status", params.status);
  if (params.limit) sp.set("limit", String(params.limit));
  return getJson<OrdersResponse>(`/api/orders?${sp.toString()}`);
}

export function ordersExportUrl(params: {
  date?: string;
  dept_id?: string;
  search?: string;
  status?: string;
}): string {
  const sp = new URLSearchParams();
  if (params.date) sp.set("date", params.date);
  if (params.dept_id && params.dept_id !== "(all)") sp.set("dept_id", params.dept_id);
  if (params.search) sp.set("search", params.search);
  if (params.status && params.status !== "all") sp.set("status", params.status);
  return `/api/orders/export?${sp.toString()}`;
}

export function postSimulate(params: {
  spoilage_multiplier: number;
  stockout_penalty: number;
  dept_id?: string;
}) {
  return getJson<SimulateResponse>("/api/simulate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      spoilage_multiplier: params.spoilage_multiplier,
      stockout_penalty: params.stockout_penalty,
      dept_id: params.dept_id ?? "(all)",
    }),
  });
}

export function fetchItemDetail(itemId: string) {
  return getJson<ItemDetailResponse>(`/api/item/${encodeURIComponent(itemId)}`);
}

export function postChat(messages: ChatMessage[], dept_id: string) {
  return getJson<ChatResponse>("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages, dept_id }),
  });
}
