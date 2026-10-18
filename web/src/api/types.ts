export type Summary = {
  date_start: string;
  date_end: string;
  items: number;
  rows: number;
  ml_profit: number;
  human_profit: number;
  profit_lift: number;
  oracle_profit: number;
  ml_waste_cost: number;
  ml_stockout_cost: number;
  ml_stockout_rate_pct: number;
  human_waste_cost?: number;
  waste_saved?: number;
};

export type DailyPoint = {
  date: string;
  ml_profit: number;
  human_profit: number;
};

export type DemandPoint = {
  date: string;
  actual: number;
  predicted: number;
  q75: number;
  q90: number;
};

export type DailyPayload = {
  profit: DailyPoint[];
  demand: DemandPoint[];
};

export type DeptRow = {
  dept_id: string;
  ml_profit: number;
  human_profit: number;
  profit_lift: number;
  waste_cost: number;
  stockout_cost: number;
  stockout_rate_pct: number;
};

export type ItemRow = {
  date: string;
  item_id: string;
  dept_id: string;
  sales: number;
  pred_demand: number;
  pred_q75: number;
  pred_q90: number;
  stock_decision: number;
  waste: number;
  missed_sales: number;
  net_profit: number;
  sma_profit: number;
};

export type FeatureRow = {
  feature: string;
  gain: number;
};

export type Health = {
  ok: boolean;
  data_ready: boolean;
  detail: string;
  chat_enabled: boolean;
};

export type ChatMessage = {
  role: "user" | "assistant";
  content: string;
};

export type ChatResponse = {
  ok: boolean;
  reply: string;
  model?: string;
  tools_used?: string[];
};
