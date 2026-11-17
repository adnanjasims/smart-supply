export interface Health {
  ok: boolean;
  data_ready: boolean;
  detail: string;
  chat_enabled: boolean;
}

export interface Summary {
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
  ml_waste_units?: number;
  human_waste_units?: number;
  waste_units_saved?: number;
}

export interface DailyPoint {
  date: string;
  ml_profit: number;
  human_profit: number;
}

export interface DemandPoint {
  date: string;
  actual: number;
  predicted: number;
  q75: number;
  q90: number;
}

export interface DailyResponse {
  profit: DailyPoint[];
  demand: DemandPoint[];
}

export interface DeptRow {
  dept_id: string;
  ml_profit: number;
  human_profit: number;
  profit_lift: number;
  waste_cost: number;
  human_waste_cost?: number | null;
  waste_saved?: number | null;
  stockout_cost: number;
  stockout_rate_pct: number;
}

export interface ItemRow {
  date: string;
  item_id: string;
  dept_id?: string;
  sales: number;
  pred_demand: number;
  pred_q75: number;
  pred_q90: number;
  stock_decision: number;
  waste: number;
  missed_sales: number;
  net_profit: number;
  sma_profit: number;
}

export interface FeatureRow {
  feature: string;
  gain: number;
}

export interface FeaturesResponse {
  importance: FeatureRow[];
  cv_caption: string | null;
}

export interface OrderRow {
  date: string;
  item_id: string;
  dept_id: string;
  sell_price: number;
  sales: number;
  pred_demand: number;
  pred_q75: number;
  pred_q90: number;
  recommended_order: number;
  baseline_order: number;
  delta_vs_baseline: number;
  action: "Trim Overstock" | "Boost Stock" | "Hold" | string;
  waste_cost: number;
  net_profit: number;
  sma_profit: number;
}

export interface OrdersResponse {
  selected_date: string;
  dates: string[];
  summary: {
    sku_count: number;
    total_order_units: number;
    baseline_order_units: number;
    trim_count: number;
    boost_count: number;
    estimated_profit: number;
    baseline_profit: number;
    estimated_waste_cost: number;
  };
  rows: OrderRow[];
  returned: number;
}

export interface SimulateDeptRow {
  dept_id: string;
  ml_profit: number;
  human_profit: number;
  profit_lift: number;
  ml_waste_cost: number;
  human_waste_cost: number;
  waste_saved: number;
  ml_stockout_cost: number;
}

export interface SimulateResponse {
  spoilage_multiplier: number;
  stockout_penalty: number;
  dept_id: string;
  ml_profit: number;
  human_profit: number;
  profit_lift: number;
  ml_waste_cost: number;
  human_waste_cost: number;
  waste_saved: number;
  ml_waste_units: number;
  human_waste_units: number;
  ml_stockout_cost: number;
  human_stockout_cost: number;
  departments: SimulateDeptRow[];
}

export interface ItemSeriesPoint {
  date: string;
  sales: number;
  pred_demand: number;
  pred_q75: number;
  pred_q90: number;
  stock_decision: number;
  human_stock: number;
  waste: number;
  waste_cost: number;
  net_profit: number;
  sma_profit: number;
}

export interface ItemDetailResponse {
  item_id: string;
  dept_id: string;
  avg_sell_price: number;
  days: number;
  total_sales: number;
  total_ml_profit: number;
  total_human_profit: number;
  total_waste_cost: number;
  total_human_waste_cost: number | null;
  stockout_days: number;
  series: ItemSeriesPoint[];
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface PipelineDept {
  dept_id: string;
  label: string;
  sku_count: number;
  cogs_ratio: number;
  spoilage_rate: number;
  stockout_penalty_fraction: number;
  theoretical_cf: number;
  waste_aware_cap: number;
  avg_sell_price: number;
  unit_margin: number;
  unit_spoilage_cost: number;
  unit_stockout_cost: number;
  zero_sale_pct: number;
  avg_daily_sales: number;
  mean_pred_demand: number;
  mean_ml_stock: number;
  mean_human_stock: number;
  ml_profit: number;
  human_profit: number;
  profit_lift: number;
  ml_waste_cost: number;
  human_waste_cost: number;
  waste_saved: number;
  quantile_means: Record<string, number>;
}

export interface PipelineResponse {
  store_id: string;
  category: string;
  skus: number;
  holdout_days: number;
  holdout_rows: number;
  zero_sale_pct: number;
  holdout_mae: number;
  cleaning: {
    lookback_days: number;
    raw_rows: number;
    warmup_dropped: number;
    missing_price_dropped: number;
    kept_rows: number;
  };
  feature_count: number;
  feature_groups: {
    group: string;
    count: number;
    examples: string[];
  }[];
  cv_scores: number[];
  cv_mae_mean: number | null;
  cv_mae_std: number | null;
  quantile_ladder: string[];
  best_params: {
    learning_rate: number | null;
    num_leaves: number | null;
    min_data_in_leaf: number | null;
  };
  departments: PipelineDept[];
}

export interface ChatResponse {
  ok: boolean;
  reply: string;
  model: string;
  tools_used: string[];
}
