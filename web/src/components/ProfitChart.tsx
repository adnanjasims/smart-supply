import { memo } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { DailyPoint } from "../api/types";

export const ProfitChart = memo(function ProfitChart({ data }: { data: DailyPoint[] }) {
  return (
    <div className="chart-wrap">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
          <CartesianGrid stroke="#e2e6e3" strokeDasharray="3 3" />
          <XAxis dataKey="date" tick={{ fontSize: 11, fill: "#4b5550" }} minTickGap={28} />
          <YAxis tick={{ fontSize: 11, fill: "#4b5550" }} width={56} />
          <Tooltip />
          <Legend />
          <Line type="monotone" dataKey="ml_profit" name="SmartSupply ML" stroke="#166534" strokeWidth={2.2} dot={false} isAnimationActive={false} />
          <Line type="monotone" dataKey="human_profit" name="Clerk Rule (EWMA)" stroke="#4b5550" strokeWidth={1.8} dot={false} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
});
