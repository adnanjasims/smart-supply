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
import type { DemandPoint } from "../api/types";

export function DemandChart({ data }: { data: DemandPoint[] }) {
  return (
    <div className="chart-wrap">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
          <CartesianGrid stroke="#d5cdb8" strokeDasharray="3 3" />
          <XAxis dataKey="date" tick={{ fontSize: 11 }} minTickGap={28} />
          <YAxis tick={{ fontSize: 11 }} width={56} />
          <Tooltip />
          <Legend />
          <Line type="monotone" dataKey="actual" name="Actual" stroke="#1a2420" strokeWidth={2} dot={false} />
          <Line type="monotone" dataKey="predicted" name="Predicted" stroke="#2a6f7c" strokeWidth={2} dot={false} />
          <Line type="monotone" dataKey="q75" name="q75" stroke="#3f6b4e" strokeDasharray="4 3" dot={false} />
          <Line type="monotone" dataKey="q90" name="q90" stroke="#c45c26" strokeDasharray="4 3" dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
