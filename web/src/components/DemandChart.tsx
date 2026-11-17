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
import type { DemandPoint } from "../api/types";

export const DemandChart = memo(function DemandChart({ data }: { data: DemandPoint[] }) {
  return (
    <div className="chart-wrap">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
          <CartesianGrid stroke="#e2e6e3" strokeDasharray="3 3" />
          <XAxis dataKey="date" tick={{ fontSize: 11, fill: "#4b5550" }} minTickGap={28} />
          <YAxis tick={{ fontSize: 11, fill: "#4b5550" }} width={56} />
          <Tooltip />
          <Legend />
          <Line type="monotone" dataKey="actual" name="Actual" stroke="#18201b" strokeWidth={2} dot={false} isAnimationActive={false} />
          <Line type="monotone" dataKey="predicted" name="Expected (Mean)" stroke="#1d4ed8" strokeWidth={2} dot={false} isAnimationActive={false} />
          <Line type="monotone" dataKey="q75" name="q75 Cushion" stroke="#166534" strokeDasharray="4 3" dot={false} isAnimationActive={false} />
          <Line type="monotone" dataKey="q90" name="q90 Cushion" stroke="#b42318" strokeDasharray="4 3" dot={false} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
});
