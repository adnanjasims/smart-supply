import { memo } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { FeatureRow } from "../api/types";

export const FeatureChart = memo(function FeatureChart({
  rows,
  caption,
}: {
  rows: FeatureRow[];
  caption: string | null;
}) {
  if (!rows.length) {
    return <p className="muted">No model bundle found. Run grocery.py to export feature gains.</p>;
  }

  const data = [...rows].reverse();

  return (
    <>
      <div className="chart-wrap">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} layout="vertical" margin={{ top: 4, right: 12, left: 4, bottom: 0 }}>
            <CartesianGrid stroke="#e2e6e3" strokeDasharray="3 3" />
            <XAxis type="number" tick={{ fontSize: 11, fill: "#4b5550" }} />
            <YAxis type="category" dataKey="feature" width={116} tick={{ fontSize: 11, fill: "#18201b" }} />
            <Tooltip />
            <Bar dataKey="gain" fill="#166534" radius={[0, 4, 4, 0]} isAnimationActive={false} />
          </BarChart>
        </ResponsiveContainer>
      </div>
      {caption ? <p className="muted">{caption}</p> : null}
    </>
  );
});
