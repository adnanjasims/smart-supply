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

export function FeatureChart({
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
          <BarChart data={data} layout="vertical" margin={{ top: 4, right: 12, left: 8, bottom: 0 }}>
            <CartesianGrid stroke="#d5cdb8" strokeDasharray="3 3" />
            <XAxis type="number" tick={{ fontSize: 11 }} />
            <YAxis type="category" dataKey="feature" width={120} tick={{ fontSize: 11 }} />
            <Tooltip />
            <Bar dataKey="gain" fill="#3f6b4e" />
          </BarChart>
        </ResponsiveContainer>
      </div>
      {caption ? <p className="muted">{caption}</p> : null}
    </>
  );
}
