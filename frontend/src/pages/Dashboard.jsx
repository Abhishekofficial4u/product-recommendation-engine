// frontend/src/pages/Dashboard.jsx
import { useEffect, useState } from "react";
import { api } from "../services/api";
import { Spinner, ErrorBanner, Card } from "../components/UI";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
         Legend, ResponsiveContainer, RadarChart, Radar,
         PolarGrid, PolarAngleAxis, PolarRadiusAxis } from "recharts";

const MODEL_COLORS = {
  "Popularity Baseline": "#94A3B8",
  "User-based CF":       "#3B82F6",
  "Item-based CF":       "#8B5CF6",
  "Content-Based":       "#F59E0B",
  "SVD (tuned)":         "#10B981",
  "Hybrid ★":            "#EF4444",
};

const METRIC_INFO = {
  rmse:   { label: "RMSE",          desc: "Rating prediction error (lower = better)",  lowerBetter: true  },
  p10:    { label: "Precision@10",  desc: "% of top-10 recs the user actually liked",  lowerBetter: false },
  r10:    { label: "Recall@10",     desc: "% of liked items found in top-10",          lowerBetter: false },
  ndcg10: { label: "NDCG@10",       desc: "Ranking quality — best items ranked highest",lowerBetter: false },
};

export default function Dashboard() {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error,   setError]   = useState(null);
  const [active,  setActive]  = useState("rmse");

  useEffect(() => {
    api.metrics()
      .then(d => { setMetrics(d.models); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div style={{ padding: 40 }}><Spinner/></div>;
  if (error)   return <div style={{ padding: 40 }}><ErrorBanner message={error}/></div>;
  if (!metrics) return null;

  const metricKey = active;
  const info      = METRIC_INFO[active];

  // Bar chart data for active metric
  const barData = metrics
    .filter(m => m[metricKey] != null)
    .map(m => ({ name: m.model.replace(" ★",""), value: Number(m[metricKey]) }));

  // Radar data — normalise all metrics 0→1
  const radarModels = metrics.filter(m => m.rmse != null);
  const maxRmse     = Math.max(...radarModels.map(m => m.rmse || 0));
  const maxP10      = Math.max(...metrics.map(m => m.p10 || 0));
  const maxR10      = Math.max(...metrics.map(m => m.r10 || 0));
  const radarData = [
    { subject: "Accuracy (inv RMSE)", ...Object.fromEntries(radarModels.map(m => [m.model, +(1 - m.rmse/maxRmse).toFixed(3)])) },
    { subject: "Precision@10",        ...Object.fromEntries(radarModels.map(m => [m.model, +(( m.p10||0)/maxP10).toFixed(3)]))  },
    { subject: "Recall@10",           ...Object.fromEntries(radarModels.map(m => [m.model, +(( m.r10||0)/maxR10).toFixed(3)]))  },
  ];

  return (
    <div style={{ maxWidth: 1100, margin: "0 auto", padding: "28px 20px" }}>
      <div style={{ marginBottom: 28 }}>
        <h1 style={{ fontSize: 26, fontWeight: 700, color: "#1E293B", margin: "0 0 6px" }}>
          Model Comparison Dashboard
        </h1>
        <p style={{ color: "#64748B", fontSize: 14, margin: 0 }}>
          All 6 models evaluated on the held-out test set. Lower RMSE = better rating predictions. Higher P@10 = better recommendations.
        </p>
      </div>

      {/* Metric selector tabs */}
      <div style={{ display: "flex", gap: 8, marginBottom: 20, flexWrap: "wrap" }}>
        {Object.entries(METRIC_INFO).map(([key, inf]) => (
          <button key={key} onClick={() => setActive(key)} style={{
            padding: "7px 16px", borderRadius: 8,
            border: active === key ? "1.5px solid #1B3A6B" : "1px solid #E2E8F0",
            background: active === key ? "#1B3A6B" : "#F8FAFC",
            color: active === key ? "#fff" : "#64748B",
            fontSize: 13, fontWeight: active === key ? 600 : 400,
            cursor: "pointer", transition: "all .15s"
          }}>
            {inf.label}
          </button>
        ))}
      </div>

      {/* Main bar chart */}
      <Card style={{ padding: "20px 16px", marginBottom: 20 }}>
        <div style={{ marginBottom: 12 }}>
          <div style={{ fontSize: 15, fontWeight: 600, color: "#1E293B" }}>{info.label} by Model</div>
          <div style={{ fontSize: 12, color: "#64748B", marginTop: 2 }}>{info.desc}</div>
        </div>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={barData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9"/>
            <XAxis dataKey="name" tick={{ fontSize: 11, fill: "#64748B" }} interval={0}/>
            <YAxis tick={{ fontSize: 11, fill: "#64748B" }}/>
            <Tooltip
              formatter={(v) => [v.toFixed(4), info.label]}
              contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid #E2E8F0" }}
            />
            <Bar dataKey="value" radius={[4,4,0,0]}
              fill="#3B82F6"
              label={{ position: "top", fontSize: 11, fill: "#64748B", formatter: v => v.toFixed(4) }}
            />
          </BarChart>
        </ResponsiveContainer>
        <div style={{ fontSize: 11, color: "#94A3B8", textAlign: "right", marginTop: 4 }}>
          {info.lowerBetter ? "← lower is better" : "higher is better →"}
        </div>
      </Card>

      {/* Full metrics table */}
      <Card style={{ marginBottom: 20, overflow: "hidden" }}>
        <div style={{ padding: "16px 20px 0" }}>
          <div style={{ fontSize: 15, fontWeight: 600, color: "#1E293B" }}>Full Metrics Table</div>
        </div>
        <div style={{ overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", marginTop: 12 }}>
            <thead>
              <tr style={{ background: "#F8FAFC" }}>
                {["Model","RMSE ↓","Precision@10 ↑","Recall@10 ↑","NDCG@10 ↑"].map(h => (
                  <th key={h} style={{
                    padding: "10px 16px", textAlign: h === "Model" ? "left" : "center",
                    fontSize: 12, fontWeight: 600, color: "#374151",
                    borderBottom: "1px solid #E2E8F0"
                  }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {metrics.map((m, i) => {
                const isHybrid = m.model.includes("Hybrid");
                return (
                  <tr key={m.model} style={{
                    background: isHybrid ? "#F0FDF4" : i % 2 === 0 ? "#fff" : "#FAFAFA"
                  }}>
                    <td style={{ padding: "10px 16px", fontSize: 13, fontWeight: isHybrid ? 600 : 400, color: "#1E293B" }}>
                      <span style={{ display: "flex", alignItems: "center", gap: 8 }}>
                        <span style={{ width: 12, height: 12, borderRadius: "50%", background: MODEL_COLORS[m.model] || "#94A3B8", flexShrink: 0 }}/>
                        {m.model}
                      </span>
                    </td>
                    {[m.rmse, m.p10, m.r10, m.ndcg10].map((v, j) => (
                      <td key={j} style={{
                        padding: "10px 16px", textAlign: "center",
                        fontSize: 13, color: v != null ? (isHybrid ? "#166534" : "#1E293B") : "#CBD5E1",
                        fontWeight: isHybrid ? 600 : 400,
                        borderTop: "1px solid #F1F5F9"
                      }}>
                        {v != null ? v.toFixed(4) : "—"}
                      </td>
                    ))}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Radar chart */}
      <Card style={{ padding: "20px 16px" }}>
        <div style={{ fontSize: 15, fontWeight: 600, color: "#1E293B", marginBottom: 4 }}>
          Normalised Radar Comparison
        </div>
        <div style={{ fontSize: 12, color: "#64748B", marginBottom: 16 }}>
          All metrics normalised to 0–1. Larger area = better overall model.
        </div>
        <ResponsiveContainer width="100%" height={320}>
          <RadarChart data={radarData}>
            <PolarGrid stroke="#E2E8F0"/>
            <PolarAngleAxis dataKey="subject" tick={{ fontSize: 11, fill: "#64748B" }}/>
            <PolarRadiusAxis angle={30} domain={[0,1]} tick={{ fontSize: 10, fill: "#94A3B8" }}/>
            {radarModels.map((m, i) => (
              <Radar key={m.model} name={m.model} dataKey={m.model}
                stroke={MODEL_COLORS[m.model] || "#94A3B8"}
                fill={MODEL_COLORS[m.model] || "#94A3B8"}
                fillOpacity={0.12}
              />
            ))}
            <Legend wrapperStyle={{ fontSize: 11 }}/>
            <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid #E2E8F0" }}/>
          </RadarChart>
        </ResponsiveContainer>
      </Card>
    </div>
  );
}
