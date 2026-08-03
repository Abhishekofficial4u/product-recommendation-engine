// frontend/src/pages/Analytics.jsx
import React, { useEffect, useState } from "react";
import { api } from "../services/api";
import { Spinner, ErrorBanner, Card } from "../components/UI";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
         ResponsiveContainer, Cell, PieChart, Pie, Legend } from "recharts";

const COLORS = ["#3B82F6", "#8B5CF6", "#10B981", "#F59E0B", "#EF4444", "#94A3B8"];

export default function Analytics() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    api.analyticsSummary = () => {
      // Direct call to fetch from service
      const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
      const token = localStorage.getItem("token");
      const headers = { "Content-Type": "application/json" };
      if (token) headers["Authorization"] = `Bearer ${token}`;
      return fetch(`${API_BASE_URL}/analytics/summary`, { headers })
        .then(res => {
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          return res.json();
        });
    };

    api.analyticsSummary()
      .then(d => { setData(d); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div style={{ padding: 40 }}><Spinner/></div>;
  if (error) return <div style={{ padding: 40 }}><ErrorBanner message={error}/></div>;
  if (!data) return null;

  const pieData = data.model_usage.map(item => ({
    name: item.model_used,
    value: item.count
  }));

  const chartData = data.model_usage.map(item => ({
    model: item.model_used,
    queries: item.count
  }));

  return (
    <div style={{ maxWidth: 1100, margin: "0 auto", padding: "28px 20px" }}>
      <div style={{ marginBottom: 28 }}>
        <h1 style={{ fontSize: 26, fontWeight: 700, color: "#1E293B", margin: "0 0 6px" }}>
          System Telemetry & Telemetry Analytics
        </h1>
        <p style={{ color: "#64748B", fontSize: 14, margin: 0 }}>
          Real-time analytics aggregating recommendation queries, system latency, and model popularity.
        </p>
      </div>

      {/* Stats Cards */}
      <div style={{ display: "flex", gap: 20, marginBottom: 24, flexWrap: "wrap" }}>
        <Card style={{ flex: "1 1 240px", padding: 20, borderLeft: "4px solid #3B82F6" }}>
          <div style={{ fontSize: 13, color: "#64748B", fontWeight: 500 }}>Total API Requests</div>
          <div style={{ fontSize: 28, fontWeight: 700, color: "#1E293B", marginTop: 4 }}>
            {data.total_requests.toLocaleString()}
          </div>
        </Card>
        <Card style={{ flex: "1 1 240px", padding: 20, borderLeft: "4px solid #10B981" }}>
          <div style={{ fontSize: 13, color: "#64748B", fontWeight: 500 }}>Total Telemetry Ratings</div>
          <div style={{ fontSize: 28, fontWeight: 700, color: "#1E293B", marginTop: 4 }}>
            {data.total_ratings.toLocaleString()}
          </div>
        </Card>
        <Card style={{ flex: "1 1 240px", padding: 20, borderLeft: "4px solid #F59E0B" }}>
          <div style={{ fontSize: 13, color: "#64748B", fontWeight: 500 }}>Average API Latency</div>
          <div style={{ fontSize: 28, fontWeight: 700, color: "#1E293B", marginTop: 4 }}>
            {data.avg_response_ms != null ? `${data.avg_response_ms} ms` : "—"}
          </div>
        </Card>
      </div>

      <div style={{ display: "flex", gap: 20, marginBottom: 24, flexWrap: "wrap" }}>
        {/* Model distribution chart */}
        <Card style={{ flex: "1 1 500px", padding: "20px 16px" }}>
          <div style={{ fontSize: 15, fontWeight: 600, color: "#1E293B", marginBottom: 12 }}>
            API Usage by Recommendation Model
          </div>
          {chartData.length === 0 ? (
            <div style={{ textAlign: "center", padding: 40, color: "#94A3B8", fontSize: 14 }}>
              No request logs generated yet. Run recommendations to see charts!
            </div>
          ) : (
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" />
                <XAxis dataKey="model" tick={{ fontSize: 11, fill: "#64748B" }} />
                <YAxis tick={{ fontSize: 11, fill: "#64748B" }} />
                <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid #E2E8F0" }} />
                <Bar dataKey="queries" radius={[4, 4, 0, 0]}>
                  {chartData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          )}
        </Card>

        {/* Model distribution pie */}
        <Card style={{ flex: "1 1 340px", padding: "20px 16px", display: "flex", flexDirection: "column", alignItems: "center" }}>
          <div style={{ fontSize: 15, fontWeight: 600, color: "#1E293B", marginBottom: 12, alignSelf: "flex-start" }}>
            Model Distribution Share
          </div>
          {pieData.length === 0 ? (
            <div style={{ textAlign: "center", padding: 40, color: "#94A3B8", fontSize: 14 }}>
              No data.
            </div>
          ) : (
            <ResponsiveContainer width="100%" height={280}>
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={80}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {pieData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip formatter={(v) => [`${v} requests`, "Usage"]} />
                <Legend layout="horizontal" verticalAlign="bottom" align="center" wrapperStyle={{ fontSize: 11 }} />
              </PieChart>
            </ResponsiveContainer>
          )}
        </Card>
      </div>

      {/* Recent requests logs table */}
      <Card style={{ padding: "20px 0", overflow: "hidden" }}>
        <div style={{ padding: "0 20px 12px" }}>
          <div style={{ fontSize: 15, fontWeight: 600, color: "#1E293B" }}>Recent Server Request Logs (Last 10)</div>
        </div>
        <div style={{ overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse" }}>
            <thead>
              <tr style={{ background: "#F8FAFC" }}>
                {["Request ID", "User ID", "Model Used", "Top-K", "Response Time", "Timestamp"].map(h => (
                  <th key={h} style={{
                    padding: "10px 20px", textAlign: "left",
                    fontSize: 12, fontWeight: 600, color: "#475569",
                    borderBottom: "1px solid #E2E8F0"
                  }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.recent_requests.length === 0 ? (
                <tr>
                  <td colSpan={6} style={{ padding: "30px 20px", textAlign: "center", color: "#94A3B8", fontSize: 13 }}>
                    No recent recommendation logs found in database.
                  </td>
                </tr>
              ) : (
                data.recent_requests.map((r, i) => (
                  <tr key={r.id} style={{ background: i % 2 === 0 ? "#fff" : "#FAFAFA" }}>
                    <td style={{ padding: "10px 20px", fontSize: 13, color: "#1E293B" }}>#{r.id}</td>
                    <td style={{ padding: "10px 20px", fontSize: 13, color: "#1E293B" }}>User {r.user_id}</td>
                    <td style={{ padding: "10px 20px", fontSize: 13, color: "#1E293B" }}>
                      <span style={{
                        padding: "3px 8px", borderRadius: 4, fontSize: 11, fontWeight: 500,
                        background: r.model_used === "hybrid" ? "#F0FDF4" : "#F1F5F9",
                        color: r.model_used === "hybrid" ? "#166534" : "#475569"
                      }}>
                        {r.model_used}
                      </span>
                    </td>
                    <td style={{ padding: "10px 20px", fontSize: 13, color: "#1E293B" }}>{r.top_k} items</td>
                    <td style={{ padding: "10px 20px", fontSize: 13, color: "#1E293B", fontWeight: 600 }}>
                      {r.response_ms != null ? `${r.response_ms} ms` : "—"}
                    </td>
                    <td style={{ padding: "10px 20px", fontSize: 13, color: "#64748B" }}>
                      {new Date(r.created_at).toLocaleTimeString()} {new Date(r.created_at).toLocaleDateString()}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
