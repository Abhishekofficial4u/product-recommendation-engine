// frontend/src/pages/Home.jsx
import { useState, useEffect } from "react";
import { api } from "../services/api";
import RecommendCard from "../components/RecommendCard";
import { Spinner, ErrorBanner, ModelBadge } from "../components/UI";

const MODELS = [
  { key: "hybrid",     label: "Hybrid",     desc: "SVD + Content (best)" },
  { key: "svd",        label: "SVD",         desc: "Matrix factorization" },
  { key: "user_cf",    label: "User-CF",     desc: "Similar users" },
  { key: "item_cf",    label: "Item-CF",     desc: "Similar items" },
  { key: "content",    label: "Content",     desc: "Genre similarity" },
  { key: "popularity", label: "Popularity",  desc: "Top rated globally" },
];

export default function Home() {
  const [users,     setUsers]     = useState([]);
  const [userId,    setUserId]    = useState("");
  const [model,     setModel]     = useState("hybrid");
  const [topK,      setTopK]      = useState(10);
  const [result,    setResult]    = useState(null);
  const [loading,   setLoading]   = useState(false);
  const [error,     setError]     = useState(null);
  const [health,    setHealth]    = useState(null);

  useEffect(() => {
    api.users().then(setUsers).catch(() => {});
    api.health().then(setHealth).catch(() => {});
  }, []);

  async function getRecommendations() {
    if (!userId) return;
    setLoading(true); setError(null); setResult(null);
    try {
      const data = await api.recommend(Number(userId), { topK, model });
      setResult(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{ maxWidth: 1100, margin: "0 auto", padding: "28px 20px" }}>

      {/* Header */}
      <div style={{ marginBottom: 28 }}>
        <h1 style={{ fontSize: 26, fontWeight: 700, color: "#1E293B", margin: "0 0 6px" }}>
          Product Recommendation Engine
        </h1>
        <p style={{ color: "#64748B", fontSize: 14, margin: 0 }}>
          Personalised recommendations powered by 4 ML models — built in 25 days.
        </p>
      </div>

      {/* API status banner */}
      {health && (
        <div style={{
          background: "#F0FDF4", border: "1px solid #BBF7D0",
          borderRadius: 8, padding: "8px 14px", marginBottom: 20,
          display: "flex", alignItems: "center", gap: 10, fontSize: 13
        }}>
          <span style={{ color: "#16A34A", fontWeight: 600 }}>● API live</span>
          <span style={{ color: "#15803D" }}>
            {health.total_users} users · {health.total_items} items · {health.total_ratings?.toLocaleString()} ratings
          </span>
          <span style={{ marginLeft: "auto", color: "#94A3B8", fontSize: 11 }}>
            Models loaded: {health.models_loaded?.join(", ")}
          </span>
        </div>
      )}

      {/* Controls */}
      <div style={{
        background: "#fff", borderRadius: 12, border: "1px solid #E2E8F0",
        padding: "20px 24px", marginBottom: 24,
        boxShadow: "0 1px 3px rgba(0,0,0,0.05)"
      }}>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr auto", gap: 16, alignItems: "end" }}>
          {/* User selector */}
          <div>
            <label style={{ fontSize: 12, fontWeight: 500, color: "#374151", display: "block", marginBottom: 6 }}>
              User ID
            </label>
            <select
              value={userId}
              onChange={e => setUserId(e.target.value)}
              style={{
                width: "100%", height: 38, borderRadius: 8, padding: "0 12px",
                border: "1px solid #D1D5DB", fontSize: 14, background: "#FAFAFA",
                color: "#1E293B"
              }}
            >
              <option value="">— Select a user —</option>
              {users.slice(0, 100).map(u => (
                <option key={u} value={u}>User {u}</option>
              ))}
            </select>
          </div>

          {/* Top-K slider */}
          <div>
            <label style={{ fontSize: 12, fontWeight: 500, color: "#374151", display: "block", marginBottom: 6 }}>
              Results: {topK}
            </label>
            <input type="range" min={5} max={20} step={1} value={topK}
              onChange={e => setTopK(Number(e.target.value))}
              style={{ width: "100%", accentColor: "#3B82F6" }}
            />
          </div>

          {/* Get button */}
          <button
            onClick={getRecommendations}
            disabled={!userId || loading}
            style={{
              height: 38, padding: "0 24px", borderRadius: 8,
              background: !userId ? "#94A3B8" : "#1B3A6B",
              color: "#fff", border: "none", cursor: !userId ? "not-allowed" : "pointer",
              fontSize: 14, fontWeight: 600, transition: "background .15s",
              whiteSpace: "nowrap"
            }}
          >
            {loading ? "Loading…" : "Get Recommendations →"}
          </button>
        </div>

        {/* Model tabs */}
        <div style={{ marginTop: 16 }}>
          <div style={{ fontSize: 12, fontWeight: 500, color: "#374151", marginBottom: 8 }}>
            Model
          </div>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
            {MODELS.map(m => (
              <button key={m.key} onClick={() => setModel(m.key)} style={{
                padding: "5px 14px", borderRadius: 20,
                border: model === m.key ? "1.5px solid #3B82F6" : "1px solid #E2E8F0",
                background: model === m.key ? "#EFF6FF" : "#F8FAFC",
                color: model === m.key ? "#1D4ED8" : "#64748B",
                fontSize: 12, fontWeight: model === m.key ? 600 : 400,
                cursor: "pointer", transition: "all .15s"
              }}>
                {m.label}
                <span style={{ display: "block", fontSize: 10, color: model === m.key ? "#60A5FA" : "#94A3B8", marginTop: 1 }}>
                  {m.desc}
                </span>
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Error */}
      {error && <ErrorBanner message={error}/>}

      {/* Loading */}
      {loading && <Spinner/>}

      {/* Results */}
      {result && !loading && (
        <div>
          {/* Result header */}
          <div style={{
            display: "flex", alignItems: "center", justifyContent: "space-between",
            marginBottom: 16
          }}>
            <div style={{ fontSize: 15, fontWeight: 600, color: "#1E293B" }}>
              Top {result.items.length} recommendations for User {result.user_id}
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <ModelBadge model={result.model_used}/>
              {result.cold_start && (
                <span style={{
                  fontSize: 11, padding: "2px 8px", borderRadius: 20,
                  background: "#FFF7ED", color: "#C2410C", fontWeight: 500
                }}>
                  ⚡ Cold start mode
                </span>
              )}
              {result.response_ms && (
                <span style={{ fontSize: 11, color: "#94A3B8" }}>
                  {result.response_ms}ms
                </span>
              )}
            </div>
          </div>

          {/* Cards grid */}
          <div style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fill, minmax(260px, 1fr))",
            gap: 16
          }}>
            {result.items.map(item => (
              <RecommendCard key={item.item_id} item={item} userId={result.user_id}/>
            ))}
          </div>
        </div>
      )}

      {/* Empty state */}
      {!result && !loading && !error && (
        <div style={{
          textAlign: "center", padding: "60px 20px",
          color: "#94A3B8"
        }}>
          <div style={{ fontSize: 48, marginBottom: 12 }}>🎯</div>
          <div style={{ fontSize: 16, fontWeight: 500, color: "#64748B", marginBottom: 6 }}>
            Select a user to get started
          </div>
          <div style={{ fontSize: 13 }}>
            Choose a user ID above and click Get Recommendations
          </div>
        </div>
      )}
    </div>
  );
}
