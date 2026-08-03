// frontend/src/pages/Similar.jsx
import { useState, useEffect } from "react";
import { api } from "../services/api";
import { GenreBadge, Spinner, ErrorBanner, Card } from "../components/UI";

export default function Similar() {
  const [items,   setItems]   = useState([]);
  const [itemId,  setItemId]  = useState("");
  const [topK,    setTopK]    = useState(6);
  const [result,  setResult]  = useState(null);
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState(null);

  useEffect(() => {
    api.items(100).then(setItems).catch(() => {});
  }, []);

  async function search() {
    if (!itemId) return;
    setLoading(true); setError(null); setResult(null);
    try {
      const data = await api.similar(Number(itemId), topK);
      setResult(data);
    } catch (e) { setError(e.message); }
    finally { setLoading(false); }
  }

  return (
    <div style={{ maxWidth: 900, margin: "0 auto", padding: "28px 20px" }}>
      <div style={{ marginBottom: 24 }}>
        <h1 style={{ fontSize: 26, fontWeight: 700, color: "#1E293B", margin: "0 0 6px" }}>
          Similar Items
        </h1>
        <p style={{ color: "#64748B", fontSize: 14, margin: 0 }}>
          Find items with similar genres using TF-IDF cosine similarity. Powers "you might also like" features.
        </p>
      </div>

      <Card style={{ padding: "20px 24px", marginBottom: 24 }}>
        <div style={{ display: "grid", gridTemplateColumns: "1fr auto auto", gap: 14, alignItems: "end" }}>
          <div>
            <label style={{ fontSize: 12, fontWeight: 500, color: "#374151", display: "block", marginBottom: 6 }}>
              Select an Item
            </label>
            <select value={itemId} onChange={e => setItemId(e.target.value)} style={{
              width: "100%", height: 38, borderRadius: 8, padding: "0 12px",
              border: "1px solid #D1D5DB", fontSize: 14, background: "#FAFAFA", color: "#1E293B"
            }}>
              <option value="">— Choose an item —</option>
              {items.map(it => (
                <option key={it.item_id} value={it.item_id}>
                  {it.title} ({it.genres.split("|").slice(0,2).join(", ")})
                </option>
              ))}
            </select>
          </div>
          <div>
            <label style={{ fontSize: 12, fontWeight: 500, color: "#374151", display: "block", marginBottom: 6 }}>
              Show {topK}
            </label>
            <input type="range" min={3} max={15} step={1} value={topK}
              onChange={e => setTopK(Number(e.target.value))}
              style={{ width: 100, accentColor: "#3B82F6" }}
            />
          </div>
          <button onClick={search} disabled={!itemId || loading} style={{
            height: 38, padding: "0 24px", borderRadius: 8,
            background: !itemId ? "#94A3B8" : "#1B3A6B",
            color: "#fff", border: "none",
            cursor: !itemId ? "not-allowed" : "pointer",
            fontSize: 14, fontWeight: 600
          }}>
            Find Similar →
          </button>
        </div>
      </Card>

      {error && <ErrorBanner message={error}/>}
      {loading && <Spinner/>}

      {result && !loading && (
        <div>
          {/* Source item */}
          <div style={{
            background: "#EFF6FF", borderRadius: 10, border: "1px solid #BFDBFE",
            padding: "14px 18px", marginBottom: 20
          }}>
            <div style={{ fontSize: 12, color: "#3B82F6", fontWeight: 600, marginBottom: 4 }}>
              SOURCE ITEM
            </div>
            <div style={{ fontSize: 15, fontWeight: 600, color: "#1E3A8A" }}>{result.source_title}</div>
            <div style={{ marginTop: 6 }}>
              {result.source_genres.split("|").map(g => <GenreBadge key={g} genre={g}/>)}
            </div>
          </div>

          {/* Similar items */}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(250px, 1fr))", gap: 14 }}>
            {result.items.map((it) => (
              <Card key={it.item_id} style={{ padding: "14px 16px" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 8 }}>
                  <div style={{ fontWeight: 600, fontSize: 14, color: "#1E293B" }}>{it.title}</div>
                  <div style={{
                    background: "#F0FDF4", color: "#166534",
                    fontSize: 12, fontWeight: 600,
                    padding: "2px 8px", borderRadius: 20, whiteSpace: "nowrap", marginLeft: 8
                  }}>
                    {(it.similarity * 100).toFixed(0)}%
                  </div>
                </div>
                <div>{it.genres.split("|").map(g => <GenreBadge key={g} genre={g}/>)}</div>
                {/* Similarity bar */}
                <div style={{ marginTop: 10 }}>
                  <div style={{ fontSize: 10, color: "#94A3B8", marginBottom: 3 }}>Similarity</div>
                  <div style={{ background: "#F1F5F9", borderRadius: 4, height: 5 }}>
                    <div style={{
                      height: 5, borderRadius: 4, background: "#3B82F6",
                      width: `${(it.similarity * 100).toFixed(0)}%`,
                      transition: "width .5s ease"
                    }}/>
                  </div>
                </div>
              </Card>
            ))}
          </div>
        </div>
      )}

      {!result && !loading && !error && (
        <div style={{ textAlign: "center", padding: "60px 20px", color: "#94A3B8" }}>
          <div style={{ fontSize: 48, marginBottom: 12 }}>🔍</div>
          <div style={{ fontSize: 15, fontWeight: 500, color: "#64748B" }}>
            Select an item to find similar ones
          </div>
        </div>
      )}
    </div>
  );
}
