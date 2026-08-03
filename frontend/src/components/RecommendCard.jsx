// frontend/src/components/RecommendCard.jsx
import { useState } from "react";
import { GenreBadge, RatingStars } from "./UI";
import { api } from "../services/api";

export default function RecommendCard({ item, userId }) {
  const [rated,     setRated]     = useState(false);
  const [userRating,setUserRating]= useState(0);
  const [submitting,setSubmitting]= useState(false);
  const [showSim,   setShowSim]   = useState(false);
  const [similar,   setSimilar]   = useState(null);

  const genres = (item.genres || "").split("|").filter(Boolean);
  const stars  = Math.round(item.predicted_rating);

  async function submitRating(val) {
    if (rated || submitting) return;
    setUserRating(val);
    setSubmitting(true);
    try {
      await api.rate(userId, item.item_id, val);
      setRated(true);
    } catch (e) {
      console.error("Rating failed:", e);
    } finally {
      setSubmitting(false);
    }
  }

  async function loadSimilar() {
    if (similar) { setShowSim(!showSim); return; }
    try {
      const res = await api.similar(item.item_id, 4);
      setSimilar(res.items);
      setShowSim(true);
    } catch (e) { console.error(e); }
  }

  return (
    <div style={{
      background: "#fff", borderRadius: 12, border: "1px solid #E2E8F0",
      overflow: "hidden", transition: "box-shadow .2s",
      boxShadow: "0 1px 3px rgba(0,0,0,0.06)"
    }}
      onMouseEnter={e => e.currentTarget.style.boxShadow="0 4px 16px rgba(0,0,0,0.10)"}
      onMouseLeave={e => e.currentTarget.style.boxShadow="0 1px 3px rgba(0,0,0,0.06)"}
    >
      {/* Rank ribbon */}
      <div style={{
        padding: "10px 16px 8px", display: "flex",
        alignItems: "center", gap: 10,
        borderBottom: "1px solid #F1F5F9"
      }}>
        <div style={{
          width: 28, height: 28, borderRadius: "50%",
          background: item.rank === 1 ? "#F59E0B" : item.rank <= 3 ? "#3B82F6" : "#E2E8F0",
          color: item.rank <= 3 ? "#fff" : "#64748B",
          display: "flex", alignItems: "center", justifyContent: "center",
          fontSize: 12, fontWeight: 700, flexShrink: 0
        }}>
          {item.rank}
        </div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ fontWeight: 600, fontSize: 14, color: "#1E293B", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
            {item.title}
          </div>
          <div style={{ fontSize: 11, color: "#94A3B8", marginTop: 1 }}>
            Item ID: {item.item_id}
          </div>
        </div>
      </div>

      {/* Body */}
      <div style={{ padding: "12px 16px" }}>
        {/* Genre badges */}
        <div style={{ marginBottom: 10 }}>
          {genres.map(g => <GenreBadge key={g} genre={g}/>)}
        </div>

        {/* Predicted rating */}
        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 12 }}>
          <div style={{ fontSize: 22, fontWeight: 700, color: "#1E293B" }}>
            {item.predicted_rating.toFixed(2)}
          </div>
          <div>
            <div style={{ fontSize: 10, color: "#94A3B8", lineHeight: 1.2 }}>Predicted</div>
            <div style={{ color: "#F59E0B", fontSize: 14, lineHeight: 1 }}>
              {"★".repeat(stars)}{"☆".repeat(5-stars)}
            </div>
          </div>
        </div>

        {/* Rate this item */}
        <div style={{ borderTop: "1px solid #F1F5F9", paddingTop: 10 }}>
          {rated ? (
            <div style={{ fontSize: 12, color: "#16A34A", fontWeight: 500 }}>
              ✓ Thanks! Rated {userRating}/5
            </div>
          ) : (
            <div>
              <div style={{ fontSize: 11, color: "#94A3B8", marginBottom: 4 }}>Rate this item:</div>
              <RatingStars value={userRating} onChange={submitRating}/>
            </div>
          )}
        </div>

        {/* Similar items button */}
        <button onClick={loadSimilar} style={{
          marginTop: 10, width: "100%", padding: "6px 0",
          background: showSim ? "#EFF6FF" : "transparent",
          border: "1px solid #DBEAFE", borderRadius: 6,
          fontSize: 12, color: "#3B82F6", cursor: "pointer",
          fontWeight: 500, transition: "all .15s"
        }}>
          {showSim ? "▲ Hide similar" : "▼ Show similar items"}
        </button>

        {/* Similar items list */}
        {showSim && similar && (
          <div style={{ marginTop: 8 }}>
            {similar.map(s => (
              <div key={s.item_id} style={{
                display: "flex", justifyContent: "space-between",
                alignItems: "center", padding: "5px 0",
                borderBottom: "1px solid #F8FAFC", fontSize: 12
              }}>
                <span style={{ color: "#475569" }}>{s.title}</span>
                <span style={{ color: "#94A3B8", whiteSpace: "nowrap", marginLeft: 8 }}>
                  {(s.similarity * 100).toFixed(0)}% similar
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
