// frontend/src/components/UI.jsx
// Shared reusable components

const GENRE_COLORS = {
  Action:      ["#FEF3C7","#92400E"], Adventure: ["#D1FAE5","#065F46"],
  Animation:   ["#EDE9FE","#5B21B6"], Comedy:    ["#FEE2E2","#991B1B"],
  Crime:       ["#F3F4F6","#374151"], Documentary:["#DBEAFE","#1E40AF"],
  Drama:       ["#FCE7F3","#9D174D"], Fantasy:   ["#FFF7ED","#9A3412"],
  Horror:      ["#1F2937","#F9FAFB"], Romance:   ["#FDF2F8","#831843"],
  "Sci-Fi":    ["#EFF6FF","#1E3A8A"], Thriller:  ["#F8FAFC","#334155"],
};

export function GenreBadge({ genre }) {
  const [bg, text] = GENRE_COLORS[genre] || ["#F1F5F9","#475569"];
  return (
    <span style={{
      display: "inline-block", padding: "2px 8px", borderRadius: 20,
      fontSize: 11, fontWeight: 500, background: bg, color: text,
      margin: "2px 3px 2px 0", whiteSpace: "nowrap"
    }}>
      {genre}
    </span>
  );
}

export function Spinner({ size = 28 }) {
  return (
    <div style={{ display: "flex", justifyContent: "center", padding: 32 }}>
      <div style={{
        width: size, height: size, borderRadius: "50%",
        border: `3px solid #E2E8F0`, borderTopColor: "#3B82F6",
        animation: "spin 0.7s linear infinite"
      }}/>
      <style>{`@keyframes spin{to{transform:rotate(360deg)}}`}</style>
    </div>
  );
}

export function ErrorBanner({ message }) {
  return (
    <div style={{
      background: "#FEF2F2", border: "1px solid #FECACA",
      borderRadius: 8, padding: "12px 16px", color: "#991B1B",
      fontSize: 14, margin: "12px 0"
    }}>
      ⚠️ {message}
    </div>
  );
}

export function Card({ children, style = {} }) {
  return (
    <div style={{
      background: "#fff", borderRadius: 12,
      border: "1px solid #E2E8F0",
      boxShadow: "0 1px 3px rgba(0,0,0,0.06)", ...style
    }}>
      {children}
    </div>
  );
}

export function StatCard({ label, value, color = "#3B82F6", icon = "📊" }) {
  return (
    <div style={{
      background: "#fff", borderRadius: 12, border: "1px solid #E2E8F0",
      padding: "16px 20px", display: "flex", alignItems: "center", gap: 14
    }}>
      <div style={{
        width: 44, height: 44, borderRadius: 10,
        background: color + "20", display: "flex",
        alignItems: "center", justifyContent: "center", fontSize: 22
      }}>
        {icon}
      </div>
      <div>
        <div style={{ fontSize: 22, fontWeight: 600, color: "#1E293B", lineHeight: 1.2 }}>{value}</div>
        <div style={{ fontSize: 12, color: "#64748B", marginTop: 2 }}>{label}</div>
      </div>
    </div>
  );
}

export function ModelBadge({ model }) {
  const styles = {
    hybrid:     ["#EEF2FF","#4F46E5"], svd:        ["#F0FDF4","#166534"],
    user_cf:    ["#EFF6FF","#1E40AF"], item_cf:    ["#FFF7ED","#9A3412"],
    content:    ["#FDF4FF","#7E22CE"], popularity: ["#F8FAFC","#475569"],
  };
  const [bg, text] = styles[model] || ["#F1F5F9","#475569"];
  return (
    <span style={{
      padding: "3px 10px", borderRadius: 20, fontSize: 11,
      fontWeight: 600, background: bg, color: text, letterSpacing: "0.02em"
    }}>
      {model.replace("_", "-").toUpperCase()}
    </span>
  );
}

export function RatingStars({ value, onChange }) {
  return (
    <div style={{ display: "flex", gap: 4 }}>
      {[1,2,3,4,5].map(n => (
        <button key={n} onClick={() => onChange(n)} style={{
          background: "none", border: "none", cursor: "pointer",
          fontSize: 22, color: n <= value ? "#F59E0B" : "#D1D5DB",
          padding: 2, transition: "color .1s"
        }}>
          ★
        </button>
      ))}
    </div>
  );
}
