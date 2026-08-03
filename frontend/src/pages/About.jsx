// frontend/src/pages/About.jsx
import { Card, StatCard } from "../components/UI";

const PHASES = [
  { num: 1, label: "Setup & Data",    days: "Days 1–5",   color: "#3B82F6", desc: "Downloaded MovieLens 100K, ran EDA (discovered 95.1% sparsity), built per-user 80/20 train/test split, trained popularity baseline." },
  { num: 2, label: "ML Models",       days: "Days 6–14",  color: "#8B5CF6", desc: "Built User-CF, Item-CF (Pearson), SVD with GridSearchCV (24 param combos), TF-IDF content model, and hybrid with alpha tuning." },
  { num: 3, label: "API Backend",     days: "Days 15–18", color: "#10B981", desc: "FastAPI with 8 endpoints, Pydantic schemas, SQLite logging, CORS, Docker + docker-compose." },
  { num: 4, label: "React Frontend",  days: "Days 19–21", color: "#F59E0B", desc: "Vite + React + Recharts dashboard, recommendation cards with live rating submission, similar items explorer." },
  { num: 5, label: "Report & Present",days: "Days 22–25", color: "#EF4444", desc: "Project report, README, slides, live deployment to Render + Vercel, final mentor presentation." },
];

const STACK = [
  { cat: "Data & ML",    items: "Python · Pandas · NumPy · Scikit-learn · Scikit-Surprise · Scipy" },
  { cat: "Backend",      items: "FastAPI · Uvicorn · SQLAlchemy · SQLite · Pydantic" },
  { cat: "Frontend",     items: "React 18 · Vite · Recharts · React Router" },
  { cat: "DevOps",       items: "Git · Docker · docker-compose · Render · Vercel" },
];

export default function About() {
  return (
    <div style={{ maxWidth: 860, margin: "0 auto", padding: "28px 20px" }}>
      <div style={{ marginBottom: 28 }}>
        <h1 style={{ fontSize: 26, fontWeight: 700, color: "#1E293B", margin: "0 0 6px" }}>
          About This Project
        </h1>
        <p style={{ color: "#64748B", fontSize: 14, margin: 0 }}>
          Built in 25 days by a 4-person team during a final-year internship.
        </p>
      </div>

      {/* Stats */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))", gap: 12, marginBottom: 28 }}>
        <StatCard label="Days"        value="25"       color="#3B82F6" icon="📅"/>
        <StatCard label="Team members"value="4"        color="#8B5CF6" icon="👥"/>
        <StatCard label="ML models"   value="6"        color="#10B981" icon="🤖"/>
        <StatCard label="API endpoints"value="8"       color="#F59E0B" icon="🔌"/>
        <StatCard label="Ratings"     value="96,910"   color="#EF4444" icon="⭐"/>
      </div>

      {/* Project phases */}
      <Card style={{ padding: "20px 24px", marginBottom: 24 }}>
        <div style={{ fontSize: 15, fontWeight: 600, color: "#1E293B", marginBottom: 16 }}>
          Project Phases
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 0 }}>
          {PHASES.map((ph, i) => (
            <div key={ph.num} style={{ display: "flex", gap: 16, paddingBottom: i < PHASES.length-1 ? 16 : 0, marginBottom: i < PHASES.length-1 ? 16 : 0, borderBottom: i < PHASES.length-1 ? "1px solid #F1F5F9" : "none" }}>
              <div style={{ display: "flex", flexDirection: "column", alignItems: "center", flexShrink: 0 }}>
                <div style={{ width: 36, height: 36, borderRadius: "50%", background: ph.color, color: "#fff", display: "flex", alignItems: "center", justifyContent: "center", fontWeight: 700, fontSize: 14 }}>
                  {ph.num}
                </div>
                {i < PHASES.length-1 && <div style={{ width: 2, flex: 1, minHeight: 20, background: "#E2E8F0", marginTop: 4 }}/>}
              </div>
              <div style={{ paddingTop: 6 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 4 }}>
                  <span style={{ fontWeight: 600, fontSize: 14, color: "#1E293B" }}>{ph.label}</span>
                  <span style={{ fontSize: 11, padding: "2px 8px", borderRadius: 20, background: ph.color+"20", color: ph.color, fontWeight: 500 }}>{ph.days}</span>
                </div>
                <p style={{ fontSize: 13, color: "#64748B", margin: 0, lineHeight: 1.6 }}>{ph.desc}</p>
              </div>
            </div>
          ))}
        </div>
      </Card>

      {/* Tech stack */}
      <Card style={{ padding: "20px 24px", marginBottom: 24 }}>
        <div style={{ fontSize: 15, fontWeight: 600, color: "#1E293B", marginBottom: 14 }}>
          Technology Stack
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          {STACK.map(s => (
            <div key={s.cat} style={{ display: "flex", gap: 12, alignItems: "baseline" }}>
              <span style={{ fontSize: 12, fontWeight: 600, color: "#1B3A6B", minWidth: 100 }}>{s.cat}</span>
              <span style={{ fontSize: 13, color: "#64748B" }}>{s.items}</span>
            </div>
          ))}
        </div>
      </Card>

      {/* How the hybrid model works */}
      <Card style={{ padding: "20px 24px" }}>
        <div style={{ fontSize: 15, fontWeight: 600, color: "#1E293B", marginBottom: 10 }}>
          How the Hybrid Model Works
        </div>
        <div style={{
          background: "#F8FAFC", borderRadius: 8, padding: "12px 16px",
          fontFamily: "monospace", fontSize: 13, color: "#1E293B",
          border: "1px solid #E2E8F0", marginBottom: 14
        }}>
          score = 0.75 × SVD_score(1–5) + 0.25 × content_score(rescaled 1–5)
        </div>
        <p style={{ fontSize: 13, color: "#64748B", lineHeight: 1.7, margin: 0 }}>
          SVD learns hidden "taste profiles" for every user and "feature profiles" for every item, purely from rating patterns.
          Content-based uses TF-IDF on genres to measure item similarity. Blending them at α=0.75 gives SVD accuracy
          while the content side handles new items with no ratings (cold-start). Users with fewer than 3 ratings
          skip SVD entirely and get content-only recommendations.
        </p>
      </Card>
    </div>
  );
}
