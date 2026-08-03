// frontend/src/pages/AgentCopilot.jsx
import { useState } from "react";
import api from "../services/api";

export default function AgentCopilot() {
  const [query, setQuery] = useState("");
  const [userId, setUserId] = useState(1);
  const [topK, setTopK] = useState(5);
  const [modelPref, setModelPref] = useState("hybrid");
  const [loading, setLoading] = useState(false);
  const [chatHistory, setChatHistory] = useState([
    {
      sender: "agent",
      text: "👋 Hi! I am your Agentic AI Recommendation Copilot. Ask me for personalized recommendations, similar items, or genre-based sci-fi/action search!",
      tools: [],
      executionTimeMs: 0
    }
  ]);

  async function handleSend(e) {
    e.preventDefault();
    if (!query.trim()) return;

    const userMsg = query;
    setQuery("");
    setChatHistory(prev => [...prev, { sender: "user", text: userMsg }]);
    setLoading(true);

    try {
      const data = await api.agentChat({
        user_query: userMsg,
        user_id: Number(userId),
        top_k: Number(topK),
        model_preference: modelPref
      });

      setChatHistory(prev => [
        ...prev,
        {
          sender: "agent",
          text: data.response_text,
          intent: data.intent_detected,
          tools: data.tool_calls || [],
          executionTimeMs: data.execution_time_ms
        }
      ]);
    } catch (err) {
      setChatHistory(prev => [
        ...prev,
        {
          sender: "agent",
          text: "⚠️ Failed to execute Agent pipeline. Please check backend connection.",
          tools: []
        }
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{ maxWidth: 1000, margin: "32px auto", padding: "0 24px" }}>
      <div style={{ background: "linear-gradient(135deg, #1E293B, #0F172A)", borderRadius: 12, padding: 24, color: "#fff", marginBottom: 24 }}>
        <h2 style={{ margin: "0 0 8px", fontSize: 24, display: "flex", alignItems: "center", gap: 10 }}>
          🤖 Agentic AI Recommendation Copilot
        </h2>
        <p style={{ margin: 0, color: "#94A3B8", fontSize: 14 }}>
          Autonomous Tool-Calling Agent & RAG Vector Engine for personalized discovery.
        </p>
      </div>

      {/* Control Strip */}
      <div style={{ background: "#fff", padding: 16, borderRadius: 8, border: "1px solid #E2E8F0", marginBottom: 20, display: "flex", gap: 16, flexWrap: "wrap", alignItems: "center" }}>
        <div>
          <label style={{ fontSize: 12, fontWeight: 600, color: "#64748B", display: "block" }}>User ID</label>
          <input type="number" value={userId} onChange={e => setUserId(e.target.value)} min={1} style={{ padding: "6px 10px", borderRadius: 6, border: "1px solid #CBD5E1", width: 80 }} />
        </div>
        <div>
          <label style={{ fontSize: 12, fontWeight: 600, color: "#64748B", display: "block" }}>Top K</label>
          <input type="number" value={topK} onChange={e => setTopK(e.target.value)} min={1} max={20} style={{ padding: "6px 10px", borderRadius: 6, border: "1px solid #CBD5E1", width: 70 }} />
        </div>
        <div>
          <label style={{ fontSize: 12, fontWeight: 600, color: "#64748B", display: "block" }}>Model Preference</label>
          <select value={modelPref} onChange={e => setModelPref(e.target.value)} style={{ padding: "6px 10px", borderRadius: 6, border: "1px solid #CBD5E1" }}>
            <option value="hybrid">Hybrid (SVD + Content)</option>
            <option value="svd">SVD Matrix Factorization</option>
            <option value="user_cf">User Collaborative</option>
            <option value="item_cf">Item Collaborative</option>
            <option value="content">Content TF-IDF</option>
            <option value="popularity">Popularity</option>
          </select>
        </div>
      </div>

      {/* Chat Messages */}
      <div style={{ background: "#fff", borderRadius: 12, border: "1px solid #E2E8F0", padding: 20, minHeight: 400, display: "flex", flexDirection: "column", gap: 16 }}>
        {chatHistory.map((msg, idx) => (
          <div key={idx} style={{ alignSelf: msg.sender === "user" ? "flex-end" : "flex-start", maxWidth: "85%" }}>
            <div style={{
              background: msg.sender === "user" ? "#3B82F6" : "#F1F5F9",
              color: msg.sender === "user" ? "#fff" : "#1E293B",
              padding: "14px 18px",
              borderRadius: 12,
              whiteSpace: "pre-line",
              fontSize: 14,
              boxShadow: "0 1px 3px rgba(0,0,0,0.05)"
            }}>
              {msg.text}

              {msg.tools && msg.tools.length > 0 && (
                <div style={{ marginTop: 12, paddingTop: 10, borderTop: "1px solid #CBD5E1" }}>
                  <div style={{ fontSize: 11, fontWeight: 700, color: "#475569", textTransform: "uppercase", marginBottom: 6 }}>
                    🛠️ Autonomous Tool Call Chain ({msg.executionTimeMs} ms):
                  </div>
                  {msg.tools.map((t, tIdx) => (
                    <div key={tIdx} style={{ background: "#E2E8F0", borderRadius: 6, padding: "6px 10px", fontSize: 12, marginBottom: 4, fontFamily: "monospace" }}>
                      <strong>{t.tool_name}</strong>({JSON.stringify(t.arguments)}) → {t.output_summary}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}

        {loading && (
          <div style={{ alignSelf: "flex-start", background: "#F1F5F9", padding: "12px 16px", borderRadius: 12, fontSize: 13, color: "#64748B" }}>
            🤖 Agent planning workflow & executing tool calls...
          </div>
        )}
      </div>

      {/* Query Form */}
      <form onSubmit={handleSend} style={{ display: "flex", gap: 10, marginTop: 16 }}>
        <input
          type="text"
          placeholder="Ask AI Copilot: e.g. 'Recommend sci-fi thrillers like Star Wars' or 'Top recommendations for User 1'"
          value={query}
          onChange={e => setQuery(e.target.value)}
          style={{ flex: 1, padding: "12px 16px", borderRadius: 8, border: "1px solid #CBD5E1", fontSize: 14 }}
        />
        <button
          type="submit"
          disabled={loading}
          style={{ background: "#2563EB", color: "#fff", border: "none", borderRadius: 8, padding: "0 24px", fontWeight: 600, cursor: "pointer" }}
        >
          Send
        </button>
      </form>
    </div>
  );
}
