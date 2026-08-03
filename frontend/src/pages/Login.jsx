// frontend/src/pages/Login.jsx
import React, { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import { Card } from "../components/UI";

export default function Login() {
  const { login } = useAuth();
  const navigate = useNavigate();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState(null);
  const [submitting, setSubmitting] = useState(false);

  async function handleSubmit(e) {
    e.preventDefault();
    if (!username.trim() || !password) {
      setError("Please enter both username and password.");
      return;
    }
    setError(null);
    setSubmitting(true);
    try {
      const ok = await login(username.trim(), password);
      if (ok) {
        navigate("/");
      } else {
        setError("Invalid credentials.");
      }
    } catch (err) {
      setError(err.message || "Login failed.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div style={{ maxWidth: 420, margin: "60px auto", padding: "0 20px" }}>
      <Card style={{ padding: "32px 24px" }}>
        <h2 style={{ margin: "0 0 8px", fontSize: 22, fontWeight: 700, color: "#1E293B", textAlign: "center" }}>
          Welcome back!
        </h2>
        <p style={{ margin: "0 0 24px", fontSize: 13, color: "#64748B", textAlign: "center" }}>
          Log in to access your recommendations and rates
        </p>

        {error && (
          <div style={{
            background: "#FEF2F2", border: "1px solid #FEE2E2", borderRadius: 8,
            padding: "10px 14px", color: "#991B1B", fontSize: 13, marginBottom: 16
          }}>
            ⚠️ {error}
          </div>
        )}

        <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            <label style={{ fontSize: 12, fontWeight: 600, color: "#475569" }}>Username</label>
            <input
              type="text"
              value={username}
              onChange={e => setUsername(e.target.value)}
              placeholder="Enter your username"
              style={{
                padding: "8px 12px", borderRadius: 6, border: "1px solid #CBD5E1",
                fontSize: 14, outline: "none", transition: "border-color .15s"
              }}
              onFocus={e => e.target.style.borderColor = "#1B3A6B"}
              onBlur={e => e.target.style.borderColor = "#CBD5E1"}
            />
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            <label style={{ fontSize: 12, fontWeight: 600, color: "#475569" }}>Password</label>
            <input
              type="password"
              value={password}
              onChange={e => setPassword(e.target.value)}
              placeholder="••••••••"
              style={{
                padding: "8px 12px", borderRadius: 6, border: "1px solid #CBD5E1",
                fontSize: 14, outline: "none", transition: "border-color .15s"
              }}
              onFocus={e => e.target.style.borderColor = "#1B3A6B"}
              onBlur={e => e.target.style.borderColor = "#CBD5E1"}
            />
          </div>

          <button
            type="submit"
            disabled={submitting}
            style={{
              padding: "10px 0", background: "#1B3A6B", color: "#fff",
              border: "none", borderRadius: 6, fontSize: 14, fontWeight: 600,
              cursor: submitting ? "not-allowed" : "pointer", opacity: submitting ? 0.7 : 1,
              marginTop: 8, transition: "background .15s"
            }}
          >
            {submitting ? "Logging in..." : "Log In"}
          </button>
        </form>

        <p style={{ marginTop: 24, fontSize: 13, color: "#64748B", textAlign: "center", margin: "24px 0 0" }}>
          Don't have an account? <Link to="/register" style={{ color: "#3B82F6", fontWeight: 500 }}>Sign up</Link>
        </p>
      </Card>
    </div>
  );
}
