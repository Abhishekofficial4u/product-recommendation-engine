// frontend/src/components/Navbar.jsx
import { Link, useLocation, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

export default function Navbar() {
  const { pathname } = useLocation();
  const navigate = useNavigate();
  const { token, user, logout } = useAuth();

  const links = token
    ? [
        { to: "/",          label: "Recommendations" },
        { to: "/copilot",   label: "🤖 AI Copilot"   },
        { to: "/similar",   label: "Similar Items"   },
        { to: "/dashboard", label: "Model Dashboard" },
        { to: "/analytics", label: "System Telemetry" },
        { to: "/about",     label: "About"           },
      ]
    : [
        { to: "/copilot",   label: "🤖 AI Copilot"   },
        { to: "/about",     label: "About"           },
      ];

  function handleLogout() {
    logout();
    navigate("/login");
  }

  return (
    <nav style={{
      position: "sticky", top: 0, zIndex: 50,
      background: "#1B3A6B", borderBottom: "1px solid #2E75B6",
      padding: "0 24px", display: "flex", alignItems: "center",
      gap: 32, height: 56
    }}>
      <span style={{ fontWeight: 700, fontSize: 17, color: "#fff", letterSpacing: "-0.3px", whiteSpace: "nowrap" }}>
        🎯 RecEngine
      </span>
      <div style={{ display: "flex", gap: 4, flex: 1 }}>
        {links.map(({ to, label }) => {
          const active = pathname === to;
          return (
            <Link key={to} to={to} style={{
              padding: "6px 14px", borderRadius: 6,
              fontSize: 14, fontWeight: active ? 500 : 400,
              color: active ? "#fff" : "#93C5FD",
              background: active ? "rgba(255,255,255,0.12)" : "transparent",
              textDecoration: "none", transition: "all .15s"
            }}>
              {label}
            </Link>
          );
        })}
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
        {token ? (
          <>
            <span style={{ fontSize: 13, color: "#E0F2FE", fontWeight: 500 }}>
              👤 {user?.username}
            </span>
            <button onClick={handleLogout} style={{
              background: "transparent", border: "1px solid #93C5FD",
              borderRadius: 6, padding: "5px 12px", fontSize: 13,
              color: "#93C5FD", cursor: "pointer", transition: "all .15s"
            }}
              onMouseEnter={e => { e.target.style.background = "rgba(255,255,255,0.08)"; e.target.style.color = "#fff"; }}
              onMouseLeave={e => { e.target.style.background = "transparent"; e.target.style.color = "#93C5FD"; }}
            >
              Logout
            </button>
          </>
        ) : (
          <div style={{ display: "flex", gap: 8 }}>
            <Link to="/login" style={{
              border: "1px solid #93C5FD", borderRadius: 6, padding: "5px 12px",
              fontSize: 13, color: "#93C5FD", textDecoration: "none", transition: "all .15s"
            }}>
              Login
            </Link>
            <Link to="/register" style={{
              background: "#3B82F6", border: "1px solid #3B82F6", borderRadius: 6,
              padding: "5px 12px", fontSize: 13, color: "#fff", textDecoration: "none",
              transition: "all .15s"
            }}>
              Sign Up
            </Link>
          </div>
        )}
        <span style={{ fontSize: 11, color: "#60A5FA" }}>v1.0.0</span>
      </div>
    </nav>
  );
}
