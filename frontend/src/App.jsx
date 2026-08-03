// frontend/src/App.jsx
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { AuthProvider } from "./context/AuthContext";
import ProtectedRoute from "./components/ProtectedRoute";
import Navbar     from "./components/Navbar";
import Home       from "./pages/Home";
import Similar    from "./pages/Similar";
import Dashboard  from "./pages/Dashboard";
import About      from "./pages/About";
import Login      from "./pages/Login";
import Register   from "./pages/Register";
import Analytics  from "./pages/Analytics";
import AgentCopilot from "./pages/AgentCopilot";

export default function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <div style={{ minHeight: "100vh", background: "#F8FAFC", fontFamily: "'Inter', 'Segoe UI', sans-serif" }}>
          <Navbar />
          <Routes>
            <Route path="/"          element={<ProtectedRoute><Home /></ProtectedRoute>} />
            <Route path="/copilot"   element={<AgentCopilot />} />
            <Route path="/similar"   element={<ProtectedRoute><Similar /></ProtectedRoute>} />
            <Route path="/dashboard" element={<ProtectedRoute><Dashboard /></ProtectedRoute>} />
            <Route path="/analytics" element={<ProtectedRoute><Analytics /></ProtectedRoute>} />
            <Route path="/about"     element={<About />} />
            <Route path="/login"     element={<Login />} />
            <Route path="/register"  element={<Register />} />
            <Route path="*"          element={<NotFound />} />
          </Routes>
        </div>
      </AuthProvider>
    </BrowserRouter>
  );
}

function NotFound() {
  return (
    <div style={{ textAlign: "center", padding: "80px 20px" }}>
      <div style={{ fontSize: 64, marginBottom: 16 }}>🤷</div>
      <h2 style={{ color: "#1E293B", fontSize: 22, margin: "0 0 8px" }}>Page not found</h2>
      <a href="/" style={{ color: "#3B82F6", fontSize: 14 }}>← Back to recommendations</a>
    </div>
  );
}
