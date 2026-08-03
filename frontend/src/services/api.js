// frontend/src/services/api.js
// Central API service — all HTTP calls go through here.
// Change API_BASE_URL to your deployed Render URL for production.

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

async function apiFetch(endpoint, options = {}) {
  const token = localStorage.getItem("token");
  const headers = {
    "Content-Type": "application/json",
    ...(options.headers || {}),
  };
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  const res = await fetch(`${API_BASE_URL}${endpoint}`, {
    ...options,
    headers,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Unknown error" }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

export const api = {
  // Auth
  login: (username, password) =>
    apiFetch("/auth/login", {
      method: "POST",
      body: JSON.stringify({ username, password }),
    }),

  register: (username, password) =>
    apiFetch("/auth/register", {
      method: "POST",
      body: JSON.stringify({ username, password }),
    }),

  // System
  health: () => apiFetch("/health"),

  // Recommendations
  recommend: (userId, { topK = 10, model = "hybrid" } = {}) =>
    apiFetch(`/recommend/${userId}?top_k=${topK}&model=${model}`),

  similar: (itemId, topK = 6) =>
    apiFetch(`/similar/${itemId}?top_k=${topK}`),

  // Feedback
  rate: (userId, itemId, rating) =>
    apiFetch("/rate", {
      method: "POST",
      body: JSON.stringify({ user_id: userId, item_id: itemId, rating }),
    }),

  // Catalogue
  items: (limit = 50) => apiFetch(`/items?limit=${limit}`),
  item:  (itemId)    => apiFetch(`/items/${itemId}`),
  users: ()          => apiFetch("/users"),

  // Analytics
  metrics: () => apiFetch("/metrics"),

  // Agent Chat
  agentChat: (payload) =>
    apiFetch("/agent/chat", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
};

export default api;
