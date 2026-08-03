// frontend/src/context/AuthContext.jsx
import React, { createContext, useContext, useState, useEffect } from "react";
import api from "../services/api";

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(localStorage.getItem("token"));
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const savedToken = localStorage.getItem("token");
    const savedUser = localStorage.getItem("username");
    if (savedToken && savedUser) {
      setToken(savedToken);
      setUser({ username: savedUser });
    }
    setLoading(false);
  }, []);

  async function login(username, password) {
    try {
      const data = await api.login(username, password);
      if (data && data.access_token) {
        localStorage.setItem("token", data.access_token);
        localStorage.setItem("username", username);
        setToken(data.access_token);
        setUser({ username });
        return true;
      }
      return false;
    } catch (e) {
      console.error("Login failed:", e);
      throw e;
    }
  }

  async function register(username, password) {
    try {
      await api.register(username, password);
      return true;
    } catch (e) {
      console.error("Registration failed:", e);
      throw e;
    }
  }

  function logout() {
    localStorage.removeItem("token");
    localStorage.removeItem("username");
    setToken(null);
    setUser(null);
  }

  return (
    <AuthContext.Provider value={{ user, token, loading, login, register, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
}
