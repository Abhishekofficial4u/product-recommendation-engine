# Quickstart Guide: Running the Product Recommendation Engine

This guide provides step-by-step instructions for anyone cloning and running this project from GitHub on their local machine.

---

## 🚀 Option 1: Run via Docker (Recommended)

Running with Docker is the easiest method — it automatically installs all backend and frontend dependencies, configures the database, loads the Machine Learning models, and starts both servers.

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running on your system.

### Steps
1. **Clone the repository:**
   ```bash
   git clone https://github.com/Abhishekofficial4u/product-recommendation-engine.git
   cd product-recommendation-engine
   ```

2. **Launch the application:**
   ```bash
   docker compose up --build
   ```

3. **Access the application in your browser:**
   - 🌐 **React Frontend Web App:** [http://localhost:3000](http://localhost:3000)
   - ⚙️ **FastAPI Backend REST API:** [http://localhost:8000](http://localhost:8000)
   - 📖 **Interactive Swagger API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

4. **Stop the application:**
   Press `Ctrl + C` in your terminal or run:
   ```bash
   docker compose down
   ```

---

## 💻 Option 2: Run Locally (Development Mode)

If you prefer running without Docker using Python and Node.js directly:

### Prerequisites
- Python 3.10+
- Node.js 18+ and npm

### Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Abhishekofficial4u/product-recommendation-engine.git
   cd product-recommendation-engine
   ```

2. **Start the Backend (FastAPI):**
   ```bash
   # Create and activate virtual environment
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate

   # Install dependencies
   pip install setuptools wheel "numpy==1.26.3" Cython
   pip install --no-build-isolation -r backend/requirements.txt

   # Run FastAPI backend server
   python -m uvicorn backend.api.main:app --reload --port 8000
   ```

3. **Start the Frontend (React + Vite) in a second terminal:**
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

4. **Access the application:**
   - 🌐 **React Frontend Web App:** [http://localhost:5173](http://localhost:5173)
   - ⚙️ **FastAPI Backend Server:** [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🧪 Running Unit Tests

To run the backend test suite:
```bash
python -m pytest backend
```

---

## 🛠️ Troubleshooting

- **Docker daemon connection error:** Make sure Docker Desktop application is launched and running on your system before executing `docker compose up`.
- **Port conflicts:** Ensure ports `3000` and `8000` (or `5173`) are not currently in use by other applications on your system.
