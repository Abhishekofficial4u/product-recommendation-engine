# Product Recommendation Engine: Developer Documentation

Welcome to the Developer Documentation for the Product Recommendation Engine. This project is a hybrid recommendation system built using MovieLens-100K data, serving personal and item-similar predictions via a FastAPI backend to a Vite + React frontend.

---

## 1. Directory Structure

The project has been reorganized from phase-based folders into a clean, standard development structure:

```
workspace/
├── pyrightconfig.json        # Pyright path resolver for IDE imports
├── docker-compose.yml        # Unified Multi-container Compose configuration
├── developer_docs.md         # This manual
├── backend/                  # Python REST API & ML training scripts
│   ├── api/                  # Production FastAPI package
│   │   ├── __init__.py
│   │   ├── main.py           # FastAPI endpoints, CORS, database logger
│   │   ├── database.py       # SQLAlchemy engines, tables, sessions
│   │   ├── schemas.py        # Pydantic schemas (V2 compliant)
│   │   └── model_loader.py   # Model cache & high-performance scoring
│   ├── data/                 # Raw/preprocessed datasets & plots
│   ├── models/               # Pickled ML models
│   ├── scripts/              # ML training scripts (EDA, SVD, Hybrid, etc.)
│   ├── tests/                # Automated tests (Pytest)
│   │   ├── test_api.py       # FastAPI endpoint tests
│   │   └── test_models.py    # Model matrix & scoring validation tests
│   └── requirements.txt      # Python dependencies
└── frontend/                 # React SPA
    ├── package.json          # Node dependencies
    ├── vite.config.js        # Vite configs & dev-proxy configuration
    ├── index.html            # SPA Entry HTML
    └── src/                  # React source files
        ├── main.jsx          # React mount entrypoint
        ├── App.jsx           # Layout routes
        ├── index.css         # Reset and global styles
        ├── components/       # Reusable components (Navbar, UI badges, Cards)
        ├── pages/            # Application pages (Home, Similar, Dashboard)
        ├── services/         # Axios / API fetchers
        └── hooks/            # Custom hooks (useApi)
```

---

## 2. Local Setup & Running

### Prerequisites
- Python 3.10+
- Node.js 18+

### Setup virtual environment & dependencies
1. In the project root, create and activate a Python virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```
2. Upgrade pip and install build dependencies:
   ```bash
   pip install --upgrade pip
   pip install wheel setuptools
   ```
3. Install `scikit-surprise` (pre-compiled wheel) and requirements:
   ```bash
   pip install scikit-surprise==1.1.5
   pip install -r backend/requirements.txt
   ```

### Running the Backend
To start the FastAPI production server locally:
```bash
cd backend
..\.venv\Scripts\python -m uvicorn api.main:app --port 8000 --reload
```
The server will start at `http://127.0.0.1:8000`. You can access interactive documentation (Swagger UI) at `http://127.0.0.1:8000/docs`.

### Running the Frontend
1. Navigate to the `frontend/` directory and install packages:
   ```bash
   cd frontend
   npm install
   ```
2. Start the Vite development server:
   ```bash
   npm run dev
   ```
The client will launch at `http://localhost:5173`. Any requests to `/api` are proxied to the local backend port `8000`.

---

## 3. Automated Testing

To run the automated integration and unit test suite:
1. Ensure the virtual environment is active.
2. Run `pytest` inside the `backend/` directory:
   ```bash
   cd backend
   ..\.venv\Scripts\pytest
   ```
This will run 12 tests validating:
- Endpoint routing (`/health`, `/users`, `/recommend`, `/similar`, `/items`, `/rate`)
- Database logging persistence
- ML SVD and cosine similarity ranges
- Seen item filters (checks that recommended items do not contain movies the user has already rated)

---

## 4. ML Pipeline & Performance Optimizations

### Models Included
- **Popularity Recommender (Baseline)**: Scores items based on rating volume and mean ratings.
- **User-Based Collaborative Filtering**: Recommends items using cosine/Pearson similarities between similar users.
- **Item-Based Collaborative Filtering**: Recommends items using similarities between item rating vectors.
- **SVD Model (Matrix Factorization)**: Surprise-based Matrix Factorization tuned via GridSearch.
- **Content-Based Similarity**: Computes TF-IDF vectors of movie genres and scores items using cosine similarity to a user's highly-rated items.
- **Hybrid Model**: Blends SVD predictions ($75\%$) and rescaled content similarity scores ($25\%$).

### Critical Optimization
The content-based and hybrid recommenders score up to 600 candidate items per request. Initially, the code filtered pandas DataFrames dynamically inside the item loop, leading to **600 redundant DataFrame filters** (latency of **1.42s** per request).
We optimized this by extracting the user's top-rated seeds **once** at the start of the prediction workflow. This reduced recommendation request latency to **36ms** (a **39.5x speedup**).
