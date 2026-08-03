"""
backend/api/model_loader.py
============================
Loads every trained model ONCE at API startup.
All prediction functions live here and are imported by main.py.

Design principle: models are loaded into memory once.
Every request calls a function here — never loads from disk per request.
"""

import pickle
import os
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR   = os.path.join(BASE_DIR, "data")

# ── Global model objects (populated at startup) ─────────────────────────────
_popularity   = None
_user_cf      = None
_item_cf      = None
_svd_art      = None      # dict with "model" key
_content_art  = None      # dict with cosine_sim, item_to_idx etc.
_hybrid_art   = None      # dict with best_alpha etc.
_benchmark    = None
_train_df     = None
_items_df     = None
_all_items    = None
_global_mean  = None


def load_all_models() -> List[str]:
    """
    Called once during FastAPI startup.
    Returns list of successfully loaded model names.
    """
    global _popularity, _user_cf, _item_cf, _svd_art
    global _content_art, _hybrid_art, _benchmark
    global _train_df, _items_df, _all_items, _global_mean

    loaded = []

    # Data
    _train_df  = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    _items_df  = pd.read_csv(os.path.join(DATA_DIR, "items.csv"))
    _all_items = sorted(_train_df["item_id"].unique().tolist())
    _global_mean = float(_train_df["rating"].mean())

    def _load(name: str, key: str):
        path = os.path.join(MODELS_DIR, f"{name}.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
        raise FileNotFoundError(f"Model not found: {path}")

    try:
        _popularity  = _load("popularity_model",  "popularity")
        loaded.append("popularity")
    except FileNotFoundError as e:
        print(f"  WARNING: {e}")

    try:
        _user_cf = _load("user_cf_model", "user_cf")
        loaded.append("user_cf")
    except FileNotFoundError as e:
        print(f"  WARNING: {e}")

    try:
        _item_cf = _load("item_cf_model", "item_cf")
        loaded.append("item_cf")
    except FileNotFoundError as e:
        print(f"  WARNING: {e}")

    try:
        _svd_art = _load("svd_model", "svd")
        loaded.append("svd")
    except FileNotFoundError as e:
        print(f"  WARNING: {e}")

    try:
        _content_art = _load("content_model", "content")
        loaded.append("content")
    except FileNotFoundError as e:
        print(f"  WARNING: {e}")

    try:
        _hybrid_art = _load("hybrid_model", "hybrid")
        loaded.append("hybrid")
    except FileNotFoundError as e:
        print(f"  WARNING: {e}")

    try:
        _benchmark = _load("benchmark", "benchmark")
        loaded.append("benchmark")
    except FileNotFoundError as e:
        print(f"  WARNING: {e}")

    print(f"  Loaded models: {loaded}")
    return [m for m in loaded if m != "benchmark"]


# ── Internal helpers ─────────────────────────────────────────────────────────

def _get_seen(user_id: int) -> set:
    """Items user has already rated in training data."""
    if _train_df is None:
        raise RuntimeError("_train_df is not loaded. Ensure load_all_models() has been called at startup.")
    return set(_train_df[_train_df["user_id"] == user_id]["item_id"].tolist())


def _enrich(item_id: int, score: float, rank: int) -> Dict:
    """Add title and genre metadata to a scored item."""
    if _items_df is None:
        raise RuntimeError("_items_df is not loaded. Ensure load_all_models() has been called at startup.")
    row = _items_df[_items_df["item_id"] == item_id]
    title  = row["title"].values[0]  if not row.empty else f"Item {item_id}"
    genres = row["genres"].values[0] if not row.empty else "Unknown"
    return {
        "item_id":          item_id,
        "title":            title,
        "genres":           genres,
        "predicted_rating": round(min(5.0, max(1.0, float(score))), 3),
        "rank":             rank,
    }


def _content_score(seeds: List[int], item_id: int) -> float:
    """Cosine similarity score from content model using pre-calculated seeds."""
    if _content_art is None or not seeds:
        return 0.0
    item_to_idx = _content_art["item_to_idx"]
    if item_id not in item_to_idx:
        return 0.0
    cosine_sim = _content_art["cosine_sim"]
    item_idx   = item_to_idx[item_id]
    # FIX: filter out seeds not present in item_to_idx to avoid KeyError
    valid_seeds = [s for s in seeds if s in item_to_idx]
    if not valid_seeds:
        return 0.0
    return float(np.mean([cosine_sim[item_to_idx[s]][item_idx] for s in valid_seeds]))


def _hybrid_score(user_id: int, item_id: int, seeds: List[int]) -> float:
    """Blended SVD + content score."""
    alpha = _hybrid_art["best_alpha"] if _hybrid_art else 0.75
    # FIX: _global_mean may be None if load_all_models() was never called
    global_mean_fallback = _global_mean if _global_mean is not None else 3.0
    # SVD prediction
    # FIX: guard _svd_art before subscripting to avoid 'None is not subscriptable'
    try:
        if _svd_art is None:
            raise ValueError("SVD model not loaded")
        svd_val = _svd_art["model"].predict(user_id, item_id).est
    except Exception:
        svd_val = global_mean_fallback
    # Content score rescaled to 1-5
    cs_val = _content_score(seeds, item_id)
    content_scaled = 1 + cs_val * 4
    return alpha * svd_val + (1 - alpha) * content_scaled


# ── Public prediction functions (called by main.py) ─────────────────────────

def recommend(user_id: int, top_k: int = 10, model: str = "hybrid") -> Dict:
    """
    Core recommendation function.
    Returns dict with items list + metadata.
    """
    # FIX: guard _all_items before iterating to avoid 'None is not iterable'
    if _all_items is None:
        raise RuntimeError(
            "_all_items is not loaded. Ensure load_all_models() has been called at startup."
        )
    seen           = _get_seen(user_id)
    n_user_ratings = len(seen)
    cold_start     = n_user_ratings < 3
    unseen         = [it for it in _all_items if it not in seen]

    # Precalculate seeds for content-based matching to avoid redundant O(N) DataFrame lookups
    seeds = []
    if _content_art and _train_df is not None:
        user_ratings = _train_df[_train_df["user_id"] == user_id]
        if not user_ratings.empty:
            item_to_idx = _content_art["item_to_idx"]
            raw_seeds = user_ratings.sort_values("rating", ascending=False).head(5)["item_id"].tolist()
            seeds = [int(s) for s in raw_seeds if s in item_to_idx]

    # Score all unseen items with the chosen model
    if model == "hybrid" and _hybrid_art and _svd_art and _content_art:
        if cold_start:
            # Fall back to content-only for cold users
            scores = [(it, _content_score(seeds, it)) for it in unseen[:600]]
        else:
            scores = [(it, _hybrid_score(user_id, it, seeds)) for it in unseen[:600]]

    elif model == "svd" and _svd_art:
        scores = [(it, _svd_art["model"].predict(user_id, it).est) for it in unseen[:600]]

    elif model == "user_cf" and _user_cf:
        scores = [(it, _user_cf.predict(user_id, it).est) for it in unseen[:600]]

    elif model == "item_cf" and _item_cf:
        scores = [(it, _item_cf.predict(user_id, it).est) for it in unseen[:600]]

    elif model == "content" and _content_art:
        scores = [(it, _content_score(seeds, it)) for it in unseen[:600]]

    elif model == "popularity" and _popularity:
        popular = _popularity["popular_item_ids"]
        scores  = [(it, 5.0 - i * 0.001) for i, it in enumerate(popular) if it not in seen]

    else:
        # Ultimate fallback: popularity
        popular = _popularity["popular_item_ids"] if _popularity else _all_items
        scores  = [(it, 5.0 - i * 0.001) for i, it in enumerate(popular) if it not in seen]

    scores.sort(key=lambda x: x[1], reverse=True)
    top = scores[:top_k]

    items = [_enrich(iid, score, rank + 1) for rank, (iid, score) in enumerate(top)]
    return {"cold_start": cold_start, "items": items}


def similar_items(item_id: int, top_k: int = 5) -> Dict:
    """
    Returns top-K most similar items using content cosine similarity.
    Used for GET /similar/{item_id}.
    """
    if _content_art is None:
        return {"items": []}

    # FIX: guard _items_df before subscripting inside this function
    if _items_df is None:
        return {"items": []}

    item_to_idx = _content_art["item_to_idx"]
    idx_to_item = _content_art["idx_to_item"]
    cosine_sim  = _content_art["cosine_sim"]

    if item_id not in item_to_idx:
        return {"items": []}

    idx = item_to_idx[item_id]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores.sort(key=lambda x: x[1], reverse=True)
    # FIX: use dict.get() to avoid KeyError for indices missing from idx_to_item
    sim_scores = [
        (i, s) for i, s in sim_scores
        if idx_to_item.get(i) is not None and idx_to_item[i] != item_id
    ]

    items = []
    for rank, (i, sim) in enumerate(sim_scores[:top_k], 1):
        iid = idx_to_item[i]
        row = _items_df[_items_df["item_id"] == iid]
        items.append({
            "item_id":    iid,
            "title":      row["title"].values[0]  if not row.empty else f"Item {iid}",
            "genres":     row["genres"].values[0] if not row.empty else "Unknown",
            "similarity": round(float(sim), 4),
            "rank":       rank,
        })

    # Source item metadata
    src_row = _items_df[_items_df["item_id"] == item_id]
    return {
        "source_title":  src_row["title"].values[0]  if not src_row.empty else f"Item {item_id}",
        "source_genres": src_row["genres"].values[0] if not src_row.empty else "Unknown",
        "items": items,
    }


def get_metrics() -> List[Dict]:
    """Returns benchmark metrics for all models — used by GET /metrics."""
    if _benchmark is None:
        return []
    order  = ["popularity","user_cf","item_cf","content","svd","hybrid"]
    labels = {"popularity":"Popularity Baseline","user_cf":"User-based CF",
              "item_cf":"Item-based CF","content":"Content-Based",
              "svd":"SVD (tuned)","hybrid":"Hybrid ★"}
    result = []
    for key in order:
        if key in _benchmark:
            bm = _benchmark[key]
            result.append({
                "model":  labels.get(key, key),
                "rmse":   round(bm["rmse"],  4) if bm.get("rmse")  else None,
                "p10":    round(bm["p10"],   4) if bm.get("p10")   else None,
                "r10":    round(bm["r10"],   4) if bm.get("r10")   else None,
                "ndcg10": round(bm["ndcg10"],4) if bm.get("ndcg10")else None,
            })
    return result


def get_health_info() -> Dict:
    """Data for GET /health."""
    loaded = []
    for name, obj in [("popularity",_popularity),("user_cf",_user_cf),
                      ("item_cf",_item_cf),("svd",_svd_art),
                      ("content",_content_art),("hybrid",_hybrid_art)]:
        if obj is not None:
            loaded.append(name)
    return {
        "total_users":   int(_train_df["user_id"].nunique()) if _train_df is not None else 0,
        "total_items":   int(_train_df["item_id"].nunique()) if _train_df is not None else 0,
        "total_ratings": int(len(_train_df))                 if _train_df is not None else 0,
        "models_loaded": loaded,
    }


def get_all_items(limit: int = 50) -> List[Dict]:
    """Returns a sample of items for the frontend item browser."""
    if _items_df is None:
        return []
    return _items_df.head(limit)[["item_id","title","genres"]].to_dict(orient="records")


def get_item_by_id(item_id: int) -> Optional[Dict]:
    """Returns metadata for a single item."""
    if _items_df is None:
        return None
    row = _items_df[_items_df["item_id"] == item_id]
    if row.empty:
        return None
    return row.iloc[0][["item_id","title","genres"]].to_dict()


def get_users() -> List[int]:
    """Returns all valid user IDs for the frontend dropdown."""
    if _train_df is None:
        return []
    return sorted(_train_df["user_id"].unique().tolist())
