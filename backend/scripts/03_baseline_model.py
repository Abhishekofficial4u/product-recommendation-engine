"""
Phase 1 — Day 5: Popularity Baseline Model
==========================================
Purpose: build the SIMPLEST possible recommender.
Every model in Phase 2 must beat this.

Two baselines:
  1. Random recommender     → pure chance (floor of all floors)
  2. Popularity recommender → top-rated items (simple but beatable)

Metrics introduced here:
  - RMSE  : how far off are our rating PREDICTIONS?
  - MAE   : mean absolute error of predictions
  - Hit@K : did at least one relevant item appear in top-K?

Output:
  backend/models/popularity_model.pkl
  A printed benchmark table to beat in Phase 2
"""

import pandas as pd
import numpy as np
import pickle
import os
from math import sqrt

# ── Paths ──────────────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA   = os.path.join(BASE, "data")
MODELS = os.path.join(BASE, "models")
os.makedirs(MODELS, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════
# STEP 1 — Load preprocessed data
# ══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1: Loading preprocessed splits")
print("=" * 60)

train = pd.read_csv(os.path.join(DATA, "train.csv"))
test  = pd.read_csv(os.path.join(DATA, "test.csv"))
items = pd.read_csv(os.path.join(DATA, "items.csv"))

print(f"Train: {len(train):,} | Test: {len(test):,}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 2 — Helper: evaluation functions
# ══════════════════════════════════════════════════════════════════════════
def rmse(actual, predicted):
    """Root Mean Squared Error — penalises large errors heavily."""
    actual    = np.array(actual, dtype=float)
    predicted = np.array(predicted, dtype=float)
    return sqrt(np.mean((actual - predicted) ** 2))

def mae(actual, predicted):
    """Mean Absolute Error — average of absolute differences."""
    actual    = np.array(actual, dtype=float)
    predicted = np.array(predicted, dtype=float)
    return np.mean(np.abs(actual - predicted))

def precision_at_k(recommended_ids, relevant_ids, k=10):
    """Of top-K items recommended, what fraction did the user actually like?"""
    rec_k = recommended_ids[:k]
    hits  = len(set(rec_k) & set(relevant_ids))
    return hits / k

def recall_at_k(recommended_ids, relevant_ids, k=10):
    """Of all items the user liked, what fraction appeared in top-K?"""
    if not relevant_ids:
        return 0.0
    rec_k = recommended_ids[:k]
    hits  = len(set(rec_k) & set(relevant_ids))
    return hits / len(relevant_ids)

def evaluate_ranking(model_fn, test_df, train_df, k=10, threshold=4):
    """
    For each user in test set:
      - Get items they rated highly (>= threshold) → 'relevant'
      - Get model's top-K recommendations (excluding already seen items)
      - Calculate Precision@K and Recall@K

    model_fn(user_id, seen_item_ids) → list of item_ids (ranked)
    """
    precisions, recalls = [], []
    test_users = test_df["user_id"].unique()

    for uid in test_users:
        # Items user already rated (in training) — must exclude from recs
        seen = set(train_df[train_df["user_id"] == uid]["item_id"].tolist())

        # Items user liked in test set
        user_test = test_df[test_df["user_id"] == uid]
        relevant  = set(user_test[user_test["rating"] >= threshold]["item_id"].tolist())

        if not relevant:
            continue

        # Get recommendations from model
        recs = model_fn(uid, seen)
        if not recs:
            continue

        precisions.append(precision_at_k(recs, relevant, k))
        recalls.append(recall_at_k(recs, relevant, k))

    return np.mean(precisions), np.mean(recalls)

# ══════════════════════════════════════════════════════════════════════════
# STEP 3 — Baseline 1: Random recommender
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3: Baseline 1 — Random recommender")
print("=" * 60)

all_items = train["item_id"].unique().tolist()
global_mean = train["rating"].mean()

# RMSE: just predict global mean for everything
test_preds_random = [np.random.choice([1,2,3,4,5]) for _ in range(len(test))]
test_preds_mean   = [global_mean] * len(test)

rmse_random = rmse(test["rating"], test_preds_random)
rmse_mean   = rmse(test["rating"], test_preds_mean)
mae_random  = mae(test["rating"],  test_preds_random)
mae_mean    = mae(test["rating"],  test_preds_mean)

print(f"Global mean rating    : {global_mean:.3f}")
print(f"RMSE (random guess)   : {rmse_random:.4f}")
print(f"RMSE (always mean)    : {rmse_mean:.4f}  <- better to always predict the mean!")
print(f"MAE  (random guess)   : {mae_random:.4f}")
print(f"MAE  (always mean)    : {mae_mean:.4f}")

def random_rec_fn(user_id, seen_items, top_k=10):
    unseen = [it for it in all_items if it not in seen_items]
    return list(np.random.choice(unseen, min(top_k, len(unseen)), replace=False))

print("\nEvaluating ranking quality (this takes ~10 sec)...")
# Use a sample of users for speed
sample_users = test["user_id"].unique()[:100]
test_sample  = test[test["user_id"].isin(sample_users)]
train_sample = train[train["user_id"].isin(sample_users)]

p_rand, r_rand = evaluate_ranking(random_rec_fn, test_sample, train_sample, k=10)
print(f"Precision@10 (random) : {p_rand:.4f}")
print(f"Recall@10    (random) : {r_rand:.4f}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 4 — Baseline 2: Popularity model
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 4: Baseline 2 — Popularity recommender")
print("=" * 60)

# Item popularity: weighted average rating with minimum count threshold
# (Bayesian average to avoid items with 1 rating = 5 stars ranking top)
MIN_RATINGS_FOR_POPULAR = 20
C = train["rating"].mean()  # global mean (prior)

item_stats = train.groupby("item_id")["rating"].agg(["mean","count"]).reset_index()
item_stats.columns = ["item_id","avg_rating","n_ratings"]

# Bayesian average: (n*avg + C*m) / (n + m)  where m = minimum count
m = MIN_RATINGS_FOR_POPULAR
item_stats["bayesian_avg"] = (
    (item_stats["n_ratings"] * item_stats["avg_rating"] + C * m)
    / (item_stats["n_ratings"] + m)
)

popular_items = item_stats[item_stats["n_ratings"] >= MIN_RATINGS_FOR_POPULAR] \
                .sort_values("bayesian_avg", ascending=False)

print(f"Items with ≥ {MIN_RATINGS_FOR_POPULAR} ratings : {len(popular_items)}")
print(f"\nTop 10 most popular items:")
top10 = popular_items.head(10).merge(items[["item_id","title"]], on="item_id")
print(top10[["item_id","title","avg_rating","n_ratings","bayesian_avg"]].to_string(index=False))

# RMSE for popularity model: predict item's average rating
test_with_avg = test.merge(item_stats[["item_id","avg_rating"]], on="item_id", how="left")
test_with_avg = test_with_avg.fillna({"avg_rating": global_mean})
rmse_pop = rmse(test_with_avg["rating"], test_with_avg["avg_rating"])
mae_pop  = mae(test_with_avg["rating"],  test_with_avg["avg_rating"])
print(f"\nRMSE (popularity)  : {rmse_pop:.4f}")
print(f"MAE  (popularity)  : {mae_pop:.4f}")

popular_item_ids = popular_items["item_id"].tolist()

def popularity_rec_fn(user_id, seen_items, top_k=10):
    """Recommend top popular items the user hasn't seen yet."""
    recs = [it for it in popular_item_ids if it not in seen_items]
    return recs[:top_k]

p_pop, r_pop = evaluate_ranking(popularity_rec_fn, test_sample, train_sample, k=10)
print(f"Precision@10 (pop) : {p_pop:.4f}")
print(f"Recall@10    (pop) : {r_pop:.4f}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 5 — Save popularity model
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 5: Saving popularity model")
print("=" * 60)

popularity_model = {
    "popular_item_ids":  popular_item_ids,
    "item_stats":        item_stats,
    "global_mean":       global_mean,
    "all_items":         all_items,
}
with open(os.path.join(MODELS, "popularity_model.pkl"), "wb") as f:
    pickle.dump(popularity_model, f)
print("Saved: popularity_model.pkl")

# ══════════════════════════════════════════════════════════════════════════
# STEP 6 — Benchmark table (print and store for comparison in Phase 2)
# ══════════════════════════════════════════════════════════════════════════
print(f"""
{"=" * 60}
PHASE 1 BENCHMARK TABLE — targets for Phase 2 to beat
{"=" * 60}
Model               RMSE      MAE    Precision@10  Recall@10
─────────────────────────────────────────────────────────────
Random guess        {rmse_random:.4f}   {mae_random:.4f}    {p_rand:.4f}       {r_rand:.4f}
Always predict mean {rmse_mean:.4f}   {mae_mean:.4f}      —              —
Popularity baseline {rmse_pop:.4f}   {mae_pop:.4f}    {p_pop:.4f}       {r_pop:.4f}
─────────────────────────────────────────────────────────────
Phase 2 target      < 0.95    < 0.75   > 0.15        > 0.10
{"=" * 60}

→ Phase 2 will build: User-CF, Item-CF, SVD, Hybrid
→ Each must beat the popularity baseline on ALL metrics
→ NEXT STEP: Run 04_collaborative_filtering.py  (Day 6–7)
""")

# Save benchmark for later comparison
benchmark = {
    "random":     {"rmse": rmse_random, "mae": mae_random, "p10": p_rand,  "r10": r_rand},
    "mean":       {"rmse": rmse_mean,   "mae": mae_mean,   "p10": None,    "r10": None},
    "popularity": {"rmse": rmse_pop,    "mae": mae_pop,    "p10": p_pop,   "r10": r_pop},
}
with open(os.path.join(MODELS, "benchmark.pkl"), "wb") as f:
    pickle.dump(benchmark, f)
print("Saved: benchmark.pkl  (used by Phase 2 notebooks)")
