"""
Phase 2 — Day 6 & 7: Collaborative Filtering
=============================================
Two flavours of CF:
  Day 6 → User-based CF : "users who rated things similarly to you also liked X"
  Day 7 → Item-based CF : "items that are rated similarly to what you liked"

Why both?
  User-CF is intuitive but scales poorly (millions of users = slow similarity search)
  Item-CF is more stable — items don't change behaviour, users do

Library: scikit-surprise (wraps CF + matrix ops cleanly)
Output:
  backend/models/user_cf_model.pkl
  backend/models/item_cf_model.pkl
"""

import pandas as pd
import numpy as np
import pickle, os, time
from math import sqrt

from surprise import KNNBasic, KNNWithMeans, KNNWithZScore, Dataset, Reader
from surprise.model_selection import cross_validate, GridSearchCV

# ── Paths ───────────────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA   = os.path.join(BASE, "data")
MODELS = os.path.join(BASE, "models")

# ── Helpers (same as baseline) ───────────────────────────────────────────────
def rmse(actual, predicted):
    return sqrt(np.mean((np.array(actual) - np.array(predicted)) ** 2))

def mae(actual, predicted):
    return np.mean(np.abs(np.array(actual, float) - np.array(predicted, float)))

def precision_at_k(rec_ids, relevant_ids, k=10):
    return len(set(rec_ids[:k]) & set(relevant_ids)) / k

def recall_at_k(rec_ids, relevant_ids, k=10):
    if not relevant_ids: return 0.0
    return len(set(rec_ids[:k]) & set(relevant_ids)) / len(relevant_ids)

def evaluate_ranking(predict_fn, test_df, train_df, all_items, k=10, threshold=4, n_users=150):
    """Evaluate top-K ranking for a sample of users."""
    precisions, recalls = [], []
    users = test_df["user_id"].unique()[:n_users]
    for uid in users:
        seen     = set(train_df[train_df["user_id"] == uid]["item_id"])
        user_test = test_df[test_df["user_id"] == uid]
        relevant  = set(user_test[user_test["rating"] >= threshold]["item_id"])
        if not relevant: continue
        unseen = [it for it in all_items if it not in seen]
        scores = [(it, predict_fn(uid, it)) for it in unseen]
        scores.sort(key=lambda x: x[1], reverse=True)
        recs = [it for it, _ in scores[:k]]
        precisions.append(precision_at_k(recs, relevant, k))
        recalls.append(recall_at_k(recs, relevant, k))
    return np.mean(precisions), np.mean(recalls)

# ══════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("Loading Phase 1 outputs")
print("=" * 60)

train = pd.read_csv(os.path.join(DATA, "train.csv"))
test  = pd.read_csv(os.path.join(DATA, "test.csv"))
with open(os.path.join(MODELS, "benchmark.pkl"), "rb") as f:
    benchmark = pickle.load(f)

all_items = sorted(train["item_id"].unique().tolist())
print(f"Train: {len(train):,}  Test: {len(test):,}  Items: {len(all_items)}")

# Build Surprise Dataset from training data only
reader      = Reader(rating_scale=(1, 5))
surprise_df = train[["user_id", "item_id", "rating"]]
data        = Dataset.load_from_df(surprise_df, reader)
trainset    = data.build_full_trainset()  # full training set (no val split)

# ══════════════════════════════════════════════════════════════════════════
# DAY 6 — USER-BASED CF
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("DAY 6: User-based Collaborative Filtering")
print("=" * 60)
print("""
How it works:
  1. Represent each user as a vector of their ratings
  2. Compute cosine similarity between all user vectors
  3. For user U, find the K most similar users (neighbours)
  4. Predict U's rating for item I = weighted avg of neighbours' ratings for I
  5. Recommend the top-N unrated items with highest predicted rating
""")

# --- Try KNNBasic (pure cosine), KNNWithMeans (subtracts user mean — handles bias)
t0 = time.time()

# Quick K sweep to find best K
print("Sweeping K values (20, 30, 40, 50) with 3-fold CV...")
best_k_user, best_rmse_user = 40, 999

for k_val in [20, 30, 40, 50]:
    algo = KNNWithMeans(
        k=k_val,
        sim_options={"name": "cosine", "user_based": True, "min_support": 3}
    )
    cv = cross_validate(algo, data, measures=["RMSE"], cv=3, verbose=False)
    avg = np.mean(cv["test_rmse"])
    print(f"  K={k_val:3d}  CV-RMSE={avg:.4f}")
    if avg < best_rmse_user:
        best_rmse_user, best_k_user = avg, k_val

print(f"\nBest K for user-CF: {best_k_user}  (CV-RMSE={best_rmse_user:.4f})")

# Train final user-CF model
user_cf = KNNWithMeans(
    k=best_k_user,
    sim_options={"name": "cosine", "user_based": True, "min_support": 3}
)
user_cf.fit(trainset)
print(f"Model trained in {time.time()-t0:.1f}s")

# Test-set RMSE
preds_user = [user_cf.predict(row.user_id, row.item_id).est for row in test.itertuples()]
rmse_user  = rmse(test["rating"], preds_user)
mae_user   = mae(test["rating"],  preds_user)
print(f"Test RMSE  : {rmse_user:.4f}  (baseline was {benchmark['popularity']['rmse']:.4f})")
print(f"Test MAE   : {mae_user:.4f}")

# Ranking metrics
print("\nEvaluating Precision@10 / Recall@10 (100 users sample)...")
p_user, r_user = evaluate_ranking(
    lambda uid, iid: user_cf.predict(uid, iid).est,
    test, train, all_items, k=10, n_users=100
)
print(f"Precision@10 : {p_user:.4f}  (baseline was {benchmark['popularity']['p10']:.4f})")
print(f"Recall@10    : {r_user:.4f}")

# Demo: show top-10 for a sample user
sample_uid = train["user_id"].iloc[0]
seen_by_user = set(train[train["user_id"] == sample_uid]["item_id"])
unseen = [it for it in all_items if it not in seen_by_user]
user_scores = [(it, user_cf.predict(sample_uid, it).est) for it in unseen[:300]]
user_scores.sort(key=lambda x: x[1], reverse=True)
print(f"\nTop-5 recommendations for user {sample_uid}:")
for item_id, score in user_scores[:5]:
    print(f"  item_id={item_id:5d}  predicted_rating={score:.2f}")

# ══════════════════════════════════════════════════════════════════════════
# DAY 7 — ITEM-BASED CF
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("DAY 7: Item-based Collaborative Filtering")
print("=" * 60)
print("""
How it works:
  1. Represent each item as a vector of all user ratings it received
  2. Compute Pearson correlation between all item vectors
     (Pearson handles the fact that some users always rate high/low)
  3. For user U and unseen item I:
       Look at items U has rated that are most similar to I
       Predict rating = weighted avg of those similar items' ratings by U
  4. Recommend items with highest predicted rating

Why Pearson for item-CF?
  Cosine treats "user A rates everything 5" same as
  "user A rates everything 1, except this item = 5"
  Pearson normalises by each user's mean → more accurate
""")

t0 = time.time()
print("Sweeping K values (20, 30, 40, 50) with 3-fold CV...")
best_k_item, best_rmse_item = 40, 999

for k_val in [20, 30, 40, 50]:
    algo = KNNWithMeans(
        k=k_val,
        sim_options={"name": "pearson", "user_based": False, "min_support": 3}
    )
    cv = cross_validate(algo, data, measures=["RMSE"], cv=3, verbose=False)
    avg = np.mean(cv["test_rmse"])
    print(f"  K={k_val:3d}  CV-RMSE={avg:.4f}")
    if avg < best_rmse_item:
        best_rmse_item, best_k_item = avg, k_val

print(f"\nBest K for item-CF: {best_k_item}  (CV-RMSE={best_rmse_item:.4f})")

item_cf = KNNWithMeans(
    k=best_k_item,
    sim_options={"name": "pearson", "user_based": False, "min_support": 3}
)
item_cf.fit(trainset)
print(f"Model trained in {time.time()-t0:.1f}s")

preds_item = [item_cf.predict(row.user_id, row.item_id).est for row in test.itertuples()]
rmse_item  = rmse(test["rating"], preds_item)
mae_item   = mae(test["rating"],  preds_item)
print(f"Test RMSE  : {rmse_item:.4f}")
print(f"Test MAE   : {mae_item:.4f}")

print("\nEvaluating Precision@10 / Recall@10 (100 users sample)...")
p_item, r_item = evaluate_ranking(
    lambda uid, iid: item_cf.predict(uid, iid).est,
    test, train, all_items, k=10, n_users=100
)
print(f"Precision@10 : {p_item:.4f}")
print(f"Recall@10    : {r_item:.4f}")

# ══════════════════════════════════════════════════════════════════════════
# SAVE MODELS
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Saving CF models")
print("=" * 60)

with open(os.path.join(MODELS, "user_cf_model.pkl"), "wb") as f:
    pickle.dump(user_cf, f)
print("Saved: user_cf_model.pkl")

with open(os.path.join(MODELS, "item_cf_model.pkl"), "wb") as f:
    pickle.dump(item_cf, f)
print("Saved: item_cf_model.pkl")

# Update benchmark
benchmark["user_cf"] = {"rmse": rmse_user, "mae": mae_user, "p10": p_user, "r10": r_user}
benchmark["item_cf"] = {"rmse": rmse_item, "mae": mae_item, "p10": p_item, "r10": r_item}
with open(os.path.join(MODELS, "benchmark.pkl"), "wb") as f:
    pickle.dump(benchmark, f)
print("Updated: benchmark.pkl")

# ══════════════════════════════════════════════════════════════════════════
# COMPARISON TABLE
# ══════════════════════════════════════════════════════════════════════════
pop = benchmark["popularity"]
print(f"""
{"=" * 65}
RESULTS SO FAR  (lower RMSE = better | higher P@10 = better)
{"=" * 65}
Model               RMSE      MAE    Precision@10  Recall@10
─────────────────────────────────────────────────────────────
Popularity (base)   {pop['rmse']:.4f}   {pop['mae']:.4f}    {pop['p10']:.4f}       {pop['r10']:.4f}
User-based CF       {rmse_user:.4f}   {mae_user:.4f}    {p_user:.4f}       {r_user:.4f}
Item-based CF       {rmse_item:.4f}   {mae_item:.4f}    {p_item:.4f}       {r_item:.4f}
─────────────────────────────────────────────────────────────
Phase 2 target      < 0.95    < 0.75   > 0.15        > 0.10
{"=" * 65}

→ NEXT: Run 05_svd_model.py  (Day 8)
""")
