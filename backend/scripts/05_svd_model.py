"""
Phase 2 — Day 8: SVD Matrix Factorization
==========================================
SVD = Singular Value Decomposition
This is where the big RMSE jump happens.

Why SVD beats CF:
  CF uses raw similarity between users/items.
  SVD learns LATENT FACTORS — hidden abstract features.

  Example latent factors for movies might be:
    Factor 1: "how action-heavy is this?"
    Factor 2: "how romantic is this?"
    Factor 3: "how comedic is this?"
  The model learns these factors purely from rating patterns — 
  nobody tells it what the factors mean.

  Each user gets a "taste vector" in latent space.
  Each item gets a "feature vector" in latent space.
  Predicted rating = dot product of user vector · item vector
                   + user bias + item bias + global mean

Hyperparameters to tune:
  n_factors : number of latent dimensions (50–200)
  n_epochs  : gradient descent iterations (20–50)
  lr_all    : learning rate
  reg_all   : L2 regularisation (prevents overfitting)

Output:
  backend/models/svd_model.pkl
"""

import pandas as pd
import numpy as np
import pickle, os, time
from math import sqrt
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate, GridSearchCV

# ── Paths ───────────────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA   = os.path.join(BASE, "data")
MODELS = os.path.join(BASE, "models")

def rmse(actual, predicted):
    return sqrt(np.mean((np.array(actual, float) - np.array(predicted, float)) ** 2))

def mae(actual, predicted):
    return np.mean(np.abs(np.array(actual, float) - np.array(predicted, float)))

def precision_at_k(rec_ids, relevant_ids, k=10):
    return len(set(rec_ids[:k]) & set(relevant_ids)) / k

def recall_at_k(rec_ids, relevant_ids, k=10):
    if not relevant_ids: return 0.0
    return len(set(rec_ids[:k]) & set(relevant_ids)) / len(relevant_ids)

def ndcg_at_k(rec_ids, relevant_ids, k=10):
    """
    Normalised Discounted Cumulative Gain.
    Rewards putting the most relevant items at the TOP of the list.
    A hit at rank 1 scores more than a hit at rank 10.
    """
    dcg = 0.0
    for i, item in enumerate(rec_ids[:k]):
        if item in relevant_ids:
            dcg += 1.0 / np.log2(i + 2)   # log2(rank+1), ranks start at 1
    # Ideal DCG: all relevant items ranked first
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0

def evaluate_ranking(predict_fn, test_df, train_df, all_items, k=10, threshold=4, n_users=150):
    precisions, recalls, ndcgs = [], [], []
    users = test_df["user_id"].unique()[:n_users]
    for uid in users:
        seen      = set(train_df[train_df["user_id"] == uid]["item_id"])
        user_test = test_df[test_df["user_id"] == uid]
        relevant  = set(user_test[user_test["rating"] >= threshold]["item_id"])
        if not relevant: continue
        unseen = [it for it in all_items if it not in seen]
        scores = [(it, predict_fn(uid, it)) for it in unseen[:500]]   # limit for speed
        scores.sort(key=lambda x: x[1], reverse=True)
        recs = [it for it, _ in scores[:k]]
        precisions.append(precision_at_k(recs, relevant, k))
        recalls.append(recall_at_k(recs, relevant, k))
        ndcgs.append(ndcg_at_k(recs, relevant, k))
    return np.mean(precisions), np.mean(recalls), np.mean(ndcgs)

# ══════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("Loading data")
print("=" * 60)

train = pd.read_csv(os.path.join(DATA, "train.csv"))
test  = pd.read_csv(os.path.join(DATA, "test.csv"))
with open(os.path.join(MODELS, "benchmark.pkl"), "rb") as f:
    benchmark = pickle.load(f)

all_items   = sorted(train["item_id"].unique().tolist())
reader      = Reader(rating_scale=(1, 5))
data        = Dataset.load_from_df(train[["user_id","item_id","rating"]], reader)
trainset    = data.build_full_trainset()

print(f"Train: {len(train):,}  |  Test: {len(test):,}  |  Items: {len(all_items)}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 1 — Quick exploration: what do latent factors look like?
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 1: Intuition — training a tiny SVD to visualise factors")
print("=" * 60)

# Train a tiny model just to show factor structure
tiny_svd = SVD(n_factors=2, n_epochs=30, verbose=False)
tiny_svd.fit(trainset)
print("Trained SVD with 2 factors (for visualisation only)")
print(f"User factor matrix shape : {tiny_svd.pu.shape}  (users × 2 latent dims)")
print(f"Item factor matrix shape : {tiny_svd.qi.shape}  (items × 2 latent dims)")
print(f"Sample user factor vector: {tiny_svd.pu[0].round(3)}")
print(f"Sample item factor vector: {tiny_svd.qi[0].round(3)}")
print("The model learned these numbers purely from rating patterns.")

# ══════════════════════════════════════════════════════════════════════════
# STEP 2 — Hyperparameter tuning with GridSearchCV
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2: Hyperparameter tuning (GridSearchCV, 3-fold CV)")
print("=" * 60)
print("This searches for the best combination of n_factors, n_epochs, lr_all, reg_all.")
print("Warning: takes 3–5 minutes. Go make tea.\n")

param_grid = {
    "n_factors": [50, 100, 150],
    "n_epochs":  [20, 30],
    "lr_all":    [0.005, 0.010],
    "reg_all":   [0.02, 0.05],
}

t0 = time.time()
gs = GridSearchCV(SVD, param_grid, measures=["rmse", "mae"], cv=3, n_jobs=-1)
gs.fit(data)

best_params = gs.best_params["rmse"]
best_cv_rmse = gs.best_score["rmse"]
elapsed = time.time() - t0

print(f"Search complete in {elapsed:.0f}s")
print(f"Best CV-RMSE   : {best_cv_rmse:.4f}")
print(f"Best params    : {best_params}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 3 — Train final SVD with best params
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3: Training final SVD model with best hyperparameters")
print("=" * 60)

t0 = time.time()
svd_model = SVD(**best_params, verbose=False)
svd_model.fit(trainset)
print(f"Final model trained in {time.time()-t0:.1f}s")
print(f"Factor matrix sizes — Users: {svd_model.pu.shape}, Items: {svd_model.qi.shape}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 4 — Full 5-fold cross-validation on best model
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 4: 5-fold cross-validation (the number you report in your paper)")
print("=" * 60)

cv_results = cross_validate(
    SVD(**best_params, verbose=False),
    data, measures=["RMSE", "MAE"], cv=5, verbose=True
)
cv_rmse_mean = np.mean(cv_results["test_rmse"])
cv_rmse_std  = np.std(cv_results["test_rmse"])
cv_mae_mean  = np.mean(cv_results["test_mae"])
print(f"\nCV RMSE : {cv_rmse_mean:.4f} ± {cv_rmse_std:.4f}")
print(f"CV MAE  : {cv_mae_mean:.4f}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 5 — Held-out test set evaluation
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 5: Held-out test set evaluation")
print("=" * 60)

preds_svd = [svd_model.predict(row.user_id, row.item_id).est for row in test.itertuples()]
rmse_svd  = rmse(test["rating"], preds_svd)
mae_svd   = mae(test["rating"],  preds_svd)
print(f"Test RMSE : {rmse_svd:.4f}")
print(f"Test MAE  : {mae_svd:.4f}")

print("\nEvaluating ranking metrics (150 users)...")
p_svd, r_svd, ndcg_svd = evaluate_ranking(
    lambda uid, iid: svd_model.predict(uid, iid).est,
    test, train, all_items, k=10, n_users=150
)
print(f"Precision@10 : {p_svd:.4f}")
print(f"Recall@10    : {r_svd:.4f}")
print(f"NDCG@10      : {ndcg_svd:.4f}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 6 — Demo predictions
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 6: Demo — top-10 for a sample user")
print("=" * 60)

sample_uid  = train["user_id"].iloc[0]
seen        = set(train[train["user_id"] == sample_uid]["item_id"])
unseen      = [it for it in all_items if it not in seen]
svd_scores  = [(it, svd_model.predict(sample_uid, it).est) for it in unseen]
svd_scores.sort(key=lambda x: x[1], reverse=True)

print(f"Top-10 SVD recommendations for user {sample_uid}:")
print(f"{'Rank':<5} {'item_id':<10} {'predicted_rating':<18}")
print("-" * 35)
for rank, (item_id, score) in enumerate(svd_scores[:10], 1):
    print(f"{rank:<5} {item_id:<10} {score:<18.3f}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 7 — Save model and update benchmark
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 7: Saving SVD model")
print("=" * 60)

svd_artifact = {
    "model":       svd_model,
    "best_params": best_params,
    "cv_rmse":     cv_rmse_mean,
    "cv_rmse_std": cv_rmse_std,
    "all_items":   all_items,
}
with open(os.path.join(MODELS, "svd_model.pkl"), "wb") as f:
    pickle.dump(svd_artifact, f)
print("Saved: svd_model.pkl")

benchmark["svd"] = {
    "rmse": rmse_svd, "mae": mae_svd,
    "p10": p_svd, "r10": r_svd, "ndcg10": ndcg_svd,
    "cv_rmse": cv_rmse_mean, "best_params": best_params
}
with open(os.path.join(MODELS, "benchmark.pkl"), "wb") as f:
    pickle.dump(benchmark, f)
print("Updated: benchmark.pkl")

# ══════════════════════════════════════════════════════════════════════════
# RESULTS TABLE
# ══════════════════════════════════════════════════════════════════════════
pop  = benchmark["popularity"]
ucf  = benchmark.get("user_cf",  {"rmse": "—", "p10": "—", "r10": "—"})
icf  = benchmark.get("item_cf",  {"rmse": "—", "p10": "—", "r10": "—"})

print(f"""
{"=" * 65}
RESULTS SO FAR
{"=" * 65}
Model               RMSE      MAE    Precision@10  Recall@10  NDCG@10
─────────────────────────────────────────────────────────────────────
Popularity (base)   {pop['rmse']:.4f}                 {pop['p10']:.4f}        {pop['r10']:.4f}
User-based CF       {ucf['rmse']:.4f}                 {ucf['p10']:.4f}        {ucf['r10']:.4f}
Item-based CF       {icf['rmse']:.4f}                 {icf['p10']:.4f}        {icf['r10']:.4f}
SVD (tuned) ★       {rmse_svd:.4f}   {mae_svd:.4f}    {p_svd:.4f}        {r_svd:.4f}      {ndcg_svd:.4f}
─────────────────────────────────────────────────────────────────────
Best params: {best_params}
{"=" * 65}

SVD is now your primary single model.
→ NEXT: Run 06_content_based.py  (Day 9)
""")
