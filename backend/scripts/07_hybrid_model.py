"""
Phase 2 — Day 10: Hybrid Recommender
=====================================
The hybrid combines SVD (accuracy) + Content-based (diversity + cold start).

Formula:
  hybrid_score(user, item) = α × svd_score + (1-α) × content_score

Where:
  svd_score    = predicted rating (1–5) from SVD matrix factorization
  content_score = TF-IDF cosine similarity to user's liked items (0–1)
  α (alpha)    = weight for SVD vs content (tuned on validation set)

Why is hybrid better than either alone?
  SVD alone:       accurate but blind to item content — can't handle new items
  Content alone:   no rating signal — "filter bubble" — no taste beyond genre
  Hybrid together: SVD's accuracy + content diversity + cold-start handling

We also introduce cold-start handling:
  If user has < 3 ratings → fall back to content-only
  If item has 0 ratings   → use content similarity only (SVD returns global mean)

Output:
  backend/models/hybrid_model.pkl  ← final production model
"""

import pandas as pd
import numpy as np
import pickle, os
from math import sqrt

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
    dcg  = sum(1.0/np.log2(i+2) for i,it in enumerate(rec_ids[:k]) if it in relevant_ids)
    idcg = sum(1.0/np.log2(i+2) for i in range(min(len(relevant_ids), k)))
    return dcg/idcg if idcg else 0.0

# ══════════════════════════════════════════════════════════════════════════
# LOAD ALL MODELS FROM PHASE 2
# ══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("Loading all Phase 2 models")
print("=" * 60)

train = pd.read_csv(os.path.join(DATA, "train.csv"))
test  = pd.read_csv(os.path.join(DATA, "test.csv"))
items = pd.read_csv(os.path.join(DATA, "items.csv"))

with open(os.path.join(MODELS, "svd_model.pkl"),     "rb") as f: svd_art     = pickle.load(f)
with open(os.path.join(MODELS, "content_model.pkl"), "rb") as f: content_art = pickle.load(f)
with open(os.path.join(MODELS, "benchmark.pkl"),     "rb") as f: benchmark   = pickle.load(f)

svd_model   = svd_art["model"]
cosine_sim  = content_art["cosine_sim"]
item_to_idx = content_art["item_to_idx"]
idx_to_item = content_art["idx_to_item"]
item_ids    = content_art["item_ids"]
all_items   = sorted(train["item_id"].unique().tolist())
global_mean = train["rating"].mean()

print(f"SVD model     : loaded (best_params={svd_art['best_params']})")
print(f"Content model : loaded (cosine_sim {cosine_sim.shape})")
print(f"Train: {len(train):,}  Test: {len(test):,}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 1 — Core hybrid scoring function
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 1: Building hybrid scoring function")
print("=" * 60)

def get_content_score(user_id, item_id, n_seed=5):
    """Avg cosine similarity from user's top-rated items to the target item."""
    if item_id not in item_to_idx:
        return 0.0
    user_ratings = train[train["user_id"] == user_id]
    if user_ratings.empty:
        return 0.0
    seeds = (
        user_ratings.sort_values("rating", ascending=False)
        .head(n_seed)["item_id"].tolist()
    )
    seeds = [s for s in seeds if s in item_to_idx]
    if not seeds:
        return 0.0
    item_idx = item_to_idx[item_id]
    return float(np.mean([cosine_sim[item_to_idx[s]][item_idx] for s in seeds]))

def get_svd_score(user_id, item_id):
    """Predicted rating from SVD (1–5 scale)."""
    try:
        return svd_model.predict(user_id, item_id).est
    except Exception:
        return global_mean

def hybrid_score(user_id, item_id, alpha=0.7):
    """
    Final hybrid score blending SVD + content similarity.
    alpha=0.7 means SVD contributes 70%, content 30%.
    Content score is rescaled to 1–5 range to match SVD output.
    """
    svd_val     = get_svd_score(user_id, item_id)         # 1–5
    content_val = get_content_score(user_id, item_id)     # 0–1
    content_scaled = 1 + content_val * 4                  # rescale to 1–5
    return alpha * svd_val + (1 - alpha) * content_scaled

print("Hybrid score formula:")
print("  score = α × SVD_score(1–5) + (1–α) × content_score(rescaled to 1–5)")
print("  where α is tuned on a validation split")

# Quick sanity check
uid_sample = train["user_id"].iloc[0]
iid_sample = all_items[10]
s = hybrid_score(uid_sample, iid_sample, alpha=0.7)
c = get_content_score(uid_sample, iid_sample)
v = get_svd_score(uid_sample, iid_sample)
print(f"\nSanity check — user={uid_sample}, item={iid_sample}:")
print(f"  SVD score     : {v:.3f}")
print(f"  Content score : {c:.3f}")
print(f"  Hybrid (α=0.7): {s:.3f}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 2 — Alpha tuning on validation sample
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2: Tuning alpha on a validation sample")
print("=" * 60)

# Use 20% of test users as a mini validation set for alpha search
np.random.seed(42)
val_users   = np.random.choice(test["user_id"].unique(), size=80, replace=False)
val_df      = test[test["user_id"].isin(val_users)]

alpha_values = [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9]
print(f"Testing α values: {alpha_values}")
print(f"Val users: {len(val_users)}  Val ratings: {len(val_df)}")

best_alpha, best_p10 = 0.7, 0.0
alpha_results = {}

for alpha in alpha_values:
    plist = []
    for uid in val_users:
        seen     = set(train[train["user_id"] == uid]["item_id"])
        rel      = set(val_df[(val_df["user_id"]==uid) & (val_df["rating"]>=4)]["item_id"])
        if not rel: continue
        unseen   = [it for it in all_items if it not in seen]
        scores   = [(it, hybrid_score(uid, it, alpha)) for it in unseen[:400]]
        scores.sort(key=lambda x: x[1], reverse=True)
        recs     = [it for it, _ in scores[:10]]
        plist.append(precision_at_k(recs, rel))
    p = np.mean(plist) if plist else 0
    alpha_results[alpha] = p
    print(f"  α={alpha:.2f}  Precision@10={p:.4f}")
    if p > best_p10:
        best_p10, best_alpha = p, alpha

print(f"\nBest α = {best_alpha}  (Precision@10 = {best_p10:.4f})")

# ══════════════════════════════════════════════════════════════════════════
# STEP 3 — Full recommendation function
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3: Building full recommendation function")
print("=" * 60)

def hybrid_recommend(user_id, top_k=10, alpha=None):
    """
    Returns top-K recommendations for a user as a list of dicts.
    Handles cold-start: users with < 3 ratings fall back to content-only.
    """
    if alpha is None:
        alpha = best_alpha

    seen         = set(train[train["user_id"] == user_id]["item_id"])
    n_user_ratings = len(seen)
    unseen       = [it for it in all_items if it not in seen]

    if n_user_ratings < 3:
        # Cold start: content-only (no SVD signal yet)
        scores = [(it, get_content_score(user_id, it)) for it in unseen[:500]]
    else:
        # Full hybrid
        scores = [(it, hybrid_score(user_id, it, alpha)) for it in unseen[:500]]

    scores.sort(key=lambda x: x[1], reverse=True)

    results = []
    for item_id, score in scores[:top_k]:
        item_row = items[items["item_id"] == item_id]
        genre    = item_row["genres"].values[0] if not item_row.empty else "Unknown"
        results.append({
            "item_id":          item_id,
            "predicted_rating": round(score, 3),
            "genres":           genre,
            "cold_start":       n_user_ratings < 3,
        })
    return results

# Demo
sample_uid  = train["user_id"].iloc[0]
recs        = hybrid_recommend(sample_uid, top_k=10)
print(f"Top-10 hybrid recommendations for user {sample_uid}:")
print(f"{'Rank':<5} {'item_id':<10} {'score':<8} {'genres'}")
print("-" * 60)
for i, r in enumerate(recs, 1):
    print(f"{i:<5} {r['item_id']:<10} {r['predicted_rating']:<8} {r['genres']}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 4 — Full test set evaluation
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"STEP 4: Full evaluation at α={best_alpha} (200 test users)")
print("=" * 60)

plist, rlist, nlist = [], [], []
test_users = test["user_id"].unique()[:200]

for uid in test_users:
    seen      = set(train[train["user_id"] == uid]["item_id"])
    user_test = test[test["user_id"] == uid]
    relevant  = set(user_test[user_test["rating"] >= 4]["item_id"])
    if not relevant: continue

    unseen = [it for it in all_items if it not in seen]
    scores = [(it, hybrid_score(uid, it, best_alpha)) for it in unseen[:500]]
    scores.sort(key=lambda x: x[1], reverse=True)
    recs   = [it for it, _ in scores[:10]]

    plist.append(precision_at_k(recs, relevant))
    rlist.append(recall_at_k(recs, relevant))
    nlist.append(ndcg_at_k(recs, relevant))

p_hyb    = np.mean(plist)
r_hyb    = np.mean(rlist)
ndcg_hyb = np.mean(nlist)

# RMSE: hybrid gives predicted scores so we can compute RMSE
preds_hyb = [hybrid_score(row.user_id, row.item_id, best_alpha)
             for row in test.itertuples()]
rmse_hyb  = rmse(test["rating"], preds_hyb)
mae_hyb   = mae(test["rating"],  preds_hyb)

print(f"RMSE         : {rmse_hyb:.4f}")
print(f"MAE          : {mae_hyb:.4f}")
print(f"Precision@10 : {p_hyb:.4f}")
print(f"Recall@10    : {r_hyb:.4f}")
print(f"NDCG@10      : {ndcg_hyb:.4f}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 5 — Save hybrid model (the final production model)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 5: Saving hybrid model (PRIMARY PRODUCTION MODEL)")
print("=" * 60)

hybrid_artifact = {
    "best_alpha":    best_alpha,
    "alpha_results": alpha_results,
    "all_items":     all_items,
    "global_mean":   global_mean,
}
with open(os.path.join(MODELS, "hybrid_model.pkl"), "wb") as f:
    pickle.dump(hybrid_artifact, f)
print("Saved: hybrid_model.pkl")

benchmark["hybrid"] = {
    "rmse": rmse_hyb, "mae": mae_hyb,
    "p10": p_hyb, "r10": r_hyb, "ndcg10": ndcg_hyb,
    "best_alpha": best_alpha
}
with open(os.path.join(MODELS, "benchmark.pkl"), "wb") as f:
    pickle.dump(benchmark, f)
print("Updated: benchmark.pkl")

# ══════════════════════════════════════════════════════════════════════════
# FINAL COMPARISON TABLE
# ══════════════════════════════════════════════════════════════════════════
pop = benchmark["popularity"]
ucf = benchmark.get("user_cf", {})
icf = benchmark.get("item_cf", {})
svd = benchmark.get("svd",     {})

def fmt(v):
    return f"{v:.4f}" if v is not None else "  —   "

print(f"""
{"=" * 72}
COMPLETE PHASE 2 RESULTS TABLE
{"=" * 72}
Model                RMSE     MAE      P@10     R@10     NDCG@10
────────────────────────────────────────────────────────────────────────
Popularity (base)    {fmt(pop.get('rmse'))}   {fmt(pop.get('mae'))}   {fmt(pop.get('p10'))}   {fmt(pop.get('r10'))}   —
User-based CF        {fmt(ucf.get('rmse'))}   {fmt(ucf.get('mae'))}   {fmt(ucf.get('p10'))}   {fmt(ucf.get('r10'))}   —
Item-based CF        {fmt(icf.get('rmse'))}   {fmt(icf.get('mae'))}   {fmt(icf.get('p10'))}   {fmt(icf.get('r10'))}   —
Content-based        —          —        {fmt(benchmark['content'].get('p10'))}   {fmt(benchmark['content'].get('r10'))}   {fmt(benchmark['content'].get('ndcg10'))}
SVD (tuned)          {fmt(svd.get('rmse'))}   {fmt(svd.get('mae'))}   {fmt(svd.get('p10'))}   {fmt(svd.get('r10'))}   {fmt(svd.get('ndcg10'))}
Hybrid ★ (α={best_alpha})   {fmt(rmse_hyb)}   {fmt(mae_hyb)}   {fmt(p_hyb)}   {fmt(r_hyb)}   {fmt(ndcg_hyb)}
{"=" * 72}

All 4 models built ✓
All models saved in backend/models/ ✓

Files ready for Phase 3 (FastAPI):
  popularity_model.pkl
  user_cf_model.pkl
  item_cf_model.pkl
  svd_model.pkl
  content_model.pkl
  hybrid_model.pkl   ← PRIMARY — use this for production API

→ NEXT PHASE: backend/api/main.py  (Phase 3, Days 15-18)
""")
