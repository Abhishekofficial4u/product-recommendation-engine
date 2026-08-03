"""
Phase 2 — Day 9: Content-Based Filtering
=========================================
Content-based = recommend items SIMILAR TO WHAT THE USER ALREADY LIKED.
No user-user or user-item interaction matrix needed.

How it works:
  1. Each item has text features (genre, description, category)
  2. TF-IDF converts that text into a numeric vector per item
  3. Cosine similarity measures how similar two item vectors are
  4. For user U: find their top-rated items → find items similar to those

Key advantage over CF:
  COLD START for items — a brand new item with zero ratings
  can still be recommended if its genre/description matches what the user likes.
  CF cannot do this (no ratings = no signal).

Key disadvantage:
  "Filter bubble" — only recommends more of the same genre.
  A user who only watches Action gets only Action recommendations.
  The hybrid model (Day 10) fixes this.

Output:
  backend/models/content_model.pkl   ← TF-IDF matrix + cosine sim + vectorizer
"""

import pandas as pd
import numpy as np
import pickle, os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel

# ── Paths ───────────────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA   = os.path.join(BASE, "data")
MODELS = os.path.join(BASE, "models")

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
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("Loading data")
print("=" * 60)

train = pd.read_csv(os.path.join(DATA, "train.csv"))
test  = pd.read_csv(os.path.join(DATA, "test.csv"))
items = pd.read_csv(os.path.join(DATA, "items.csv"))
with open(os.path.join(MODELS, "benchmark.pkl"), "rb") as f:
    benchmark = pickle.load(f)

print(f"Items: {len(items)}  |  Train: {len(train):,}  |  Test: {len(test):,}")
print(f"\nSample items:\n{items.head()}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 1 — Build item feature strings for TF-IDF
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 1: Building item feature strings")
print("=" * 60)
print("""
TF-IDF needs a text string per item.
We take the genre field (e.g. "Action|Adventure|Sci-Fi")
and convert it into a space-separated string ("Action Adventure SciFi").

In a real e-commerce project you would use:
  product_name + category + brand + description
The code is IDENTICAL — just different column names.
""")

# Replace pipe separator with space, remove special chars
items["genre_str"] = (
    items["genres"]
    .fillna("Unknown")
    .str.replace("|", " ", regex=False)
    .str.replace("-", "", regex=False)
    .str.strip()
)

print("Sample genre strings:")
for _, row in items.head(5).iterrows():
    print(f"  item_id={row['item_id']:5d}  genres='{row['genres']}'  -> '{row['genre_str']}'")

# ══════════════════════════════════════════════════════════════════════════
# STEP 2 — TF-IDF vectorisation
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2: TF-IDF vectorisation")
print("=" * 60)
print("""
TF-IDF = Term Frequency × Inverse Document Frequency
  TF  : how often does "Action" appear in this item's feature string?
  IDF : how rare is "Action" across ALL items?
        Common genres (Drama) get low weight.
        Rare genres (Film-Noir) get high weight → more distinctive signal.
""")

tfidf = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 1),   # single words
    min_df=2,             # ignore genres appearing in < 2 items
    max_features=500,     # cap vocabulary size
    sublinear_tf=True,    # log(1+tf) smoothing
)

tfidf_matrix = tfidf.fit_transform(items["genre_str"])

print(f"TF-IDF matrix shape : {tfidf_matrix.shape}  (items × genre terms)")
print(f"Vocabulary size     : {len(tfidf.vocabulary_)}")
print(f"Matrix density      : {tfidf_matrix.nnz / (tfidf_matrix.shape[0]*tfidf_matrix.shape[1]):.3%}")
print(f"Sample vocab        : {list(tfidf.vocabulary_.keys())[:10]}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 3 — Cosine similarity matrix
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3: Computing cosine similarity matrix")
print("=" * 60)
print("""
cosine_similarity(A, B) = (A · B) / (|A| × |B|)
Value range: 0 (totally different) → 1 (identical)

For 1,682 items: matrix is 1682 × 1682 = ~2.8M pairs.
We use linear_kernel (faster for TF-IDF sparse matrices).
""")

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
print(f"Cosine similarity matrix shape : {cosine_sim.shape}")
print(f"Memory                         : {cosine_sim.nbytes / 1024 / 1024:.1f} MB")

# Build item_id → matrix index mapping
item_ids    = items["item_id"].tolist()
item_to_idx = {iid: i for i, iid in enumerate(item_ids)}
idx_to_item = {i: iid for iid, i in item_to_idx.items()}

# Verify: items in the same genre should have high similarity
print("\nVerification — checking similarity for a few item pairs:")
for i in range(3):
    item_a = item_ids[i]
    genre_a = items.iloc[i]["genres"]
    # Find most similar
    sim_row = cosine_sim[i]
    top_idx = sim_row.argsort()[::-1][1:4]  # top 3 (skip self at idx 0)
    for j in top_idx:
        item_b  = item_ids[j]
        genre_b = items.iloc[j]["genres"]
        sim_val = sim_row[j]
        print(f"  item {item_a} ({genre_a}) <-> item {item_b} ({genre_b}) -> similarity={sim_val:.3f}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 4 — Content-based recommendation function
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 4: Content-based recommendation function")
print("=" * 60)

def content_recommend(user_id, train_df, top_k=10, n_seed=5):
    """
    1. Find user's top-rated items (seed items)
    2. Aggregate similarity scores for all unseen items
    3. Return top-K unseen items by aggregated similarity
    """
    user_ratings = train_df[train_df["user_id"] == user_id].copy()
    if user_ratings.empty:
        # Cold start: return globally most-similar items to a random seed
        return [item_ids[i] for i in range(top_k)]

    # Use top-N highest-rated items as seeds
    seeds = (
        user_ratings.sort_values("rating", ascending=False)
        .head(n_seed)["item_id"]
        .tolist()
    )
    seeds = [s for s in seeds if s in item_to_idx]

    seen      = set(user_ratings["item_id"].tolist())
    agg_score = np.zeros(len(item_ids))

    for seed_id in seeds:
        idx = item_to_idx[seed_id]
        agg_score += cosine_sim[idx]

    # Zero out already-seen items
    for s in seen:
        if s in item_to_idx:
            agg_score[item_to_idx[s]] = -1

    top_indices = agg_score.argsort()[::-1][:top_k]
    return [idx_to_item[i] for i in top_indices]

def content_score(user_id, item_id, train_df, n_seed=5):
    """Return a numeric similarity score for (user, item) — used in hybrid."""
    user_ratings = train_df[train_df["user_id"] == user_id]
    if user_ratings.empty or item_id not in item_to_idx:
        return 0.0
    seeds = (
        user_ratings.sort_values("rating", ascending=False)
        .head(n_seed)["item_id"]
        .tolist()
    )
    seeds = [s for s in seeds if s in item_to_idx]
    if not seeds: return 0.0
    item_idx = item_to_idx[item_id]
    return float(np.mean([cosine_sim[item_to_idx[s]][item_idx] for s in seeds]))

# Demo
sample_uid = train["user_id"].iloc[0]
cb_recs    = content_recommend(sample_uid, train, top_k=10)
print(f"Top-10 content-based recommendations for user {sample_uid}:")
recs_df = pd.DataFrame({"item_id": cb_recs}).merge(items[["item_id","genres"]], on="item_id")
print(recs_df.to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════
# STEP 5 — Evaluate ranking metrics
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 5: Evaluating content-based model")
print("=" * 60)

precisions, recalls, ndcgs = [], [], []
test_users = test["user_id"].unique()[:150]

for uid in test_users:
    user_test = test[test["user_id"] == uid]
    relevant  = set(user_test[user_test["rating"] >= 4]["item_id"])
    if not relevant: continue
    recs = content_recommend(uid, train, top_k=10)
    precisions.append(precision_at_k(recs, relevant))
    recalls.append(recall_at_k(recs, relevant))
    ndcgs.append(ndcg_at_k(recs, relevant))

p_cb   = np.mean(precisions)
r_cb   = np.mean(recalls)
ndcg_cb = np.mean(ndcgs)
print(f"Precision@10 : {p_cb:.4f}")
print(f"Recall@10    : {r_cb:.4f}")
print(f"NDCG@10      : {ndcg_cb:.4f}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 6 — Save content model
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 6: Saving content model")
print("=" * 60)

content_artifact = {
    "tfidf_matrix": tfidf_matrix,
    "cosine_sim":   cosine_sim,
    "vectorizer":   tfidf,
    "item_ids":     item_ids,
    "item_to_idx":  item_to_idx,
    "idx_to_item":  idx_to_item,
    "items_df":     items,
}
with open(os.path.join(MODELS, "content_model.pkl"), "wb") as f:
    pickle.dump(content_artifact, f)
print("Saved: content_model.pkl")

benchmark["content"] = {"p10": p_cb, "r10": r_cb, "ndcg10": ndcg_cb, "rmse": None}
with open(os.path.join(MODELS, "benchmark.pkl"), "wb") as f:
    pickle.dump(benchmark, f)
print("Updated: benchmark.pkl")

print(f"""
NOTE — Content-based has no RMSE because it doesn't predict a rating score.
It only ranks items by similarity. RMSE requires a predicted number.
In the hybrid model, we combine SVD's score with content similarity.

→ NEXT: Run 07_hybrid_model.py  (Day 10)
""")
