"""
Phase 1 — Day 4: Data Preprocessing & Train/Test Split
=======================================================
What this script does:
  1. Filters out low-activity users and items (cold start reduction)
  2. Does a per-user 80/20 train/test split (correct way!)
  3. Builds and saves the user-item matrix as a sparse scipy matrix
  4. Saves clean train/test CSVs for the model notebooks

Output:
  backend/data/train.csv
  backend/data/test.csv
  backend/data/user_item_matrix.npz   ← sparse matrix
  backend/data/user_encoder.csv       ← user_id ↔ matrix index mapping
  backend/data/item_encoder.csv       ← item_id ↔ matrix index mapping

WHY PER-USER SPLIT (not random split)?
  Random split: user 5's row 1 in train, row 2 in test → model "sees" the user
  Per-user split: for each user, 80% of their ratings go to train, 20% to test
  The second approach correctly simulates "can we predict what user hasn't rated yet"
"""

import pandas as pd
import numpy as np
import scipy.sparse as sp
import os

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")

# ══════════════════════════════════════════════════════════════════════════
# STEP 1 — Load raw data
# ══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1: Loading raw data")
print("=" * 60)

ratings = pd.read_csv(os.path.join(DATA, "ratings.csv"))
items   = pd.read_csv(os.path.join(DATA, "items.csv"))

print(f"Raw ratings: {len(ratings):,} rows")

# ══════════════════════════════════════════════════════════════════════════
# STEP 2 — Filter cold-start users and items
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2: Filtering low-activity users and items")
print("=" * 60)

# Users with fewer than 5 ratings → too few to learn from, too cold to predict
MIN_USER_RATINGS = 5
MIN_ITEM_RATINGS = 3

user_counts = ratings.groupby("user_id")["rating"].count()
active_users = user_counts[user_counts >= MIN_USER_RATINGS].index
print(f"Users before filter : {ratings['user_id'].nunique()}")
print(f"Users after  filter : {len(active_users)}  (removed {ratings['user_id'].nunique() - len(active_users)} cold users)")

ratings = ratings[ratings["user_id"].isin(active_users)]

item_counts = ratings.groupby("item_id")["rating"].count()
active_items = item_counts[item_counts >= MIN_ITEM_RATINGS].index
print(f"Items before filter : {ratings['item_id'].nunique()}")
print(f"Items after  filter : {len(active_items)}  (removed {ratings['item_id'].nunique() - len(active_items)} cold items)")

ratings = ratings[ratings["item_id"].isin(active_items)].reset_index(drop=True)
print(f"\nClean ratings: {len(ratings):,} rows")

# ══════════════════════════════════════════════════════════════════════════
# STEP 3 — Per-user 80/20 train/test split
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3: Per-user 80/20 train/test split")
print("=" * 60)

np.random.seed(42)  # reproducible splits

train_list = []
test_list  = []

for user_id, group in ratings.groupby("user_id"):
    group = group.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    split_idx = max(1, int(len(group) * 0.8))  # at least 1 rating in test
    train_list.append(group.iloc[:split_idx])
    test_list.append(group.iloc[split_idx:])

train = pd.concat(train_list).reset_index(drop=True)
test  = pd.concat(test_list).reset_index(drop=True)

print(f"Train set : {len(train):,} ratings ({len(train)/len(ratings)*100:.1f}%)")
print(f"Test  set : {len(test):,}  ratings ({len(test)/len(ratings)*100:.1f}%)")
print(f"Users in train : {train['user_id'].nunique()}")
print(f"Users in test  : {test['user_id'].nunique()}")

# Verify no data leakage: no (user, item) pair should appear in both sets
train_pairs = set(zip(train["user_id"], train["item_id"]))
test_pairs  = set(zip(test["user_id"],  test["item_id"]))
overlap = train_pairs & test_pairs
print(f"\nData leakage check - overlapping (user, item) pairs: {len(overlap)}  <- must be 0")
assert len(overlap) == 0, "DATA LEAKAGE DETECTED — fix the split!"

# ══════════════════════════════════════════════════════════════════════════
# STEP 4 — Build user/item encoders (integer index mapping)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 4: Building encoders for sparse matrix")
print("=" * 60)

# Models need 0-indexed integers, not arbitrary user/item IDs
unique_users = sorted(ratings["user_id"].unique())
unique_items = sorted(ratings["item_id"].unique())

user_to_idx = {u: i for i, u in enumerate(unique_users)}
item_to_idx = {it: i for i, it in enumerate(unique_items)}
idx_to_user = {i: u for u, i in user_to_idx.items()}
idx_to_item = {i: it for it, i in item_to_idx.items()}

print(f"Users encoded: 0 -> {len(unique_users)-1}")
print(f"Items encoded: 0 -> {len(unique_items)-1}")

# Save encoders as CSVs for the API to use later
pd.DataFrame({"user_id": unique_users, "user_idx": range(len(unique_users))}) \
  .to_csv(os.path.join(DATA, "user_encoder.csv"), index=False)

pd.DataFrame({"item_id": unique_items, "item_idx": range(len(unique_items))}) \
  .to_csv(os.path.join(DATA, "item_encoder.csv"), index=False)

print("Encoders saved: user_encoder.csv, item_encoder.csv")

# ══════════════════════════════════════════════════════════════════════════
# STEP 5 — Build sparse user-item matrix from TRAINING data only
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 5: Building sparse user-item matrix")
print("=" * 60)

# Map IDs to indices
train["user_idx"] = train["user_id"].map(user_to_idx)
train["item_idx"] = train["item_id"].map(item_to_idx)
test["user_idx"]  = test["user_id"].map(user_to_idx)
test["item_idx"]  = test["item_id"].map(item_to_idx)

n_users_clean = len(unique_users)
n_items_clean = len(unique_items)

# Sparse CSR matrix: rows = users, cols = items, values = ratings
user_item_matrix = sp.csr_matrix(
    (train["rating"].values,
     (train["user_idx"].values, train["item_idx"].values)),
    shape=(n_users_clean, n_items_clean)
)

sparsity = (1 - user_item_matrix.nnz / (n_users_clean * n_items_clean)) * 100
print(f"Matrix shape    : {user_item_matrix.shape}")
print(f"Non-zero values : {user_item_matrix.nnz:,}")
print(f"Sparsity        : {sparsity:.2f}%")
print(f"Memory (sparse) : {user_item_matrix.data.nbytes / 1024:.1f} KB")
print(f"Memory (dense)  : {n_users_clean * n_items_clean * 8 / 1024 / 1024:.1f} MB  <- why we use sparse!")

# Save sparse matrix
sp.save_npz(os.path.join(DATA, "user_item_matrix.npz"), user_item_matrix)
print("Saved: user_item_matrix.npz")

# ══════════════════════════════════════════════════════════════════════════
# STEP 6 — Save clean train/test splits
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 6: Saving train/test splits")
print("=" * 60)

train.to_csv(os.path.join(DATA, "train.csv"), index=False)
test.to_csv(os.path.join(DATA,  "test.csv"),  index=False)

print(f"Saved: train.csv ({len(train):,} rows)")
print(f"Saved: test.csv  ({len(test):,} rows)")

# ══════════════════════════════════════════════════════════════════════════
# STEP 7 — Preprocessing summary
# ══════════════════════════════════════════════════════════════════════════
print(f"""
{"=" * 60}
PREPROCESSING SUMMARY
{"=" * 60}
Raw ratings    : {len(ratings):,}
Train set      : {len(train):,} ratings  ({len(train)/len(ratings)*100:.0f}%)
Test  set      : {len(test):,}  ratings  ({len(test)/len(ratings)*100:.0f}%)
Users (clean)  : {n_users_clean}
Items (clean)  : {n_items_clean}
Matrix shape   : {n_users_clean} × {n_items_clean}
Sparsity       : {sparsity:.1f}%
Data leakage   : NONE ✓

Files saved to backend/data/:
  train.csv           ← use this for ALL model training
  test.csv            ← use this for ALL model evaluation  
  user_item_matrix.npz ← sparse matrix for CF models
  user_encoder.csv    ← user_id ↔ matrix index
  item_encoder.csv    ← item_id ↔ matrix index

NEXT STEP: Run 03_baseline_model.py
""")
