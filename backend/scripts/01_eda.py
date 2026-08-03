"""
Phase 1 — Day 2 & 3: Data Loading & Exploratory Data Analysis
=============================================================
Run this file to:
  1. Load ratings and items CSVs
  2. Print key statistics
  3. Generate and save all EDA plots
  4. Understand the data before building any model

Output: backend/data/eda_plots/ folder with 5 PNG charts
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os

# ── Paths ──────────────────────────────────────────────────────────────────
BASE    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA    = os.path.join(BASE, "data")
PLOTS   = os.path.join(DATA, "eda_plots")
os.makedirs(PLOTS, exist_ok=True)

# ── Plot style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "font.size":        12,
})
PALETTE = ["#3B82F6","#8B5CF6","#10B981","#F59E0B","#EF4444"]

# ══════════════════════════════════════════════════════════════════════════
# STEP 1 — Load data
# ══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1: Loading data")
print("=" * 60)

ratings = pd.read_csv(os.path.join(DATA, "ratings.csv"))
items   = pd.read_csv(os.path.join(DATA, "items.csv"))

print(f"\nratings.csv shape : {ratings.shape}")
print(f"items.csv   shape : {items.shape}")
print(f"\nratings.dtypes:\n{ratings.dtypes}")
print(f"\nFirst 5 rows of ratings:\n{ratings.head()}")
print(f"\nFirst 5 rows of items:\n{items.head()}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 2 — Basic statistics
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2: Basic statistics")
print("=" * 60)

n_users   = ratings["user_id"].nunique()
n_items   = ratings["item_id"].nunique()
n_ratings = len(ratings)

print(f"\nTotal ratings : {n_ratings:,}")
print(f"Unique users  : {n_users:,}")
print(f"Unique items  : {n_items:,}")

# Sparsity: how much of the user-item matrix is empty?
possible   = n_users * n_items
sparsity   = (1 - n_ratings / possible) * 100
density    = 100 - sparsity
print(f"\nPossible user-item pairs : {possible:,}")
print(f"Filled pairs             : {n_ratings:,}")
print(f"Sparsity                 : {sparsity:.2f}%  <- this is why plain CF struggles")
print(f"Density                  : {density:.2f}%")

# Rating stats
print(f"\nRating statistics:")
print(ratings["rating"].describe())
print(f"\nRating value counts:")
print(ratings["rating"].value_counts().sort_index())

avg_ratings_per_user = ratings.groupby("user_id")["rating"].count().mean()
avg_ratings_per_item = ratings.groupby("item_id")["rating"].count().mean()
print(f"\nAvg ratings per user : {avg_ratings_per_user:.1f}")
print(f"Avg ratings per item : {avg_ratings_per_item:.1f}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 3 — Check missing values & duplicates
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3: Data quality checks")
print("=" * 60)

print(f"\nMissing values in ratings:\n{ratings.isnull().sum()}")
print(f"\nMissing values in items:\n{items.isnull().sum()}")

dupes = ratings.duplicated(subset=["user_id","item_id"]).sum()
print(f"\nDuplicate (user, item) pairs: {dupes}  <- should be 0")

# ══════════════════════════════════════════════════════════════════════════
# STEP 4 — EDA plots
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 4: Generating EDA plots")
print("=" * 60)

# ── Plot 1: Rating distribution ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
counts = ratings["rating"].value_counts().sort_index()
bars = ax.bar(counts.index, counts.values, color=PALETTE[0], edgecolor="white", linewidth=0.5, width=0.6)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
            f"{val:,}", ha="center", va="bottom", fontsize=11, color="#374151")
ax.set_xlabel("Star rating", fontsize=13)
ax.set_ylabel("Number of ratings", fontsize=13)
ax.set_title("Rating distribution\n(most users rate 3–4 stars)", fontsize=14, pad=15)
ax.set_xticks([1,2,3,4,5])
plt.tight_layout()
p = os.path.join(PLOTS, "01_rating_distribution.png")
plt.savefig(p, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {p}")

# ── Plot 2: Ratings per user (long-tail) ─────────────────────────────────
user_counts = ratings.groupby("user_id")["rating"].count().sort_values(ascending=False)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(range(len(user_counts)), user_counts.values, color=PALETTE[1], linewidth=1.5)
axes[0].fill_between(range(len(user_counts)), user_counts.values, alpha=0.15, color=PALETTE[1])
axes[0].set_xlabel("Users (sorted by activity)", fontsize=12)
axes[0].set_ylabel("Number of ratings", fontsize=12)
axes[0].set_title("Ratings per user — the long tail\n(most users rate very few items)", fontsize=13)

axes[1].hist(user_counts.values, bins=40, color=PALETTE[1], edgecolor="white", linewidth=0.5)
axes[1].set_xlabel("Number of ratings", fontsize=12)
axes[1].set_ylabel("Number of users", fontsize=12)
axes[1].set_title("Distribution of user activity", fontsize=13)
plt.suptitle(f"Active users: {(user_counts >= 20).sum()} | Inactive (<5 ratings): {(user_counts < 5).sum()}",
             fontsize=11, color="#6B7280", y=1.01)
plt.tight_layout()
p = os.path.join(PLOTS, "02_ratings_per_user.png")
plt.savefig(p, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {p}")

# ── Plot 3: Ratings per item (popularity) ─────────────────────────────────
item_counts = ratings.groupby("item_id")["rating"].count().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(range(len(item_counts)), item_counts.values, color=PALETTE[2], linewidth=1.5)
ax.fill_between(range(len(item_counts)), item_counts.values, alpha=0.15, color=PALETTE[2])
ax.axvline(x=100, color=PALETTE[4], linestyle="--", linewidth=1, label="Top 100 items")
ax.set_xlabel("Items (sorted by popularity)", fontsize=12)
ax.set_ylabel("Times rated", fontsize=12)
ax.set_title("Item popularity — power law distribution\n(few items dominate, most are rarely rated)", fontsize=13)
ax.legend(fontsize=11)
p80 = item_counts.iloc[int(0.8*len(item_counts))]
ax.annotate(f"80th pct: {p80:.0f} ratings",
            xy=(int(0.8*len(item_counts)), p80),
            xytext=(int(0.8*len(item_counts))+80, p80+20),
            arrowprops=dict(arrowstyle="->", color="#374151"), fontsize=10)
plt.tight_layout()
p = os.path.join(PLOTS, "03_ratings_per_item.png")
plt.savefig(p, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {p}")

# ── Plot 4: Top 20 most-rated items ───────────────────────────────────────
top20 = item_counts.head(20).reset_index()
top20.columns = ["item_id","count"]
top20 = top20.merge(items[["item_id","title"]], on="item_id")

fig, ax = plt.subplots(figsize=(10, 7))
colors = [PALETTE[0] if i < 5 else "#93C5FD" for i in range(20)]
bars = ax.barh(top20["title"], top20["count"], color=colors, edgecolor="white")
ax.invert_yaxis()
ax.set_xlabel("Number of ratings", fontsize=12)
ax.set_title("Top 20 most-rated items\n(top 5 highlighted — your popularity baseline)", fontsize=13)
for bar, val in zip(bars, top20["count"]):
    ax.text(bar.get_width() + 3, bar.get_y() + bar.get_height()/2,
            f"{val:,}", va="center", fontsize=9)
plt.tight_layout()
p = os.path.join(PLOTS, "04_top20_items.png")
plt.savefig(p, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {p}")

# ── Plot 5: Genre distribution ─────────────────────────────────────────────
all_genres = []
for g in items["genres"].dropna():
    all_genres.extend(g.split("|"))
genre_series = pd.Series(all_genres).value_counts()

fig, ax = plt.subplots(figsize=(10, 6))
colors_g = plt.cm.Blues(np.linspace(0.4, 0.9, len(genre_series)))[::-1]
bars = ax.barh(genre_series.index, genre_series.values,
               color=colors_g, edgecolor="white", linewidth=0.5)
ax.invert_yaxis()
ax.set_xlabel("Number of items", fontsize=12)
ax.set_title("Items per genre\n(used as content features for TF-IDF)", fontsize=13)
for bar, val in zip(bars, genre_series.values):
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
            str(val), va="center", fontsize=10)
plt.tight_layout()
p = os.path.join(PLOTS, "05_genre_distribution.png")
plt.savefig(p, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {p}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 5 — Summary for team
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 5: EDA SUMMARY — share with team")
print("=" * 60)
print(f"""
KEY FINDINGS FROM EDA
─────────────────────
Dataset size   : {n_ratings:,} ratings | {n_users} users | {n_items} items
Sparsity       : {sparsity:.1f}% — matrix is almost empty → CF alone will struggle
Rating bias    : Most ratings cluster at 3–4 stars → mild positive bias
Cold users     : {(user_counts < 5).sum()} users have < 5 ratings → need fallback strategy
Popular items  : Top 100 items get most ratings → strong popularity bias

IMPLICATIONS FOR MODELS
────────────────────────
1. High sparsity ({sparsity:.0f}%) → SVD handles this better than user-based CF
2. Cold users → content-based model as fallback
3. Popularity bias → always include as a simple baseline to beat
4. Genre info available → TF-IDF content features ready

NEXT STEP: Run 02_preprocessing.py
""")
