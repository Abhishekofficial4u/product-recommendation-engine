# Product Recommendation Engine: Project Report

This report outlines the problem statement, dataset exploratory findings, algorithmic architectures, empirical evaluation results, and architectural optimizations implemented for the Product Recommendation Engine project.

---

## 1. Problem Statement & Objectives
Digital streaming platforms and modern e-commerce catalogs host extensive libraries that vastly exceed an individual user's manual browsing capacity. Presenting uniform, non-personalized product lists leads to decision fatigue and reduced user retention. Mainstream recommendation paradigms include:
- **Collaborative Filtering**: Recommends items based on historical ratings from similar users or items. Struggles under sparse interaction matrices and unrated cold-start profiles.
- **Content-Based Filtering**: Matches item metadata (e.g., genres, categories) against historical user preferences. Ensures robust coverage but lacks discovery diversity.
- **Matrix Factorization (SVD)**: Decomposes sparse user-item matrices into dense lower-dimensional latent vectors.

This project delivers a **Hybrid Recommendation Engine** blending SVD matrix factorization with TF-IDF content-based cosine similarity to capture collaborative precision while safeguarding cold-start fallback coverage.

---

## 2. Dataset & Exploratory Data Analysis (EDA)
The system is trained and benchmarked on the **MovieLens-100K** dataset:
- **Matrix Dimensions**: 943 Users, 1,682 Items, 96,910 explicit ratings (filtered).
- **Matrix Sparsity**: **95.1%** (fewer than 1 in 20 user-item interaction cells are observed).
- **Activity Threshold**: Filtered for users with $\ge 5$ ratings and items with $\ge 3$ ratings.
- **Data Partitioning**: Implemented a per-user 80/20 train/test split (77,144 train / 19,766 test), preventing data leakage between user interaction sets.

---

## 3. Algorithmic Architecture
Six models were evaluated under identical experimental conditions:
1. **Popularity Baseline**: Bayesian-adjusted mean rating score preventing low-rating count bias.
2. **User-based Collaborative Filtering**: Scikit-Surprise `KNNWithMeans` using cosine similarity across top 40 user neighbors.
3. **Item-based Collaborative Filtering**: Scikit-Surprise `KNNWithMeans` using Pearson correlation across item similarities.
4. **SVD Matrix Factorization**: 100-latent factor SVD optimized via `GridSearchCV` ($n\_epochs = 30, lr = 0.005, reg = 0.02$).
5. **Content-Based Similarity**: TF-IDF vectorization across movie genres with pairwise cosine similarity matrices ($1,682 \times 1,682$).
6. **Hybrid Recommender**: Ensembles SVD predictions and TF-IDF content similarity scores using a weighted alpha formula:
   $$\text{Score} = \alpha \times \text{SVD Score} + (1 - \alpha) \times \text{Content Similarity}$$
   Optimized at $\alpha = 0.6$, achieving top ranking recall (NDCG@10 of **0.1057**).

---

## 4. Empirical Evaluation Results

All algorithms were evaluated on the identical 19,766 held-out test split:

| Model | RMSE (lower is better) | MAE (lower is better) | Precision@10 (higher is better) | Recall@10 (higher is better) | NDCG@10 (higher is better) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Popularity Baseline** | 1.0239 | 0.8128 | 0.0970 | 0.0630 | — |
| **User-based CF** | 0.9519 | 0.7500 | 0.0120 | 0.0163 | — |
| **Item-based CF** | 0.9430 | 0.7365 | 0.0160 | 0.0106 | — |
| **SVD (Tuned)** | 0.9249 | 0.7276 | 0.0832 | 0.0616 | 0.1010 |
| **Content-Based** | — | — | 0.0174 | 0.0141 | 0.0212 |
| **Hybrid ★** | 1.1720 | 0.9859 | 0.0808 | 0.0630 | **0.1057** |

---

## 5. System Latency Optimization
The hybrid recommendation routine evaluates up to 600 candidate items per request:
- **Initial Bottleneck**: Executing dynamic DataFrame filtering inside candidate evaluation loops incurred 600 redundant pandas operations, causing latencies of **1.42 seconds**.
- **Vectorized Optimization**: Pre-extracting user rating seeds prior to candidate iterations reduced request execution to **36ms (a 39.5x latency reduction)**.

---

## 6. Key Challenges & Architectural Mitigations
- **Cold Start Users**: Users with $< 3$ historical ratings bypass collaborative models, dynamically routing to content-based filtering with explicit API metadata flags.
- **Console Encoding Safety**: Replaced non-ASCII unicode logging characters with standard ASCII sequences to prevent Windows terminal encoding errors.
- **SPA Deployment Routing**: Configured catch-all rewrite rules in multi-stage Docker builds and `vercel.json` to prevent 404 client-side routing errors on browser refreshes.
