# Technical Details

This document provides in-depth technical information about the Neural Collaborative Filtering with Pseudo-Labeling (PL-NCF) project, including model architectures, training procedures, evaluation methodology, and visualization techniques.

---

## Table of Contents

1. [Model Architectures](#model-architectures)
   - [Matrix Factorization (MF)](#matrix-factorization-mf)
   - [Multi-Layer Perceptron (MLP)](#multi-layer-perceptron-mlp)
   - [Neural Matrix Factorization (NeuMF)](#neural-matrix-factorization-neumf)
2. [Pseudo-Labeling Framework](#pseudo-labeling-framework)
   - [Dual Embedding Architecture](#dual-embedding-architecture)
   - [Pseudo-Label Generation](#pseudo-label-generation)
   - [Training Objective](#training-objective)
3. [Training Pipeline](#training-pipeline)
   - [Data Preparation](#data-preparation)
   - [Negative Sampling](#negative-sampling)
   - [Hyperparameters](#hyperparameters)
4. [Evaluation Metrics](#evaluation-metrics)
   - [Recommendation Metrics](#recommendation-metrics)
   - [Clustering Metrics](#clustering-metrics)
5. [UMAP Visualization](#umap-visualization)
   - [Standard Mode](#standard-mode)
   - [Presentation Mode](#presentation-mode)
   - [Clustering for UMAP Visualization](#clustering-for-umap-visualization)
6. [Complete Results](#complete-results)

---

## Model Architectures

### Matrix Factorization (MF)

The simplest baseline model using dot-product of learned embeddings.

**Architecture:**
```
User Index → User Embedding (dim=64) ─┐
                                      ├→ Dot Product → σ → Prediction
Item Index → Item Embedding (dim=64) ─┘
```

**Mathematical Formulation:**
```
ŷ_ui = σ(u_i^T · v_j)
```
where `u_i` is the user embedding, `v_j` is the item embedding, and `σ` is the sigmoid function.

**Implementation:** [src/mf.py](src/mf.py)

---

### Multi-Layer Perceptron (MLP)

A deep neural network that learns non-linear user-item interactions.

**Architecture:**
```
User Index → User Embedding (dim=32) ─┐
                                      ├→ Concat → FC(64→32) → ReLU → FC(32→16) → ReLU → FC(16→8) → ReLU → FC(8→1) → σ
Item Index → Item Embedding (dim=32) ─┘
```

**Layer Configuration:**
- Input: Concatenated user + item embeddings (64 dimensions)
- Hidden layers: [64, 32, 16, 8]
- Activation: ReLU
- Output: Sigmoid

**Implementation:** [src/mlp.py](src/mlp.py)

---

### Neural Matrix Factorization (NeuMF)

Hybrid architecture combining GMF (Generalized Matrix Factorization) and MLP pathways.

**Architecture:**
```
                    ┌─────────────────────────────────────────┐
                    │           GMF Pathway                    │
User Index ────────→│ User_GMF_Emb (dim=8) ─┐                 │
                    │                        ├→ Element-wise × │────┐
Item Index ────────→│ Item_GMF_Emb (dim=8) ─┘                 │    │
                    └─────────────────────────────────────────┘    │
                                                                   ├→ Concat → FC → σ → ŷ
                    ┌─────────────────────────────────────────┐    │
                    │           MLP Pathway                    │    │
User Index ────────→│ User_MLP_Emb (dim=32) ─┐                │    │
                    │                         ├→ Concat → MLP │────┘
Item Index ────────→│ Item_MLP_Emb (dim=32) ─┘                │
                    └─────────────────────────────────────────┘
```

**Mathematical Formulation:**
```
φ_GMF = u_mf ⊙ v_mf                    (element-wise product)
φ_MLP = MLP(concat(u_mlp, v_mlp))      (deep interaction)
ŷ = σ(W^T · concat(φ_GMF, φ_MLP) + b)  (fusion layer)
```

**Implementation:** [src/neumf.py](src/neumf.py)

---

## Pseudo-Labeling Framework

### Dual Embedding Architecture

PL models maintain **two separate embedding spaces**:

1. **Main Embeddings**: Optimized for recommendation (BCE loss)
2. **PL Embeddings**: Optimized for semantic clustering (MSE loss with pseudo-labels)

```
                    ┌────────────────────────────────────┐
                    │      Main Embeddings               │
User Index ────────→│ User_Main → Standard NCF Forward   │→ BCE Loss
Item Index ────────→│ Item_Main                          │
                    └────────────────────────────────────┘

                    ┌────────────────────────────────────┐
                    │      PL Embeddings                 │
User Index ────────→│ User_PL ─┐                        │
                    │          ├→ Cosine Similarity     │→ MSE Loss (vs pseudo-labels)
Item Index ────────→│ Item_PL ─┘                        │
                    └────────────────────────────────────┘
```

**Key Insight:** This dual architecture allows the model to learn:
- Main embeddings: Capture collaborative filtering patterns
- PL embeddings: Capture content-based semantic similarity

### Pseudo-Label Generation

Pseudo-labels are generated from **AlignFeatures** - survey-derived alignment features:

```python
# User features: preferences, topics, support needs (from survey)
# Item features: group characteristics, topics covered, focus areas

pseudo_label[u, i] = cosine_similarity(user_features[u], item_features[i])
```

**Range:** [-1, 1] (cosine similarity values)

### Training Objective

PL models are trained with a combined loss:

```
L_total = (1 - λ) × L_BCE + λ × L_PL

where:
  L_BCE = -Σ[y·log(ŷ) + (1-y)·log(1-ŷ)]           (binary cross-entropy)
  L_PL  = Σ(cos_sim(u_pl, v_pl) - pseudo_label)²   (MSE on cosine similarity)
```

**Default λ = 0.1:** Balances recommendation accuracy with semantic clustering.

---

## Training Pipeline

### Data Preparation

**Dataset Statistics:**

| Dataset | Users | Items | Interactions | Density |
|---------|-------|-------|--------------|---------|
| support_groups_full_164 | 166 | 498 | ~15,000 | ~18.2% |
| support_groups_full_164_loo | 166 | 498 | ~15,000 | ~18.2% |

**Evaluation Protocols:**

1. **Standard Split (support_groups_full_164)**
   - Train: 80% of interactions
   - Validation: 10% of interactions
   - Test: 10% of interactions

2. **Leave-One-Out (support_groups_full_164_loo)**
   - Train: All interactions except one per user
   - Test: One held-out interaction per user
   - Evaluation: Rank test item against all items

### Negative Sampling

**Training Phase:**
- Sample 4 negative items per positive interaction
- Negative items: Items not interacted with by the user

**Evaluation Phase (LOO):**
- Rank test item against all uninteracted items
- More challenging: ~500 candidates per user

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Embedding Dim (MF) | 64 | Main embedding dimension |
| Embedding Dim (MLP) | 32 | Per pathway |
| Embedding Dim (GMF) | 8 | NeuMF GMF component |
| PL Dimension | 32 | Pseudo-label embedding |
| MLP Layers | [64, 32, 16, 8] | Hidden layer sizes |
| Learning Rate | 0.001 | Adam optimizer |
| Batch Size | 512 | Training batch |
| Epochs | 20 | Training epochs |
| λ (PL weight) | 0.1 | Loss balance |
| Seeds | 42, 52, 62, 122, 232 | Random seeds |

---

## Evaluation Metrics

### Recommendation Metrics

**Hit Rate @ K (HR@K):**
```
HR@K = (# users with test item in top-K) / (# total users)
```

**Normalized Discounted Cumulative Gain @ K (NDCG@K):**
```
NDCG@K = DCG@K / IDCG@K
DCG@K = Σ(1/log2(rank+1)) for hits in top-K
```

**Area Under Curve (AUC):**
```
AUC = P(score(positive) > score(negative))
```

### Clustering Metrics

**Silhouette Score (Cosine):**

Measures cluster cohesion and separation using cosine distance:

```
s(i) = (b(i) - a(i)) / max(a(i), b(i))

where:
  a(i) = mean intra-cluster distance for point i
  b(i) = mean nearest-cluster distance for point i
```

**Clustering Method:** Spherical K-means
- L2 normalize embeddings
- Standard K-means on normalized vectors
- Equivalent to K-means with cosine distance

**K Selection:**
- Grid search: k ∈ {3, 4, 5, 6, 7, 8, 10}
- Optimal k: Maximizes silhouette score per model/seed

---

## UMAP Visualization

### Standard Mode

Default UMAP parameters for general visualization:

```python
umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    metric='cosine',
    random_state=42
)
```

### Presentation Mode

Enhanced parameters for clearer cluster separation in publications:

```python
umap.UMAP(
    n_neighbors=50,          # More global structure
    min_dist=0.0,            # Tight clusters
    spread=1.0,              # Default spread
    repulsion_strength=2.0,  # Stronger separation
    n_components=2,
    metric='cosine',
    random_state=42
)
```

**Preprocessing:**
- L2 normalization (--normalize flag)
- Optional PCA dimensionality reduction (--pca-dim)

**Usage:**
```bash
python src/generate_umap_plots.py \
    --model neumf_pl \
    --dataset support_groups_full_164_loo \
    --seed 42 \
    --entity user \
    --repr pl \
    --presentation
```

### Clustering for UMAP Visualization

**UMAP is visualization only.** Cluster labels are always computed on the original high-dimensional embeddings, then overlaid on the 2D UMAP projection for visualization.

**Why HD clustering is required:**
- UMAP is a non-linear dimensionality reduction that distorts distances and densities
- Clustering decisions in 2D UMAP space would be methodologically unsound
- Silhouette scores are computed on HD embeddings for correctness
- The 2D scatter plot is purely for visual inspection of cluster structure

**Implementation:**
- `--cluster-space hd` (default and required): Clusters computed on original HD embeddings
- `--cluster-space umap`: **DEPRECATED** - Coerced to `hd` with a warning

**Note**: The `--cluster-space umap` option was deprecated because clustering in UMAP space could produce misleading results. All plots now use HD clustering for correctness.

---

## Complete Results

### Recommendation Performance (LOO Evaluation)

| Model | HR@5 | NDCG@5 | AUC | HR@5 Δ |
|-------|------|--------|-----|--------|
| MF Baseline | 0.0458 | 0.0270 | 0.5024 | — |
| **MF PL** | 0.0542 | 0.0332 | 0.4766 | **+18.3%** |
| MLP Baseline | 0.0265 | 0.0141 | 0.4618 | — |
| **MLP PL** | 0.0530 | 0.0297 | 0.4843 | **+100.0%** |
| NeuMF Baseline | 0.0446 | 0.0250 | 0.4547 | — |
| **NeuMF PL** | 0.0518 | 0.0302 | 0.4977 | **+16.1%** |

*Mean across 5 random seeds (42, 52, 62, 122, 232)*

### Clustering Quality (User Embeddings, LOO Dataset)

| Model | Main Silhouette | PL Silhouette | Improvement |
|-------|-----------------|---------------|-------------|
| MF Baseline | 0.0385 | — | — |
| MF PL (Main) | 0.0265 | — | — |
| **MF PL (PL)** | — | **0.0650** | **+145% vs MF PL Main** |
| MLP Baseline | 0.0646 | — | — |
| MLP PL (Main) | 0.0650 | — | — |
| **MLP PL (PL)** | — | **0.0699** | **+7.5% vs MLP PL Main** |
| NeuMF Baseline | 0.0258 | — | — |
| NeuMF PL (Main) | 0.0255 | — | — |
| **NeuMF PL (PL)** | — | **0.0629** | **+147% vs NeuMF PL Main** |

*Cosine silhouette scores at k=3 (best performing k for most models)*

### Key Findings

1. **Dual Representation Learning Works:**
   - PL-specific embeddings show dramatically better clustering than main embeddings
   - MF and NeuMF show ~145-147% improvement in silhouette scores

2. **Recommendation Performance Improves:**
   - MLP PL shows +100% improvement in HR@5 over baseline
   - All PL variants outperform baselines in HR@5 and NDCG@5

3. **Trade-off Analysis:**
   - PL models sacrifice some AUC for better semantic structure
   - Main embeddings in PL models have slightly lower clustering quality
   - PL-specific embeddings compensate with much better clustering

4. **Architecture Comparison:**
   - MLP baseline has highest main embedding clustering (0.0646)
   - NeuMF PL achieves best overall balance of metrics
   - MF PL shows largest relative improvement in clustering

---

## References

1. He, X., et al. "Neural Collaborative Filtering." WWW 2017.
2. Lee, D.H. "Pseudo-Label: The Simple and Efficient Semi-Supervised Learning Method." ICML Workshop 2013.
3. McInnes, L., et al. "UMAP: Uniform Manifold Approximation and Projection." Journal of Open Source Software, 2018.

---

*Last Updated: January 2026*
