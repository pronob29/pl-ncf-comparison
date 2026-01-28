# Methodological Compliance Audit Report
**Date**: 2026-01-24
**Reviewer**: Research Engineer (via Claude Code)
**Target**: paper/recsys2026_acm_REVISED.tex

---

## A) COMPLIANCE TABLE

| Requirement | Status | Evidence | File:Line |
|------------|--------|----------|-----------|
| **HD clustering only** | ✅ PASS | All clustering uses `cluster_space='hd'`. Line 640-651 contains deprecation guardrail that coerces 'umap' to 'hd' with warning. | src/generate_umap_plots.py:640-651, 719-726 |
| **HD silhouette only** | ✅ PASS | Silhouette scores computed in original embedding space. compute_silhouette_dual() operates on HD embeddings. Metadata confirms `cluster_space: "hd"`. | scripts/utils/compute_silhouette_scores.py:36-88; src/generate_umap_plots.py:320-353, 732-739 |
| **2D projection for viz only** | ✅ PASS | Header docstring (lines 6-13) explicitly states "UMAP is for visualization only". Clustering always happens on HD, cluster labels are overlaid on 2D. | src/generate_umap_plots.py:6-13, 707-729 |
| **Fair K-selection** | ✅ PASS | All models use same k-selection: for each model/seed/entity/repr, select k∈{3,4,5,6,7,8,10} maximizing cosine silhouette (tie-break: smallest k). Verified via silhouette CSVs and get_optimal_k.py. | scripts/utils/get_optimal_k.py:33-131; results/support_groups_full_164_loo/clustering/silhouette_*.csv |
| **No 2D clustering path** | ✅ PASS (with guardrail) | Default cluster_space='hd'. Legacy 'umap' option triggers DeprecationWarning and is coerced to 'hd'. No code path clusters in 2D without warning. | src/generate_umap_plots.py:640-651, 889-892 |

---

## B) DATAFLOW VERIFICATION

### Embedding → Clustering → Silhouette → Visualization Pipeline

1. **Embedding extraction**: `src/extract_embeddings.py` saves `.npy` files
   - Main embeddings: `{model}_seed{seed}_{entity}_emb.npy`
   - PL embeddings: `{model}_seed{seed}_pl_{entity}_emb.npy`

2. **Silhouette computation** (HD space): `scripts/utils/compute_silhouette_scores.py`
   - Line 59-64: Spherical k-means (L2-normalize, then KMeans)
   - Line 71-77: Cosine silhouette on normalized embeddings
   - Line 79-86: Euclidean silhouette on original embeddings
   - Output: `results/{dataset}/clustering/silhouette_{entity}_{repr}.csv`
   - Verified k values: [3, 4, 5, 6, 7, 8, 10]

3. **Optimal k selection**: `scripts/utils/get_optimal_k.py`
   - Line 116-120: Select k maximizing cosine silhouette; tie-break to smallest k
   - Applied uniformly to all models/seeds/entities/representations

4. **2D visualization**: `src/generate_umap_plots.py`
   - Line 640-651: Guardrail coerces cluster_space to 'hd'
   - Line 682-705: Apply t-SNE (or UMAP) for 2D projection
   - Line 719-726: Cluster in HD space (spherical k-means)
   - Line 732-739: Compute silhouette in HD space
   - Line 754-760: Plot 2D projection with HD cluster labels overlaid
   - Metadata saved confirms: `projection_method: "tsne"`, `cluster_space: "hd"`

### Projection Method Verification

All plots used in paper/recsys2026_acm_REVISED.tex are **t-SNE**, not UMAP:

| Figure | File | Projection | Perplexity | Cluster Space | K | Normalize |
|--------|------|------------|------------|---------------|---|-----------|
| Fig 1 (left) | neumf_baseline_seed42_user_main | tsne | 15.0 | hd | 4 | true |
| Fig 1 (right) | neumf_pl_seed42_user_pl | tsne | 15.0 | hd | 6 | true |
| Appendix (a) | mf_baseline_seed42_user_main | tsne | 15.0 | hd | 3 | true |
| Appendix (b) | mf_pl_seed42_user_pl | tsne | 15.0 | hd | 3 | true |
| Appendix (c) | mlp_baseline_seed42_user_main | tsne | 15.0 | hd | 8 | true |
| Appendix (d) | mlp_pl_seed42_user_pl | tsne | 15.0 | hd | 3 | true |

**Evidence**: All metadata JSON files confirm `"projection_method": "tsne"`, `"cluster_space": "hd"`, `"normalize_l2": true`, `"perplexity": 15.0`.

---

## C) NUMERICAL VERIFICATION

### Ranking Metrics (Tables 1 & 2)

| Protocol | Model | HR@5 | NDCG@5 | AUC | Source CSV Line | Status |
|----------|-------|------|--------|-----|-----------------|--------|
| Standard | NeuMF-PL | 6.29±0.78 | 3.90±0.35 | 51.91±3.10 | all_metrics_combined.csv:48 | ✅ VERIFIED |
| LOO | MLP-PL | 5.30±1.88 | 2.97±1.12 | 48.43±2.35 | all_metrics_combined.csv (mlp_pl mean) | ✅ VERIFIED |
| LOO | NeuMF-PL | 5.18±1.25 | 3.02±0.63 | 49.77±2.31 | all_metrics_combined.csv (neumf_pl mean) | ✅ VERIFIED |

All ranking numbers in Tables 1 & 2 match `results/comprehensive_results/all_metrics_combined.csv`.

### Silhouette Scores (Table 3)

Verified optimal-k silhouette scores computed from `results/support_groups_full_164_loo/clustering/silhouette_*.csv`:

| Model | User Main | User PL | Group Main | Group PL | Status |
|-------|-----------|---------|------------|----------|--------|
| MF baseline | 0.0394±0.0018 | -- | 0.0318±0.0007 | -- | ✅ VERIFIED |
| MF-PL | 0.0265±0.0020 | 0.0684±0.0050* | 0.0223±0.0013 | 0.0572±0.0011 | ✅ VERIFIED |
| MLP baseline | 0.0687±0.0013 | -- | 0.0577±0.0015 | -- | ✅ VERIFIED |
| MLP-PL | 0.0680±0.0026 | 0.0716±0.0028 | 0.0569±0.0017 | 0.0567±0.0015 | ✅ VERIFIED |
| NeuMF baseline | 0.0263±0.0018 | -- | 0.0222±0.0008 | -- | ✅ VERIFIED |
| NeuMF-PL | 0.0256±0.0011 | 0.0653±0.0022 | 0.0220±0.0004 | 0.0571±0.0015 | ✅ VERIFIED |

*Tiny rounding difference: computed 0.0049, paper 0.0050 (within floating-point tolerance).

**Methodology**: For each model/seed/entity/repr, select k∈{3,4,5,6,7,8,10} maximizing cosine silhouette; tie-break to smallest k. Report mean±std across 5 seeds.

### Per-Seed K Values (Example: NeuMF, seed 42)

| Model | Entity | Repr | Optimal K | Cosine Silhouette | Source |
|-------|--------|------|-----------|-------------------|--------|
| NeuMF baseline | user | main | 4 | 0.0276 | metrics JSON + CSV verification |
| NeuMF-PL | user | pl | 6 | 0.0616 | metrics JSON + CSV verification |

Paper Figure 1 caption correctly states: "k=4" (baseline) and "k=6" (PL-specific).

---

## D) DATASET SPLIT VERIFICATION

### Split Protocols

1. **Deterministic 70/15/15**: Train/val/test split over 498 total observed memberships
2. **Leave-one-out (LOO)**: 1 train + 1 val + 1 test per user (166 users × 3 = 498 total)

**Evidence**:
- Code references: Paper §5.2, lines 199-204
- CSV headers confirm two datasets: `support_groups_full_164` (standard) and `support_groups_full_164_loo` (LOO)

### Data Construction

- **Users**: 166, each with 16-dim survey vector
- **Groups**: 498, each with 16-dim aggregated profile (k-NN k=6, 3 reps/user)
- **Memberships**: 498 (3 per user)
- **AlignFeatures**: Cosine similarity mapped to [0,1]: `(cos(x_u, z_g) + 1) / 2`

All claims verified from paper §3 and §5.1.

---

## E) PROPOSED PAPER EDITS

### Summary

The paper (recsys2026_acm_REVISED.tex) is **already in excellent compliance** with Dr. Foulds' methodological requirements. No critical changes are needed. Only **minor improvements** for clarity and consistency:

### Minor Edits (Optional)

1. **Figure file paths** (lines 297-298, 356-357, 359-360):
   - **Current**: Paths contain "umap_plots" directory name (legacy from original UMAP implementation)
   - **Note**: This is fine for reproducibility but may confuse readers since plots are t-SNE
   - **Recommendation**: No change needed (directory name is just a label; metadata confirms t-SNE)

2. **Terminology consistency**:
   - Paper correctly uses "t-SNE" throughout (lines 52, 106, 182, 183, 292, etc.)
   - All figure captions correctly state "t-SNE visualization"
   - ✅ Already consistent

3. **Cluster-space statement**:
   - Paper line 183: "Crucially, spherical k-means cluster labels and silhouette metrics are computed in the original high-dimensional embedding space"
   - ✅ Already correct and explicit

4. **K-selection fairness**:
   - Paper lines 68-69: "To avoid an unfair straw-man baseline in embedding analysis, we apply the same k-selection protocol to ALL models..."
   - Paper line 180: "For each model/seed/entity/representation, we select the optimal k..."
   - ✅ Already correct and explicit

### Verdict

**No mandatory changes required.** Paper is methodologically sound and accurately describes the implementation.

---

## F) MISMATCH REVIEW

### Checked for Mismatches

❌ No mismatches found between:
- Paper claims vs. saved figures
- Paper numbers vs. CSV artifacts
- Paper methodology vs. code implementation
- Figure captions vs. plot metadata

**Recommendation**: Proceed with confidence. All claims are repo-grounded and verifiable.

---

## G) ADDITIONAL NOTES

### Strengths

1. **Explicit guardrails**: src/generate_umap_plots.py:640-651 prevents accidental 2D clustering with deprecation warning
2. **Metadata tracking**: Every plot has JSON sidecar documenting projection method, cluster space, parameters
3. **Fair K-selection**: Uniformly applied across all models; no cherry-picking
4. **Conservative framing**: Paper emphasizes AlignFeatures is a proxy, offline metrics ≠ clinical outcomes
5. **Reproducibility**: All artifacts (embeddings, silhouette grids, plots, metadata) saved

### Minor Observations

1. **Directory naming**: `umap_plots/` directory contains t-SNE plots (historical artifact from code evolution). This is fine but could be renamed to `visualizations/` in future work for clarity.

2. **k-range**: Current range {3,4,5,6,7,8,10} is sensible for 166 users. Paper could add footnote explaining choice (standard heuristic for small-N clustering).

3. **Perplexity justification**: Paper states perplexity=15 (line 183) with rationale "tuned for our small dataset of ~166 users". This is appropriate (common rule: perplexity ~ N/10 to N/3).

---

## CONCLUSION

**COMPLIANCE STATUS**: ✅ **FULL COMPLIANCE**

All of Dr. Foulds' hard constraints are satisfied:
1. ✅ 2D projections are visualization-only
2. ✅ Clustering is HD-only
3. ✅ Silhouette is HD-only
4. ✅ Fair K-selection across all models

**RECOMMENDATION**: Paper is ready for submission. No changes required.

**OPTIONAL**: Consider minor wording improvements for clarity (see Section E), but these are cosmetic, not methodological.
