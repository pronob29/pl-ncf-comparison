# UMAP Alignment Audit Report

**Date**: 2026-01-21
**Purpose**: Document fixes addressing Dr. Foulds' methodological concerns

---

## Dr. Foulds' Concerns (Summary)

1. **Primary**: Clustering and evaluating structure in UMAP space instead of original embeddings
2. **Secondary**: K was not being selected fairly across baselines
3. **Broader**: "Weak UMAP visuals and purely geometric separation were being overinterpreted without clear semantic grounding"

---

## Fixes Implemented

### 1. HD Clustering Only (Primary Concern - FIXED)

**Problem**: Code allowed clustering on 2D UMAP coordinates (`--cluster-space umap`)

**Solution**:
- Added guardrail in `src/generate_umap_plots.py:371-382` that coerces `cluster_space='umap'` to `'hd'` with deprecation warning
- Cluster labels are ALWAYS computed on original HD embeddings via spherical K-means
- Silhouette scores are ALWAYS computed on HD embeddings
- UMAP is used for visualization only

**Verification**:
```bash
# All metrics.json files show cluster_space: "hd"
grep -r '"cluster_space": "hd"' results/*/umap_plots/*_metrics.json | wc -l
```

### 2. Fair K Selection Across Baselines (Secondary Concern - FIXED)

**Problem**: Baselines used fixed k=5 while PL models used optimal k from silhouette

**Solution**:
- ALL models (baselines AND PL) now use silhouette-selected optimal k
- K selection: maximize cosine silhouette from {3, 4, 5, 6, 7, 8, 10}
- Tie-break: smallest k wins
- Implemented in `scripts/3_generate_umap_plots_all_optimal.sh`

**Verification**:
```bash
# Check that optimal k varies across models
grep "optimal k=" logs/analysis/umap_plots_all_optimal_*.out
```

### 3. Correct Interpretation of UMAP (Broader Concern - ADDRESSED)

**Problem**: Risk of overinterpreting UMAP visual patterns

**Solution**:
- Documentation updated to emphasize UMAP is **qualitative visualization only**
- Removed artificial "enhancement" settings (excessive repulsion, spread, etc.)
- Removed visual tuning sweep tool that encouraged wrong approach
- Standard UMAP parameters used: `--normalize --n-neighbors 15 --min-dist 0.1`

**Key Message**: Visual overlap in 2D is EXPECTED and does NOT indicate poor clustering. Quantitative evaluation uses HD silhouette scores, not UMAP appearance.

---

## Current Pipeline Settings

### Shell Script (`scripts/3_generate_umap_plots_all_optimal.sh`)

```bash
PRESENTATION_FLAGS="--normalize --n-neighbors 15 --min-dist 0.1 --cluster-space hd"
```

### What This Means

| Aspect | Setting | Rationale |
|--------|---------|-----------|
| Clustering | HD only | Methodologically correct |
| K selection | Silhouette-optimal | Fair across all models |
| UMAP params | Standard | No artificial enhancement |
| Visual separation | May be weak | This is EXPECTED |

---

## Correct Interpretation Guide

### UMAP Plots Show:
- 2D projection of HD embeddings (for visualization only)
- Colors represent clusters computed on HD embeddings
- Visual overlap is expected due to UMAP's non-linear distortion

### UMAP Plots Do NOT Show:
- Quantitative cluster quality (use HD silhouette for this)
- Actual distances between points (UMAP distorts distances)
- Definitive cluster boundaries (these are computed in HD)

### Quantitative Evaluation:
- HD silhouette scores (cosine and euclidean) in `results/*/clustering/silhouette_*.csv`
- These are computed on original embeddings, NOT on UMAP

---

## Files Modified

| File | Change |
|------|--------|
| `src/generate_umap_plots.py` | Guardrail coercing umap→hd, simplified plot (no X markers) |
| `scripts/3_generate_umap_plots_all_optimal.sh` | Standard UMAP params, all models use optimal k |
| `UMAP_GENERATION_GUIDE.md` | Correct interpretation guidance |
| `paper/recsys2026_acm_FIXED.tex` | Updated to reflect HD clustering |

---

## Summary

Dr. Foulds' concerns have been addressed:

1. ✅ Clustering is performed on HD embeddings only
2. ✅ K selection is fair across baselines and PL models
3. ✅ Documentation warns against overinterpreting UMAP visuals
4. ✅ No artificial enhancement of visual separation

**The weak visual separation in UMAP is a feature, not a bug.** It correctly reflects that UMAP distorts structure, which is why all quantitative evaluation uses HD embeddings.

---

*Generated 2026-01-21*
