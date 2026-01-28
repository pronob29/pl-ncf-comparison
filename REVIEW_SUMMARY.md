# RecSys 2026 Paper Review Summary

## Quick Status: ✅ READY FOR SUBMISSION

Your paper **passes all methodological compliance checks** and numerical verification. No critical changes required.

---

## Compliance Scorecard

| Criterion | Status | Notes |
|-----------|--------|-------|
| HD clustering only | ✅ PASS | Code enforces with deprecation guardrail |
| HD silhouette only | ✅ PASS | All metrics computed in original space |
| 2D for visualization only | ✅ PASS | Explicit in code comments + paper text |
| Fair K-selection | ✅ PASS | Same grid-search for all models |
| No script/path leaks | ✅ PASS | Paper contains only figure paths (appropriate) |

---

## Numerical Verification

### All Tables Match Saved Artifacts

✅ **Table 1** (70/15/15 split): Verified against `results/comprehensive_results/all_metrics_combined.csv`
- Example: NeuMF-PL HR@5 = 6.29±0.78 ✓

✅ **Table 2** (LOO split): Verified against same CSV
- MLP-PL HR@5 = 5.30±1.88 ✓
- NeuMF-PL HR@5 = 5.18±1.25 ✓

✅ **Table 3** (Silhouette): Computed from `results/support_groups_full_164_loo/clustering/silhouette_*.csv`
- All optimal-k silhouette scores match exactly (within floating-point tolerance)
- K-selection: argmax cosine silhouette from {3,4,5,6,7,8,10}, tie-break to smallest k

---

## Projection Method Verification

All figures in paper use **t-SNE** (not UMAP), confirmed via metadata:
- Projection: t-SNE
- Perplexity: 15.0
- Normalize: L2 before projection
- Cluster space: HD (high-dimensional)
- Cluster method: Spherical k-means

**Directory name caveat**: Files are in `umap_plots/` directory (historical artifact) but metadata confirms they're t-SNE. Paper correctly states "t-SNE" throughout.

---

## Proposed Edits: NONE REQUIRED

The paper is already:
1. Methodologically sound
2. Numerically accurate
3. Clearly worded
4. Conservatively framed

### Optional Micro-Improvements (Cosmetic Only)

If you want ultra-polish for the camera-ready version:

1. **Add footnote on k-range choice** (currently lines 180-181):
   ```latex
   ... select the \emph{optimal $k$} (the $k$ maximizing cosine silhouette; tie-break: smallest $k$)
   ```
   Could add: `\footnote{The range $k\in\{3,\ldots,10\}$ is standard for clustering 166 users, balancing granularity and statistical power.}`

2. **Clarify directory naming** (currently line 345):
   Could add to reproducibility section:
   ```latex
   Note: 2D visualizations are saved in a directory named \texttt{umap\_plots/} for historical reasons, but metadata confirms all projections use t-SNE as stated.
   ```

**BUT**: These are purely cosmetic. Your paper is publication-ready as-is.

---

## Key Strengths

1. **Explicit methodology**: Paper clearly states clustering/silhouette are HD-only (line 183)
2. **Fair comparison**: K-selection applied uniformly (lines 68-69, 180)
3. **Code guardrails**: Implementation prevents accidental 2D clustering with deprecation warning
4. **Conservative claims**: AlignFeatures is a proxy; offline metrics ≠ outcomes (lines 321, 330)
5. **Full reproducibility**: All artifacts saved with metadata

---

## Reviewer Questions Answered

### Q1: Which projection method?
**A**: t-SNE with perplexity=15, cosine distance, L2-normalized embeddings. Verified via metadata JSON files.

### Q2: Where is clustering done?
**A**: Original high-dimensional embedding space. Verified via `cluster_space: "hd"` in all metadata + code guardrail at src/generate_umap_plots.py:640-651.

### Q3: Where is silhouette computed?
**A**: Original high-dimensional space. Verified via scripts/utils/compute_silhouette_scores.py (operates on HD embeddings) and plot metadata.

### Q4: Is K-selection fair?
**A**: Yes. All models use same procedure: for each model/seed/entity/repr, select k from {3,4,5,6,7,8,10} maximizing cosine silhouette (tie-break: smallest k). Verified via scripts/utils/get_optimal_k.py and silhouette CSVs.

### Q5: Any 2D clustering path?
**A**: No active path. Legacy 'umap' option exists but triggers DeprecationWarning and is coerced to 'hd'. Default is 'hd'. Verified via src/generate_umap_plots.py:640-651, 889-892.

---

## Recommendation

**Submit as-is.** Your paper satisfies all of Dr. Foulds' methodological requirements and presents repo-grounded, verifiable results.

If reviewers ask "how do we know clustering was done in HD space?", point them to:
1. Paper line 183: Explicit statement
2. Reproducibility section (line 345): All metadata available
3. Code: src/generate_umap_plots.py with guardrails

---

## Files Generated

1. **COMPLIANCE_AUDIT_REPORT.md**: Full technical audit (6 sections, evidence table)
2. **REVIEW_SUMMARY.md**: This file (executive summary)

Both saved in repo root. Ready to share with Dr. Foulds or reviewers.
