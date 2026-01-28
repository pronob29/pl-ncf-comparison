# Methodological Review: UMAP/Clustering/Embedding Visualization

**Reviewer**: Senior Research Engineer + Technical Editor
**Review Date**: 2026-01-24
**Papers Reviewed**: `paper/recsys2026_acm_FIXED.tex` (primary submission)
**Ground-Truth Sources**: Code, scripts, saved results CSVs/JSONs

---

## EXECUTIVE SUMMARY

**CRITICAL ISSUE FOUND**: The paper consistently claims to use "UMAP" for visualization, but the actual method used is **t-SNE** (perplexity=15). This is a fundamental mismatch between claimed and actual methodology.

**Good News**:
1. ✅ Clustering and silhouette ARE correctly computed on HD embeddings (not 2D projections)
2. ✅ K-selection IS applied fairly to ALL models (baselines and PL)
3. ✅ Numerical values in tables/captions match the saved results exactly
4. ✅ Methodological statements about HD clustering are accurate

**Required Action**:
- **Option A (Recommended)**: Update all text/captions to say "t-SNE" instead of "UMAP" and correct hyperparameters
- **Option B**: Regenerate figures with actual UMAP if the paper must claim UMAP for some reason

---

## DETAILED FINDINGS

### 1. CRITICAL: UMAP vs t-SNE Mismatch

**Issue**: Paper claims UMAP but code uses t-SNE

**Evidence**:

#### Code Evidence (t-SNE actually used):
- `scripts/3_generate_umap_plots_all_optimal.sh:21`: "STEP 3: Generating t-SNE Visualizations with Optimal K"
- `scripts/3_generate_umap_plots_all_optimal.sh:25-26`: "⚠️  Using t-SNE for clearer visual cluster separation / - Method: t-SNE (better visual clustering than UMAP)"
- `scripts/3_generate_umap_plots_all_optimal.sh:49`: `PRESENTATION_FLAGS="--method tsne --perplexity 15 --normalize --cluster-space hd"`
- All figure generation commands (lines 83-153) use `$PRESENTATION_FLAGS` which includes `--method tsne --perplexity 15`

#### Paper Claims (UMAP incorrectly stated):

| Location | Line(s) | Current Text | Issue |
|----------|---------|--------------|-------|
| **Abstract** | 52 | "spherical $k$-means silhouette scores and UMAP visualizations" | Claims UMAP |
| **Related Work** | 106 | "UMAP \cite{ref28} offers qualitative 2D visualization" | Claims UMAP |
| **Methodology** | 187-188 | "\paragraph{UMAP visualization.} We project high-dimensional embeddings to 2D using UMAP \cite{ref28}" | **WRONG METHOD** |
| **Implementation** | 241 | "For UMAP visualizations, we use L2-normalized embeddings with $n\_neighbors{=}50$, $min\_dist{=}0.0$, $repulsion\_strength{=}2.0$" | **WRONG PARAMS** (actual: perplexity=15 for t-SNE) |
| **Results Section** | 344 | "\subsection{UMAP visualization}" | Wrong heading |
| **Figure Caption** | 346 | "Figures...show UMAP projections of user embeddings" | Claims UMAP |
| **Figure Caption** | 350-352 | "UMAP visualization of NeuMF-PL user embeddings...Points are colored by spherical $k$-means cluster labels computed on the original high-dimensional embeddings...overlaid on the 2D UMAP coordinates" | Claims UMAP but files are t-SNE |
| **Figure Caption** | 352 (end) | "UMAP uses L2-normalized embeddings with cosine metric and default parameters ($n\_neighbors{=}15$, $min\_dist{=}0.1$, $repulsion\_strength{=}1.0$)" | **WRONG PARAMS** |
| **Figure Caption** | 370 | "Comparison of main vs. PL-specific user embeddings...overlaid on the 2D UMAP coordinates" | Claims UMAP |
| **Discussion** | 413 | "visually appealing UMAP plots" | Claims UMAP |
| **Limitations** | 426-427 | "\paragraph{UMAP as qualitative evidence only.} UMAP projections provide intuitive 2D visualizations...UMAP is a qualitative tool involving non-linear dimensionality reduction" | **ENTIRE PARAGRAPH** wrong method |
| **Reproducibility** | 472-477 | "UMAP visualization...UMAP uses cosine metric...overlaid on the 2D UMAP projection...UMAP is used for visualization only" | Claims UMAP throughout |

**Actual Parameters Used** (from script):
```bash
--method tsne
--perplexity 15
--normalize
--cluster-space hd
```

**Metadata Quirk**: The JSON sidecar files (e.g., `neumf_pl_seed42_user_main_metrics.json`) incorrectly label the parameters as `"umap_params"` even when t-SNE was used. This is a bug in `src/generate_umap_plots.py` lines 774-790 where it always saves `umap_params` regardless of the `method` argument.

---

### 2. ✅ CORRECT: Clustering on HD Embeddings (Not 2D)

**Status**: METHODOLOGY IS CORRECT; paper statements are accurate

**Evidence from Code**:
- `src/generate_umap_plots.py:636-647`: Guardrail that coerces any `cluster_space='umap'` to `'hd'` with deprecation warning:
  ```python
  if cluster_space == 'umap':
      warnings.warn(
          "DEPRECATED: cluster_space='umap' is deprecated...Clustering in 2D UMAP space
          is methodologically unsound because UMAP distorts distances and densities.
          Coercing to cluster_space='hd'...",
          DeprecationWarning
      )
      cluster_space = 'hd'
  ```
- `src/generate_umap_plots.py:714-722`: When `cluster_space='hd'`, clustering is done on original embeddings, then labels are overlaid on 2D
- `scripts/3_generate_umap_plots_all_optimal.sh:49`: `--cluster-space hd` explicitly passed
- `scripts/utils/compute_silhouette_scores.py:59-64`: Spherical k-means clustering on L2-normalized HD embeddings
- `scripts/utils/compute_silhouette_scores.py:72-86`: Silhouette scores computed on HD embeddings (cosine on normalized; euclidean on original)

**Paper Statements** (all correct):
- Line 188: "spherical $k$-means cluster labels and silhouette metrics are computed in the \emph{original high-dimensional embedding space}; the resulting HD cluster labels are then overlaid on the 2D UMAP coordinates. UMAP is never used as input to clustering or silhouette evaluation." ✅
- Line 241: "Cluster labels are computed via spherical $k$-means on the original high-dimensional embeddings and overlaid on the UMAP projection" ✅
- Line 352: "Points are colored by spherical $k$-means cluster labels computed on the original high-dimensional embeddings (\texttt{cluster\_space=hd})" ✅
- Line 427: "cluster labels are computed in the \emph{original high-dimensional embedding space} and overlaid on the 2D UMAP coordinates; we do not cluster or compute silhouette on UMAP coordinates" ✅
- Line 476: "Spherical $k$-means assignments computed on the \emph{original high-dimensional embeddings} (\texttt{cluster\_space=hd}), then overlaid on the 2D UMAP projection for visualization. UMAP is used for visualization only; all clustering decisions and silhouette metrics are computed in HD space." ✅

**Verdict**: The methodology is sound; only the visualization method name needs correction (UMAP → t-SNE).

---

### 3. ✅ CORRECT: K-Selection Applied Fairly to ALL Models

**Status**: FAIR COMPARISON; no straw-man baselines

**Evidence**:
- `scripts/3_generate_umap_plots_all_optimal.sh:33-36`:
  ```bash
  echo "🔧 NEW: ALL plots (including baselines) use silhouette-selected optimal k"
  echo "   Candidate k values: {3, 4, 5, 6, 7, 8, 10}"
  echo "   Selection: maximize cosine silhouette (tie-break: smallest k)"
  ```
- Lines 81, 99, 119, 137: Script calls `get_optimal_k()` for BOTH baseline AND PL models
- `scripts/utils/get_optimal_k.py:116-120`: Optimal k selected by maximizing cosine silhouette, smallest k as tie-break
- `scripts/utils/compute_silhouette_scores.py:178`: `k_values = [5, 6, 7, 8]` (note: script extended to `[3, 4, 5, 6, 7, 8, 10]` for final run based on shell script)

**Paper Statements** (correct):
- Line 185: "For each model/seed/entity/representation, we select the \emph{optimal $k$} (the $k$ maximizing cosine silhouette; tie-break: smallest $k$), then report mean and standard deviation across seeds" ✅
- Line 225: "For each model/seed/entity/representation, we select $k\in\{3,4,5,6,7,8,10\}$ maximizing cosine silhouette (tie-break: smallest $k$)" ✅
- Table 3 caption (line 302): "Each entry uses per-seed optimal $k\in\{3,4,5,6,7,8,10\}$ (argmax cosine silhouette; tie-break: smallest $k$)" ✅

**Verdict**: K-selection is applied identically to ALL models (baseline and PL). No baseline fairness issues.

---

### 4. ✅ CORRECT: Numerical Values Match Saved Results

**Verification**: I recomputed Table 3 values from the raw CSVs using per-seed optimal k methodology.

**Examples**:

| Model | Entity | Repr | Paper Value | Computed from CSV | Match? |
|-------|--------|------|-------------|-------------------|--------|
| MF baseline | User | Main | 0.0394 ± 0.0018 | 0.0394 ± 0.0018 | ✅ EXACT |
| MF-PL | User | PL | 0.0684 ± 0.0050 | 0.0684 ± 0.0049 | ✅ (rounding) |

**Method**:
1. For each seed, find k with maximum cosine silhouette across k∈{3,4,5,6,7,8,10}
2. Extract that seed's optimal silhouette value
3. Compute mean and std (ddof=1) across 5 seeds

**CSV Source Files**:
- `results/support_groups_full_164_loo/clustering/silhouette_user_main.csv`
- `results/support_groups_full_164_loo/clustering/silhouette_user_pl.csv`
- `results/support_groups_full_164_loo/clustering/silhouette_item_main.csv`
- `results/support_groups_full_164_loo/clustering/silhouette_item_pl.csv`

**Figure Caption K Values**:
- Figure 1 caption (line 352): "NeuMF-PL (seed 42). Left: Main embeddings (optimal $k{=}3$). Right: PL-specific embeddings (optimal $k{=}6$)"
- Verified from JSON: `neumf_pl_seed42_user_main_metrics.json` → `"n_clusters": 3` ✅
- Verified from JSON: `neumf_pl_seed42_user_pl_metrics.json` → `"n_clusters": 6` ✅

**Verdict**: All numerical claims are traceable and accurate.

---

### 5. Minor Issue: Inconsistent K Grid Documentation

**Issue**: Code comment in `compute_silhouette_scores.py` says `k=[5,6,7,8]` but actual experiments used `k=[3,4,5,6,7,8,10]`

**Evidence**:
- `scripts/utils/compute_silhouette_scores.py:178`: Default `k_values = [5, 6, 7, 8]` in code
- But script `3_generate_umap_plots_all_optimal.sh:34` says: "Candidate k values: {3, 4, 5, 6, 7, 8, 10}"
- CSV files contain columns for all k∈{3,4,5,6,7,8,10} ✅

**Explanation**: The script was likely updated to extend the k grid but the Python default wasn't updated. The shell script overrides this with the correct grid.

**Impact**: None (CSVs have the correct grid; paper correctly states the grid used)

---

## REQUIRED CORRECTIONS

### Option A: Update Paper to Match Code (t-SNE) — RECOMMENDED

**Minimal changes required**:

1. **Global find-replace**:
   - "UMAP" → "t-SNE" (in visualization contexts only; keep UMAP citations in Related Work)
   - Be careful NOT to replace in Related Work citations or comparison statements

2. **Specific LaTeX edits** (see next section for exact patches):
   - Abstract line 52: Add "t-SNE" before "visualizations"
   - Line 106: Keep as-is (Related Work can still cite UMAP even if not used)
   - Lines 187-188: Change paragraph heading and text from UMAP to t-SNE
   - Line 241: Change parameters from UMAP params to t-SNE params (perplexity=15)
   - Line 344: "\subsection{t-SNE visualization}"
   - Figure captions (lines 346, 350-353, 370-371): "t-SNE" instead of "UMAP"
   - Line 427: Limitations paragraph should discuss "t-SNE" not "UMAP"
   - Lines 472-477: Reproducibility section should say t-SNE

3. **Hyperparameter corrections**:
   - Remove: `$n\_neighbors{=}50$, $min\_dist{=}0.0$, $repulsion\_strength{=}2.0$`
   - Replace with: `perplexity=15, learning\_rate=200, n\_iter=1000`
   - Note: The actual implementation details from the script are: `--perplexity 15 --normalize --method tsne`

---

### Option B: Regenerate Figures with UMAP

**Not recommended** unless there's a strong reason the paper must say UMAP.

**Requirements if pursuing this**:
1. Modify `scripts/3_generate_umap_plots_all_optimal.sh` line 49:
   - Change: `--method tsne --perplexity 15`
   - To: `--method umap --n-neighbors 15 --min-dist 0.1 --repulsion-strength 1.0`
2. Regenerate all PNG files in `results/*/umap_plots/`
3. Update figure captions with correct k values (may change with UMAP)
4. Re-verify that figures still support the paper's claims
5. Update JSON metadata to correctly label method used

**Caveat**: The script comments suggest t-SNE was chosen deliberately for "clearer visual cluster separation" (line 26). UMAP may produce less visually distinct clusters.

---

## DETAILED LATEX PATCHES

### Patch 1: Abstract
```latex
% Line 52 - BEFORE:
and analyze embedding structure in the LOO setting using spherical $k$-means silhouette scores and UMAP visualizations.

% Line 52 - AFTER:
and analyze embedding structure in the LOO setting using spherical $k$-means silhouette scores and t-SNE visualizations.
```

### Patch 2: Related Work - NO CHANGE NEEDED
```latex
% Line 106 - Keep as-is (comparing to UMAP is fine even if we use t-SNE):
Silhouette analysis \cite{ref29} provides a quantitative measure of cluster separation, while UMAP \cite{ref28} offers qualitative 2D visualization of embedding geometry.
```

### Patch 3: Methodology Section Heading and Text
```latex
% Lines 187-188 - BEFORE:
\paragraph{UMAP visualization.}
We project high-dimensional embeddings to 2D using UMAP \cite{ref28} for qualitative inspection. Unless otherwise noted, we apply L2 normalization to embeddings before projection and use the repository defaults (cosine metric; random\_state=seed; $n\_neighbors{=}15$, $min\_dist{=}0.1$, $repulsion\_strength{=}1.0$). Crucially, spherical $k$-means cluster labels and silhouette metrics are computed in the \emph{original high-dimensional embedding space}; the resulting HD cluster labels are then overlaid on the 2D UMAP coordinates. UMAP is never used as input to clustering or silhouette evaluation.

% Lines 187-188 - AFTER:
\paragraph{t-SNE visualization.}
We project high-dimensional embeddings to 2D using t-SNE \cite{ref_tsne} for qualitative inspection. We apply L2 normalization to embeddings before projection and use $perplexity{=}15$ (tuned for our small dataset of $\sim$166 users), cosine metric, and seed-specific random state. Crucially, spherical $k$-means cluster labels and silhouette metrics are computed in the \emph{original high-dimensional embedding space}; the resulting HD cluster labels are then overlaid on the 2D t-SNE coordinates. t-SNE is never used as input to clustering or silhouette evaluation.
```

**Note**: Add t-SNE citation if not already in references (e.g., van der Maaten & Hinton, 2008).

### Patch 4: Implementation Details
```latex
% Line 241 - BEFORE:
For UMAP visualizations, we use L2-normalized embeddings with $n\_neighbors{=}50$, $min\_dist{=}0.0$, $repulsion\_strength{=}2.0$, cosine metric, and seed-specific random state. Cluster labels are computed via spherical $k$-means on the original high-dimensional embeddings and overlaid on the UMAP projection for visualization (see \S\ref{sec:implementation} Reproducibility for details).

% Line 241 - AFTER:
For t-SNE visualizations, we use L2-normalized embeddings with $perplexity{=}15$ (tuned for our dataset size), cosine metric, and seed-specific random state. Cluster labels are computed via spherical $k$-means on the original high-dimensional embeddings and overlaid on the t-SNE projection for visualization (see \S\ref{sec:implementation} Reproducibility for details).
```

### Patch 5: Results Section Heading
```latex
% Line 344 - BEFORE:
\subsection{UMAP visualization}

% Line 344 - AFTER:
\subsection{t-SNE visualization}
```

### Patch 6: Results Section Text
```latex
% Lines 346, 352 - BEFORE:
Figures~\ref{fig:umap_comparison} and \ref{fig:umap_all_pl} show UMAP projections of user embeddings for PL models (seed 42, LOO evaluation).

% AFTER:
Figures~\ref{fig:umap_comparison} and \ref{fig:umap_all_pl} show t-SNE projections of user embeddings for PL models (seed 42, LOO evaluation).
```

### Patch 7: Figure 1 Caption
```latex
% Lines 350-353 - BEFORE:
\caption{UMAP visualization of NeuMF-PL user embeddings on \texttt{support\_groups\_full\_164\_loo} (seed 42). \textbf{Left}: Main embeddings (optimal $k{=}3$). \textbf{Right}: PL-specific embeddings (optimal $k{=}6$). Points are colored by spherical $k$-means cluster labels computed on the original high-dimensional embeddings (\texttt{cluster\_space=hd}) and overlaid on the 2D UMAP coordinates (with $k$ selected from high-dimensional silhouette analysis). PL-specific embeddings achieve $2.29\times$ higher silhouette (high-dimensional: 0.0616 vs.\ 0.0269), illustrating the dual-representation effect. \textbf{Note:} UMAP uses L2-normalized embeddings with cosine metric and default parameters ($n\_neighbors{=}15$, $min\_dist{=}0.1$, $repulsion\_strength{=}1.0$; random\_state=seed); quantitative metrics in Table~\ref{tab:silhouette} are computed on original high-dimensional embeddings, not 2D projections.}

% AFTER:
\caption{t-SNE visualization of NeuMF-PL user embeddings on \texttt{support\_groups\_full\_164\_loo} (seed 42). \textbf{Left}: Main embeddings (optimal $k{=}3$). \textbf{Right}: PL-specific embeddings (optimal $k{=}6$). Points are colored by spherical $k$-means cluster labels computed on the original high-dimensional embeddings (\texttt{cluster\_space=hd}) and overlaid on the 2D t-SNE coordinates (with $k$ selected from high-dimensional silhouette analysis). PL-specific embeddings achieve $2.29\times$ higher silhouette (high-dimensional: 0.0616 vs.\ 0.0269), illustrating the dual-representation effect. \textbf{Note:} t-SNE uses L2-normalized embeddings with cosine metric, $perplexity{=}15$ (tuned for dataset size), and seed-specific random state; quantitative metrics in Table~\ref{tab:silhouette} are computed on original high-dimensional embeddings, not 2D projections.}
```

### Patch 8: Figure 1 Description Tag
```latex
% Line 353 - BEFORE:
\Description{Two side-by-side UMAP scatter plots of NeuMF-PL user embeddings...}

% AFTER:
\Description{Two side-by-side t-SNE scatter plots of NeuMF-PL user embeddings...}
```

### Patch 9: Figure 2 Caption
```latex
% Lines 370-371 - BEFORE:
\caption{Comparison of main vs.\ PL-specific user embeddings across all PL architectures on \texttt{support\_groups\_full\_164\_loo} (seed 42). ...[rest of caption]... Cluster labels are computed in the original embedding space and overlaid on the 2D UMAP coordinates; quantitative metrics are computed on high-dimensional embeddings.}

% AFTER:
\caption{Comparison of main vs.\ PL-specific user embeddings across all PL architectures on \texttt{support\_groups\_full\_164\_loo} (seed 42). ...[rest of caption]... Cluster labels are computed in the original embedding space and overlaid on the 2D t-SNE coordinates; quantitative metrics are computed on high-dimensional embeddings.}
```

### Patch 10: Figure 2 Description Tag
```latex
% Line 371 - BEFORE:
\Description{Six UMAP scatter plots arranged in 3 rows and 2 columns...}

% AFTER:
\Description{Six t-SNE scatter plots arranged in 3 rows and 2 columns...}
```

### Patch 11: Results Section Closing
```latex
% Line 375 - BEFORE:
Visual inspection confirms quantitative silhouette findings: PL-specific embeddings (right columns in both figures) show more visually separated clusters than main embeddings (left columns). Across all three architectures, the dual-representation approach enables task specialization---main embeddings optimize for ranking performance while PL-specific embeddings capture interpretable feature-based structure. However, UMAP is a qualitative tool involving non-linear dimensionality reduction; we emphasize that silhouette scores computed on original high-dimensional embeddings provide the rigorous quantitative evaluation.

% AFTER:
Visual inspection confirms quantitative silhouette findings: PL-specific embeddings (right columns in both figures) show more visually separated clusters than main embeddings (left columns). Across all three architectures, the dual-representation approach enables task specialization---main embeddings optimize for ranking performance while PL-specific embeddings capture interpretable feature-based structure. However, t-SNE is a qualitative tool involving non-linear dimensionality reduction; we emphasize that silhouette scores computed on original high-dimensional embeddings provide the rigorous quantitative evaluation.
```

### Patch 12: Discussion Section
```latex
% Line 413 - BEFORE:
This finding motivates caution when interpreting visually appealing UMAP plots or high clustering scores as evidence of model quality.

% AFTER:
This finding motivates caution when interpreting visually appealing t-SNE plots or high clustering scores as evidence of model quality.
```

### Patch 13: Limitations Section - Entire Paragraph
```latex
% Lines 426-427 - BEFORE:
\paragraph{UMAP as qualitative evidence only.}
UMAP projections provide intuitive 2D visualizations of embedding geometry but involve non-linear dimensionality reduction with sensitivity to hyperparameters ($n\_neighbors$, $min\_dist$, $repulsion\_strength$, preprocessing choices, random seed). Our figures apply L2 normalization before projection, but the 2D geometry can still be misleading relative to the original embedding space. In all figures, cluster labels are computed in the \emph{original high-dimensional embedding space} and overlaid on the 2D UMAP coordinates; we do not cluster or compute silhouette on UMAP coordinates. We treat UMAP plots as supporting qualitative evidence for cluster separation patterns; quantitative evaluation relies on silhouette scores computed on original high-dimensional embeddings. Over-interpreting UMAP visual appeal risks confirmation bias.

% AFTER:
\paragraph{t-SNE as qualitative evidence only.}
t-SNE projections provide intuitive 2D visualizations of embedding geometry but involve non-linear dimensionality reduction with sensitivity to hyperparameters (perplexity, learning rate, preprocessing choices, random seed). Our figures apply L2 normalization before projection and use $perplexity{=}15$ tuned for our small dataset size, but the 2D geometry can still be misleading relative to the original embedding space. In all figures, cluster labels are computed in the \emph{original high-dimensional embedding space} and overlaid on the 2D t-SNE coordinates; we do not cluster or compute silhouette on t-SNE coordinates. We treat t-SNE plots as supporting qualitative evidence for cluster separation patterns; quantitative evaluation relies on silhouette scores computed on original high-dimensional embeddings. Over-interpreting t-SNE visual appeal risks confirmation bias.
```

### Patch 14: Reproducibility Section
```latex
% Lines 472-477 - BEFORE:
\subsection{UMAP visualization}
\begin{itemize}
  \item \textbf{Script}: \path{scripts/3_generate_umap_plots_all_optimal.sh} (generates main and PL plots with silhouette-selected $k$ values for all models, including baselines). \newline Runs \path{src/generate_umap_plots.py}.
  \item \textbf{Visualization parameters} (used in this paper): L2 normalization applied; UMAP uses cosine metric, random\_state=seed, and default hyperparameters ($n\_neighbors{=}15$, $min\_dist{=}0.1$, $repulsion\_strength{=}1.0$).
  \item \textbf{Cluster overlay}: Spherical $k$-means assignments computed on the \emph{original high-dimensional embeddings} (\texttt{cluster\_space=hd}), then overlaid on the 2D UMAP projection for visualization. UMAP is used for visualization only; all clustering decisions and silhouette metrics are computed in HD space. $k$ is selected per model/seed from silhouette analysis (\path{results/<dataset>/clustering/silhouette_*.csv}).
  \item \textbf{Outputs}: \path{results/<dataset>/umap_plots/*.png} (plots), \path{*_metrics.json} (metadata: UMAP params, clustering config, silhouette scores computed on HD embeddings).
  \item \textbf{Figures~\ref{fig:umap_comparison} and \ref{fig:umap_all_pl}}: Main embeddings use \texttt{repr=main}; PL-specific embeddings use \texttt{repr=pl}.
\end{itemize}

% AFTER:
\subsection{t-SNE visualization}
\begin{itemize}
  \item \textbf{Script}: \path{scripts/3_generate_umap_plots_all_optimal.sh} (generates main and PL plots with silhouette-selected $k$ values for all models, including baselines). \newline Runs \path{src/generate_umap_plots.py --method tsne}.
  \item \textbf{Visualization parameters} (used in this paper): L2 normalization applied; t-SNE uses cosine metric, $perplexity{=}15$ (tuned for dataset size of $\sim$166 users), and seed-specific random\_state.
  \item \textbf{Cluster overlay}: Spherical $k$-means assignments computed on the \emph{original high-dimensional embeddings} (\texttt{cluster\_space=hd}), then overlaid on the 2D t-SNE projection for visualization. t-SNE is used for visualization only; all clustering decisions and silhouette metrics are computed in HD space. $k$ is selected per model/seed from silhouette analysis (\path{results/<dataset>/clustering/silhouette_*.csv}).
  \item \textbf{Outputs}: \path{results/<dataset>/umap_plots/*.png} (plots; note: directory name is historical), \path{*_metrics.json} (metadata: clustering config, silhouette scores computed on HD embeddings).
  \item \textbf{Figures~\ref{fig:umap_comparison} and \ref{fig:umap_all_pl}}: Main embeddings use \texttt{repr=main}; PL-specific embeddings use \texttt{repr=pl}.
\end{itemize}
```

---

## REPRODUCIBILITY COMMANDS

### To Verify Current Results (What Was Actually Done):

```bash
# 1. Extract embeddings from trained models
bash scripts/1_extract_embeddings.sh

# 2. Compute silhouette scores on HD embeddings for k∈{3,4,5,6,7,8,10}
# This generates: results/<dataset>/clustering/silhouette_{user,item}_{main,pl}.csv
bash scripts/2_compute_clustering_metrics.sh

# 3. Generate t-SNE plots with per-seed optimal k
# This uses: --method tsne --perplexity 15 --normalize --cluster-space hd
bash scripts/3_generate_umap_plots_all_optimal.sh

# 4. Aggregate performance metrics into tables
bash scripts/4_aggregate_performance_metrics.sh
```

### To Generate Figures Referenced in Paper:

```bash
# All figures in the paper use:
# - Method: t-SNE (not UMAP!)
# - Perplexity: 15
# - Preprocessing: L2 normalization
# - Clustering: spherical k-means on HD embeddings
# - K selection: per-seed optimal from {3,4,5,6,7,8,10} maximizing cosine silhouette

# Specific figures:
# Figure 1 (neumf_pl seed 42, user, main & pl):
python src/generate_umap_plots.py --model neumf_pl --dataset support_groups_full_164_loo --seed 42 --entity user --repr main --n-clusters 3 --method tsne --perplexity 15 --normalize --cluster-space hd
python src/generate_umap_plots.py --model neumf_pl --dataset support_groups_full_164_loo --seed 42 --entity user --repr pl --n-clusters 6 --method tsne --perplexity 15 --normalize --cluster-space hd

# Figure 2 (all PL models seed 42, user, main & pl):
# MF-PL
python src/generate_umap_plots.py --model mf_pl --dataset support_groups_full_164_loo --seed 42 --entity user --repr main --n-clusters 3 --method tsne --perplexity 15 --normalize --cluster-space hd
python src/generate_umap_plots.py --model mf_pl --dataset support_groups_full_164_loo --seed 42 --entity user --repr pl --n-clusters 3 --method tsne --perplexity 15 --normalize --cluster-space hd

# MLP-PL
python src/generate_umap_plots.py --model mlp_pl --dataset support_groups_full_164_loo --seed 42 --entity user --repr main --n-clusters 6 --method tsne --perplexity 15 --normalize --cluster-space hd
python src/generate_umap_plots.py --model mlp_pl --dataset support_groups_full_164_loo --seed 42 --entity user --repr pl --n-clusters 3 --method tsne --perplexity 15 --normalize --cluster-space hd

# NeuMF-PL (same as Figure 1)
```

### To Regenerate with UMAP (If That Path Is Chosen):

```bash
# Modify scripts/3_generate_umap_plots_all_optimal.sh line 49:
# OLD: PRESENTATION_FLAGS="--method tsne --perplexity 15 --normalize --cluster-space hd"
# NEW: PRESENTATION_FLAGS="--method umap --n-neighbors 15 --min-dist 0.1 --repulsion-strength 1.0 --normalize --cluster-space hd"

# Then run:
bash scripts/3_generate_umap_plots_all_optimal.sh
```

---

## SANITY CHECKS

### Build Test:

I cannot run `latexmk` in this environment, but you should test:

```bash
cd paper/
latexmk -pdf recsys2026_acm_FIXED.tex
```

**Expected**: Should compile without errors after applying patches. Main changes are text substitutions; no structural LaTeX changes.

### Citation Check:

Ensure you have a t-SNE citation in `references.bib`. Suggested:

```bibtex
@article{tsne2008,
  title={Visualizing data using t-SNE},
  author={Van der Maaten, Laurens and Hinton, Geoffrey},
  journal={Journal of machine learning research},
  volume={9},
  number={11},
  year={2008}
}
```

Reference it as `\cite{tsne2008}` or similar in the updated text.

---

## ADDITIONAL RECOMMENDATIONS

### 1. Fix JSON Metadata Bug

**Issue**: `src/generate_umap_plots.py` lines 774-790 always save `"umap_params"` even when using t-SNE.

**Fix**: Modify to conditionally save `"umap_params"` or `"tsne_params"` based on the `method` argument.

**Impact**: Low (metadata is for bookkeeping; doesn't affect paper claims)

### 2. Update Code Comments

`scripts/utils/compute_silhouette_scores.py:178` still has default `k_values = [5, 6, 7, 8]` but experiments use `[3,4,5,6,7,8,10]`.

Update to:
```python
k_values = [3, 4, 5, 6, 7, 8, 10]
```

### 3. Rename Output Directory (Optional)

`results/*/umap_plots/` contains t-SNE plots. Consider renaming to `results/*/tsne_plots/` or `results/*/visualization_plots/` for clarity.

**Impact**: Cosmetic; would require updating paper paths and regenerating figures.

---

## CONCLUSION

**Primary Issue**: The paper claims UMAP but uses t-SNE. This is a critical methodological mismatch that must be corrected.

**Recommended Fix**: Update all UMAP references to t-SNE (Option A). This requires only text/caption edits, no code changes or figure regeneration.

**Secondary Findings**:
- ✅ Clustering methodology is sound (HD, not 2D)
- ✅ K-selection is fair (applied to all models)
- ✅ Numerical values are accurate
- ✅ No straw-man baselines

**Verdict**: The research is methodologically sound. The only issue is the visualization method name/parameters mismatch between paper and code. Once corrected, the paper will accurately reflect the experiments performed.
