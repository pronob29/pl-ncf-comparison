# UMAP Plot Generation Guide

This guide explains how to generate UMAP visualizations for both main and PL-specific embeddings with appropriate clustering parameters.

## Overview

The UMAP generation workflow creates 2D visualizations of learned embeddings. **ALL plots** (main and PL, including baselines) use silhouette-selected optimal k for clustering:

- **Optimal k selection**: Chosen from candidate set {3, 4, 5, 6, 7, 8, 10} by maximizing cosine silhouette score
- **Tie-break rule**: If multiple k values share the same max score, the smallest k is chosen (stable + predictable)
- **Fallback**: k=5 is used only when silhouette data is missing or invalid
- **Source files**: `results/{dataset}/clustering/silhouette_{entity}_{repr}.csv`

## File Organization

### Scripts

- **`scripts/3_generate_umap_plots_all_optimal.sh`** - **RECOMMENDED**: Generates ALL plots with silhouette-optimal k (including baselines)
- **`scripts/3_generate_umap_plots_optimal.sh`** - **DEPRECATED**: Uses fixed k=5 for main embeddings (not recommended)
- **`scripts/utils/get_optimal_k.py`** - Helper to read optimal k from silhouette CSVs (supports `--repr {main,pl}`)
- **`scripts/utils/verify_umap_plots.py`** - Validation script to check generated files (supports `--strict` mode)

### Generated Files

UMAP plots are saved to:
```
results/{dataset}/umap_plots/
├── {model}_seed{seed}_{entity}_main_umap.png       # Main embedding visualization
├── {model}_seed{seed}_{entity}_main_metrics.json   # Main embedding metrics
├── {model}_seed{seed}_{entity}_pl_umap.png         # PL embedding visualization (PL models only)
└── {model}_seed{seed}_{entity}_pl_metrics.json     # PL embedding metrics (PL models only)
```

**Naming Convention:**
- `{model}`: `mf_baseline`, `mf_pl`, `mlp_baseline`, `mlp_pl`, `neumf_baseline`, `neumf_pl`
- `{seed}`: `42`, `52`, `62`, `122`, `232`
- `{entity}`: `user`, `item`
- `{repr}`: `main`, `pl`

**Example:**
```
neumf_pl_seed42_user_main_umap.png     # NeuMF-PL seed 42, user, main embedding, optimal k
neumf_pl_seed42_user_main_metrics.json # Corresponding metrics (n_clusters from silhouette)
neumf_pl_seed42_user_pl_umap.png       # NeuMF-PL seed 42, user, PL embedding, optimal k
neumf_pl_seed42_user_pl_metrics.json   # Corresponding metrics (n_clusters from silhouette)
mf_baseline_seed42_user_main_umap.png  # MF baseline also uses optimal k from silhouette
```

## Expected Output Counts

For each dataset (`support_groups_full_164`, `support_groups_full_164_loo`):

- **Main plots**: 6 models × 5 seeds × 2 entities = 60 plots
- **PL plots**: 3 PL models × 5 seeds × 2 entities = 30 plots
- **Total per dataset**: 90 plots + 90 metrics JSON = 180 files
- **Total for both datasets**: 360 files

## Commands to Regenerate Plots

### Option 1: Submit SLURM Job (Recommended)

```bash
# Generate ALL UMAP plots with silhouette-optimal k values
sbatch scripts/3_generate_umap_plots_all_optimal.sh
```

The job will:
- Run on GPU partition (required for UMAP acceleration)
- Generate main plots for ALL models (optimal k from silhouette analysis)
- Generate PL plots for PL models (optimal k from silhouette analysis)
- Use presentation mode settings for clearer visual separation
- Complete in ~1-2 hours depending on cluster load

### Option 2: Run Locally (Interactive)

```bash
# Set up environment
source ~/.bashrc
export PYTHON_ENV="/umbc/rs/pi_jfoulds/users/pbarman1/conda_envs/testenv"
export PATH="${PYTHON_ENV}/bin:$PATH"
export PYTHONPATH="${PWD}/src:$PYTHONPATH"

# Run the all-optimal script
bash scripts/3_generate_umap_plots_all_optimal.sh
```

### Option 3: Generate Specific Plots

```bash
# Get optimal k for main embeddings (works for baselines and PL models)
OPTIMAL_K_MAIN=$(python scripts/utils/get_optimal_k.py support_groups_full_164 mf_baseline 42 user --repr main)
echo "Optimal k for main: $OPTIMAL_K_MAIN"

# Get optimal k for PL embeddings (only for PL models)
OPTIMAL_K_PL=$(python scripts/utils/get_optimal_k.py support_groups_full_164 neumf_pl 42 user --repr pl)
echo "Optimal k for PL: $OPTIMAL_K_PL"

# Generate a single plot (main embedding) with silhouette-optimal k
OPTIMAL_K=$(python scripts/utils/get_optimal_k.py support_groups_full_164 neumf_pl 42 user --repr main)
python src/generate_umap_plots.py \
    --model neumf_pl \
    --dataset support_groups_full_164 \
    --seed 42 \
    --entity user \
    --repr main \
    --n-clusters $OPTIMAL_K

# Generate a single PL plot with silhouette-optimal k
OPTIMAL_K=$(python scripts/utils/get_optimal_k.py support_groups_full_164 neumf_pl 42 user --repr pl)
python src/generate_umap_plots.py \
    --model neumf_pl \
    --dataset support_groups_full_164 \
    --seed 42 \
    --entity user \
    --repr pl \
    --n-clusters $OPTIMAL_K
```

## Validation

After generation, verify that all expected files exist and are valid:

```bash
# Verify all datasets
python scripts/utils/verify_umap_plots.py

# Verify specific dataset
python scripts/utils/verify_umap_plots.py --dataset support_groups_full_164

# Quiet mode (summary only)
python scripts/utils/verify_umap_plots.py --quiet
```

The verification script checks:
- ✅ All expected PNG and JSON files exist
- ✅ JSON files have correct `repr` field (`main` or `pl`)
- ✅ ALL plots (main AND PL) use k ∈ {3,4,5,6,7,8,10}
- ✅ PL embeddings have lower dimensionality than main (typically 32 vs 64-96)

**Strict mode** (`--strict`): Also verifies that `n_clusters` exactly matches the silhouette-optimal k:
```bash
python scripts/utils/verify_umap_plots.py --strict
```

**Example output:**
```
================================================================================
Verifying UMAP plots for: support_groups_full_164
================================================================================

Expected files: 180 (90 PNG + 90 JSON)
Found valid: 180 (90 PNG + 90 JSON)
Errors: 0

✅ All UMAP plots validated successfully!
```

## Checking Individual Metrics

To verify a specific plot has the correct representation and parameters:

```bash
# Check main representation (now uses silhouette-optimal k, not fixed k=5)
cat results/support_groups_full_164/umap_plots/neumf_pl_seed42_user_main_metrics.json
# Expected: "repr": "main", "n_clusters": 3 (or other optimal k), "embedding_shape": [166, 96]

# Check PL representation
cat results/support_groups_full_164/umap_plots/neumf_pl_seed42_user_pl_metrics.json
# Expected: "repr": "pl", "n_clusters": 6 (or other optimal k), "embedding_shape": [166, 32]

# Check baseline main representation (also uses optimal k)
cat results/support_groups_full_164/umap_plots/mf_baseline_seed42_user_main_metrics.json
# Expected: "repr": "main", "n_clusters": 4 (or other optimal k), "embedding_shape": [166, 64]
```

## Optimal K Selection

**ALL plots** (main + PL, including baselines) use optimal k determined by:

1. Reading silhouette scores from `results/{dataset}/clustering/silhouette_{entity}_{repr}.csv`
2. Finding the k ∈ {3,4,5,6,7,8,10} that maximizes cosine silhouette score
3. Tie-breaking with smallest k if multiple values share the same max score
4. Using that k for clustering in the UMAP visualization

**Examples:**
```bash
# Get optimal k for main embeddings
$ python scripts/utils/get_optimal_k.py support_groups_full_164 mf_baseline 42 user --repr main
Optimal k=4 (cosine silhouette=0.0402)
4

# Get optimal k for PL embeddings
$ python scripts/utils/get_optimal_k.py support_groups_full_164 neumf_pl 42 user --repr pl
Optimal k=6 (cosine silhouette=0.0611)
6
```

If silhouette data is missing, the script defaults to k=5 with a warning (includes context info to stderr).

## Troubleshooting

### Missing Plots

If plots are missing after running the optimal script:

1. Check the SLURM log: `logs/analysis/umap_plots_optimal_<jobid>.out`
2. Look for error messages or failed plot generation
3. Verify embeddings exist: `ls results/{dataset}/embeddings/{model}_seed{seed}_{entity}_emb.npy`
4. For PL plots, verify: `ls results/{dataset}/embeddings/{model}_seed{seed}_pl_{entity}_emb.npy`

### Validation Errors

If verification fails:

```bash
# Check specific error
python scripts/utils/verify_umap_plots.py --dataset support_groups_full_164

# Regenerate missing plots
sbatch scripts/3_generate_umap_plots_optimal.sh
```

### Environment Issues

If running locally and encountering import errors:

```bash
# Activate the correct conda environment
source ~/.bashrc
export PYTHON_ENV="/umbc/rs/pi_jfoulds/users/pbarman1/conda_envs/testenv"
export PATH="${PYTHON_ENV}/bin:$PATH"

# Verify dependencies
python -c "import umap, matplotlib, numpy; print('OK')"
```

## Integration with Paper

The paper references UMAP plots using paths like:

```latex
\includegraphics{results/support_groups_full_164/umap_plots/neumf_pl_seed42_user_main_umap.png}
\includegraphics{results/support_groups_full_164/umap_plots/neumf_pl_seed42_user_pl_umap.png}
```

The naming convention `{model}_seed{seed}_{entity}_{repr}_umap.png` ensures that:
- Main and PL plots are clearly distinguished
- No overwrites occur when generating plots in parallel
- File paths are deterministic and reproducible

## UMAP Visualization (Qualitative Only)

**CRITICAL GUIDANCE FROM DR. FOULDS:**
> "Weak UMAP visuals and purely geometric separation should not be overinterpreted without clear semantic grounding."

### Key Principles

1. **UMAP is for qualitative visualization ONLY** - do not judge clustering quality by UMAP appearance
2. **Visual overlap in 2D is EXPECTED** - UMAP distorts distances and densities
3. **Quantitative evaluation uses HD silhouette scores** - not UMAP visual patterns
4. **Do NOT artificially enhance separation** - this would be misleading

### UMAP Settings

Standard parameters are used. **Do NOT try to make UMAP "look better":**

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `--normalize` | True | L2 normalization for consistent scale |
| `--n-neighbors` | 15 | Standard UMAP neighborhood size |
| `--min-dist` | 0.1 | Standard minimum distance |
| `--cluster-space` | hd | **Required**: Clusters computed on HD embeddings only |

### Correct Interpretation

- **Cluster colors** show labels computed via spherical K-means on **original HD embeddings**
- **Visual separation** (or lack thereof) in 2D does NOT indicate clustering quality
- **HD silhouette scores** are the quantitative measure of cluster separation
- **No metrics are computed on 2D** - UMAP projection is purely for visualization

### Quick Start: Presentation UMAPs

```bash
# Generate presentation UMAP for a specific model
python src/generate_umap_plots.py \
    --model neumf_pl \
    --dataset support_groups_full_164 \
    --seed 42 \
    --entity user \
    --repr pl \
    --presentation

# Generate with custom k value
python src/generate_umap_plots.py \
    --model neumf_pl \
    --dataset support_groups_full_164 \
    --seed 42 \
    --entity user \
    --repr pl \
    --n-clusters 6 \
    --presentation
```

### Advanced Options

You can also fine-tune individual parameters without using `--presentation`:

```bash
# Custom preprocessing only
python src/generate_umap_plots.py \
    --model neumf_pl \
    --dataset support_groups_full_164 \
    --seed 42 \
    --entity user \
    --repr pl \
    --normalize
```

### Available Options

**Preprocessing:**
- `--normalize`: L2-normalize embeddings before UMAP
- `--pca-dim N`: Apply PCA to reduce to N dimensions before UMAP

**UMAP Hyperparameters:**
- `--n-neighbors N`: Number of neighbors (default: 15)
- `--min-dist F`: Minimum distance between points (default: 0.1)
- `--spread F`: Effective scale of embedded points (default: 1.0)
- `--repulsion-strength F`: Strength of repulsion between points (default: 1.0)
- `--random-state N`: Random seed for UMAP (defaults to `--seed`)

**Clustering:**
- `--cluster-space {hd,umap}`: Cluster on high-dimensional embeddings
  - `hd` (default and required): Clusters computed on original HD embeddings, overlaid on UMAP
  - `umap`: **DEPRECATED** - Will be coerced to `hd` with a warning. Clustering in 2D UMAP space is methodologically unsound because UMAP distorts distances and densities.

**Presentation Mode:**
- `--presentation`: Applies all recommended defaults for clearer separation

### Important Warnings

⚠️ **UMAP is for visualization only:**
- UMAP is a non-linear dimensionality reduction that distorts distances and densities
- **Cluster labels are always computed on HD embeddings**, then overlaid on UMAP
- Silhouette scores are computed on HD embeddings
- The 2D scatter plot is purely for visual inspection of cluster structure
- These settings do **NOT** affect model training
- Evaluation metrics (HR, NDCG, AUC) remain unchanged

⚠️ **The `--cluster-space umap` option is DEPRECATED:**
- This option is now coerced to `hd` with a deprecation warning
- Clustering in 2D UMAP space was methodologically unsound
- All plots now use HD clustering for correctness

⚠️ **Backward Compatibility:**
- Without new flags, behavior matches existing outputs exactly
- Old commands continue to work as before
- Default settings preserve original visualization approach

### Metadata Tracking

All visualization parameters are saved in the metrics JSON file:

```json
{
  "preprocessing": {
    "normalize_l2": true,
    "pca_dim": 50
  },
  "umap_params": {
    "n_neighbors": 50,
    "min_dist": 0.0,
    "spread": 1.0,
    "repulsion_strength": 2.0,
    "metric": "cosine",
    "random_state": 42
  },
  "clustering": {
    "n_clusters": 6,
    "method": "spherical_kmeans",
    "cluster_space": "hd"
  }
}
```

**Note**: `cluster_space` is always `"hd"` (high-dimensional). Cluster labels are computed via spherical K-means on the original embeddings, then overlaid on the 2D UMAP projection for visualization.

This ensures full reproducibility and transparency about visualization choices.

## Summary

**Quick Start:**
```bash
# Generate all plots with silhouette-optimal k for ALL models
sbatch scripts/3_generate_umap_plots_all_optimal.sh

# Verify results
python scripts/utils/verify_umap_plots.py

# Strict verification (also checks k matches silhouette-optimal)
python scripts/utils/verify_umap_plots.py --strict
```

**Key Points:**
- **ALL plots** (main + PL, including baselines) use silhouette-optimal k from {3,4,5,6,7,8,10}
- Optimal k maximizes cosine silhouette; ties broken by smallest k
- k=5 is used ONLY as fallback when silhouette data is missing
- Both main and PL plots are generated for PL models
- Only main plots are generated for baseline models
- Total: 360 files (180 plots + 180 metrics JSON)
- Use `--presentation` flag for enhanced visual separation in presentations
- All visualization parameters are tracked in metrics JSON files
