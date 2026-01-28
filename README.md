# Neural Collaborative Filtering with Pseudo-Labeling

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**Enhancing recommendation systems and embedding quality through pseudo-label supervision**

This repository implements and evaluates Neural Collaborative Filtering (NCF) models enhanced with Pseudo-Labeling (PL) for support group recommendations. PL models demonstrate **superior embedding clustering quality** (up to +147% improvement in silhouette scores) while achieving better recommendation performance (up to +100% improvement in HR@5).

## Key Findings

- **Dual Representation Learning**: PL models learn separate embeddings for recommendation (main) and semantic clustering (PL-specific)
- **Dramatically Better Clustering**: PL-specific embeddings achieve +7.5% to +147% improvement in silhouette scores vs main embeddings
- **Improved Recommendations**: PL variants show up to +100% improvement in HR@5 for leave-one-out evaluation (MLP PL)
- **Consistent Improvement**: All PL models outperform their baselines in HR@5 and NDCG@5 metrics

---

## Table of Contents

- [Quick Start](#quick-start)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Training Models](#training-models)
  - [Analysis Pipeline](#analysis-pipeline)
  - [Custom Experiments](#custom-experiments)
- [Datasets](#datasets)
- [Results](#results)
- [Technical Details](#technical-details)
- [Citation](#citation)
- [License](#license)

---

## Quick Start

```bash
# Clone repository
git clone <repository-url>
cd pl-ncf-comparison

# Install dependencies
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Train a single model
python src/train_val_test.py \
  --model neumf_pl \
  --dataset support_groups_full_164_loo \
  --seed 42 \
  --epochs 20

# Run full analysis pipeline (requires trained models)
bash scripts/run_all_new_evaluations.sh
```

---

## Repository Structure

```
pl-ncf-comparison/
├── README.md                      # This file
├── TECHNICAL_DETAILS.md           # Model architectures, methodology, full results
├── UMAP_GENERATION_GUIDE.md       # UMAP visualization documentation
├── LICENSE                        # MIT License
│
├── datasets/                      # Dataset files
│   ├── support_groups_full_164/
│   └── support_groups_full_164_loo/
│
├── src/                           # Source code
│   ├── train_val_test.py         # Main training script
│   ├── extract_embeddings.py     # Extract trained embeddings
│   ├── generate_umap_plots.py    # Create UMAP visualizations
│   ├── mf.py                     # Matrix Factorization model
│   ├── mlp.py                    # MLP model
│   ├── neumf.py                  # NeuMF model
│   ├── pl.py                     # Pseudo-label embedding module
│   ├── engine.py                 # Training engine
│   ├── data.py                   # Data loading utilities
│   └── metrics.py                # Evaluation metrics
│
├── scripts/                       # Execution scripts
│   ├── train_support_groups_only.sh          # SLURM job array for training
│   ├── run_all_analysis_optimal.sh           # Full analysis pipeline
│   ├── 1_extract_embeddings.sh               # Extract embeddings
│   ├── 2_compute_clustering_metrics.sh       # Compute silhouette scores
│   ├── 3_generate_umap_plots_optimal.sh      # Generate UMAP plots
│   ├── 4_aggregate_performance_metrics.sh    # Aggregate metrics
│   ├── 5_generate_comparison_plots.sh        # Create comparison plots
│   └── 6_create_comprehensive_csv.sh         # Create summary CSVs
│
├── models/trained/                # Saved model checkpoints (60 models)
│
├── results/                       # Experimental results
│   ├── comprehensive_results/    # Aggregated metrics across datasets
│   ├── support_groups_full_164/
│   │   ├── embeddings/           # Extracted embeddings (.npy)
│   │   ├── clustering/           # Silhouette scores (.csv)
│   │   └── umap_plots/           # Visualizations (.png, .json)
│   └── support_groups_full_164_loo/
│       └── ... (same structure)
│
└── logs/                          # Training and analysis logs
```

---

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.9+
- CUDA (optional, for GPU training)

### Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import umap; print('UMAP installed successfully')"
```

### Dependencies

Core libraries:
- `torch>=1.9.0` - Deep learning framework
- `numpy>=1.20.0` - Numerical computing
- `pandas>=1.3.0` - Data manipulation
- `scikit-learn>=0.24.0` - Clustering and metrics
- `umap-learn>=0.5.0` - Dimensionality reduction
- `matplotlib>=3.4.0` - Plotting
- `seaborn>=0.11.0` - Statistical visualization
- `tqdm>=4.62.0` - Progress bars

---

## Usage

### Training Models

#### Single Model Training

```bash
python src/train_val_test.py \
  --model {mf_baseline|mf_pl|mlp_baseline|mlp_pl|neumf_baseline|neumf_pl} \
  --dataset {support_groups_full_164|support_groups_full_164_loo} \
  --seed {42|52|62|122|232} \
  --epochs 20 \
  --batch-size 512 \
  --lr 0.001
```

**Example**: Train NeuMF with pseudo-labeling
```bash
python src/train_val_test.py \
  --model neumf_pl \
  --dataset support_groups_full_164_loo \
  --seed 42 \
  --epochs 20
```

#### Batch Training (SLURM)

Train all 60 models (6 models × 2 datasets × 5 seeds):

```bash
# Submit job array
sbatch scripts/train_support_groups_only.sh

# Monitor progress
squeue -u $USER
```

**Configuration**:
- Models: `mf_baseline`, `mf_pl`, `mlp_baseline`, `mlp_pl`, `neumf_baseline`, `neumf_pl`
- Datasets: `support_groups_full_164`, `support_groups_full_164_loo`
- Seeds: 42, 52, 62, 122, 232
- GPU: 1x A100 (40GB) per job
- Runtime: ~15 minutes per model

### Analysis Pipeline

Run the complete analysis pipeline on trained models:

```bash
bash scripts/run_all_analysis_optimal.sh
```

**Pipeline steps**:

1. **Extract Embeddings** (`1_extract_embeddings.sh`)
   - Extracts user and item embeddings from trained models
   - Outputs: `results/{dataset}/embeddings/*.npy`

2. **Compute Clustering Metrics** (`2_compute_clustering_metrics.sh`)
   - Computes silhouette scores for k ∈ {3,4,5,6,7,8,10}
   - Outputs: `results/{dataset}/clustering/silhouette_*.csv`

3. **Generate UMAP Plots** (`3_generate_umap_plots_all_optimal.sh`)
   - Creates 2D visualizations with presentation mode for clear cluster separation
   - **ALL plots** (main + PL, including baselines) use silhouette-optimal k from {3,4,5,6,7,8,10}
   - Uses optimal visualization parameters: n_neighbors=50, min_dist=0.0, repulsion_strength=2.0
   - Outputs: `results/{dataset}/umap_plots/*.png` and `*.json`
   - See [UMAP_GENERATION_GUIDE.md](UMAP_GENERATION_GUIDE.md) for details

4. **Aggregate Performance** (`4_aggregate_performance_metrics.sh`)
   - Summarizes HR@5, NDCG@5, AUC across seeds
   - Outputs: `results/{dataset}/performance_metrics.csv`

5. **Create Comparison Plots** (`5_generate_comparison_plots.sh`)
   - Generates bar plots comparing models
   - Outputs: `results/{dataset}/plots/*.png`

6. **Create Comprehensive CSVs** (`6_create_comprehensive_csv.sh`)
   - Aggregates all metrics into summary files
   - Outputs: `results/comprehensive_results/*.csv`

### Custom Experiments

#### Extract Embeddings from Specific Model

```bash
python src/extract_embeddings.py \
  --model neumf_pl \
  --dataset support_groups_full_164_loo \
  --seed 42
```

#### Generate UMAP Plots

```bash
# Generate main embedding UMAP with silhouette-optimal k
OPTIMAL_K_MAIN=$(python scripts/utils/get_optimal_k.py support_groups_full_164_loo neumf_pl 42 user --repr main)
python src/generate_umap_plots.py \
  --model neumf_pl \
  --dataset support_groups_full_164_loo \
  --seed 42 \
  --entity user \
  --repr main \
  --n-clusters $OPTIMAL_K_MAIN

# Generate PL embedding UMAP with silhouette-optimal k
OPTIMAL_K_PL=$(python scripts/utils/get_optimal_k.py support_groups_full_164_loo neumf_pl 42 user --repr pl)
python src/generate_umap_plots.py \
  --model neumf_pl \
  --dataset support_groups_full_164_loo \
  --seed 42 \
  --entity user \
  --repr pl \
  --n-clusters $OPTIMAL_K_PL

# Verify generated plots
python scripts/utils/verify_umap_plots.py --dataset support_groups_full_164_loo

# Strict verification (also checks k matches silhouette-optimal)
python scripts/utils/verify_umap_plots.py --strict
```

#### Compute Silhouette Scores

```bash
python scripts/utils/compute_silhouette_scores.py \
  --dataset support_groups_full_164_loo \
  --output results/support_groups_full_164_loo/clustering/
```

---

## Datasets

### support_groups_full_164

- **Users**: 166
- **Items (support groups)**: 498
- **Evaluation**: 80/10/10 train/validation/test split
- **Use case**: Standard recommendation evaluation

### support_groups_full_164_loo

- **Users**: 166
- **Items (support groups)**: 498
- **Evaluation**: Leave-One-Out (LOO)
  - Training: All interactions except one per user
  - Test: One held-out interaction per user
- **Use case**: Challenging ranking evaluation (test item vs. all items)

### AlignFeatures Signal

Both datasets include **survey-derived alignment features**:
- User preferences (topics, support needs)
- Group characteristics (topics covered, focus areas)
- Used to generate pseudo-labels: `cosine_similarity(user_features, item_features)`

---

## Results

### Recommendation Performance (LOO Evaluation)

| Model | HR@5 | NDCG@5 | AUC |
|-------|------|--------|-----|
| MF Baseline | 0.0458 | 0.0270 | 0.5024 |
| **MF PL** | **0.0542** ↑18.3% | **0.0332** ↑23.0% | 0.4766 |
| MLP Baseline | 0.0265 | 0.0141 | 0.4618 |
| **MLP PL** | **0.0530** ↑100.0% | **0.0297** ↑110.6% | **0.4843** |
| NeuMF Baseline | 0.0446 | 0.0250 | 0.4547 |
| **NeuMF PL** | **0.0518** ↑16.1% | **0.0302** ↑20.8% | **0.4977** |

### Clustering Quality (User Embeddings, k=3)

| Model | Main Embeddings | PL Embeddings | Improvement |
|-------|-----------------|---------------|-------------|
| MF Baseline | 0.0385 | — | — |
| **MF PL** | 0.0265 | **0.0650** | **+145%** vs PL main |
| MLP Baseline | 0.0646 | — | — |
| **MLP PL** | 0.0650 | **0.0699** | **+7.5%** vs PL main, **+8.2%** vs baseline |
| NeuMF Baseline | 0.0258 | — | — |
| **NeuMF PL** | 0.0255 | **0.0629** | **+147%** vs PL main, **+144%** vs baseline |

*Cosine silhouette scores at k=3, support_groups_full_164_loo, mean across 5 seeds*

📊 **See [TECHNICAL_DETAILS.md](TECHNICAL_DETAILS.md) for complete results and methodology**

---

## Technical Details

### Model Architectures

- **MF (Matrix Factorization)**: Dot product of user and item embeddings
- **MLP (Multi-Layer Perceptron)**: Deep network on concatenated embeddings
- **NeuMF (Neural Matrix Factorization)**: Hybrid GMF + MLP fusion

### Pseudo-Labeling

PL models add a second branch that:
1. Learns separate PL-specific embeddings
2. Predicts alignment scores using cosine similarity
3. Minimizes MSE loss against feature-based pseudo-labels

**Training objective**:
```
Total Loss = (1-λ) × BCE(predictions, labels) + λ × MSE(pl_predictions, pseudo_labels)
```

### Clustering Technique

- **Method**: Spherical K-means (L2-normalized embeddings)
- **Metric**: Cosine silhouette score
- **Optimal K**: Selected per model from {3,4,5,6,7,8,10}

### Why PL Models Have Better Clustering

1. **Dual representation learning**: Decoupled objectives for prediction vs. semantics
2. **Explicit semantic supervision**: Pseudo-labels encode feature-based similarity
3. **Cosine alignment**: Training loss aligns with clustering evaluation metric

🔬 **See [TECHNICAL_DETAILS.md](TECHNICAL_DETAILS.md) for in-depth explanations**

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{pl-ncf-comparison,
  title={Neural Collaborative Filtering with Pseudo-Labeling for Enhanced Embedding Quality},
  author={Your Name},
  year={2025},
  howpublished={\url{https://github.com/yourusername/pl-ncf-comparison}}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- NCF framework based on He et al. "Neural Collaborative Filtering" (WWW 2017)
- Pseudo-labeling inspired by Lee "Pseudo-Label: The Simple and Efficient Semi-Supervised Learning Method" (ICML Workshop 2013)
- UMAP implementation from McInnes et al. (2018)

---

## Contact

For questions or issues, please open a GitHub issue or contact the repository maintainer.

---

**Last Updated**: January 2026
