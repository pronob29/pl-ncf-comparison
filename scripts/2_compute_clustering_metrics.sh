#!/bin/bash
#SBATCH --job-name=clustering
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --cluster=chip-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=logs/analysis/clustering_%j.out
#SBATCH --error=logs/analysis/clustering_%j.err

source ~/.bashrc 2>/dev/null || true
export PYTHON_ENV="/umbc/rs/pi_jfoulds/users/pbarman1/conda_envs/testenv"
export PATH="${PYTHON_ENV}/bin:$PATH"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:$PYTHONPATH"
cd $SLURM_SUBMIT_DIR

mkdir -p logs/analysis

echo "============================================"
echo "STEP 2: Computing Clustering Quality Metrics"
echo "============================================"
echo "Start time: $(date)"
echo ""

# Compute silhouette scores for support_groups_full_164
echo "Dataset: support_groups_full_164"
python scripts/utils/compute_silhouette_scores.py \
    --dataset support_groups_full_164 \
    --k-values 3 4 5 6 7 8 10

echo ""
echo "---"
echo ""

# Compute silhouette scores for support_groups_full_164_loo
echo "Dataset: support_groups_full_164_loo"
python scripts/utils/compute_silhouette_scores.py \
    --dataset support_groups_full_164_loo \
    --k-values 3 4 5 6 7 8 10

echo ""
echo "============================================"
echo "Clustering Metrics Completed"
echo "End time: $(date)"
echo "============================================"
echo ""

# Show results
echo "Clustering metrics saved to:"
echo "  results/support_groups_full_164/clustering/silhouette_scores.csv"
echo "  results/support_groups_full_164_loo/clustering/silhouette_scores.csv"
