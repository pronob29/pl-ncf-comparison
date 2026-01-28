#!/bin/bash
#SBATCH --job-name=aggregate_metrics
#SBATCH --time=00:15:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --cluster=chip-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=logs/analysis/aggregate_%j.out
#SBATCH --error=logs/analysis/aggregate_%j.err

source ~/.bashrc 2>/dev/null || true
export PYTHON_ENV="/umbc/rs/pi_jfoulds/users/pbarman1/conda_envs/testenv"
export PATH="${PYTHON_ENV}/bin:$PATH"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:$PYTHONPATH"
cd $SLURM_SUBMIT_DIR

mkdir -p logs/analysis

echo "============================================"
echo "STEP 4: Aggregating Performance Metrics"
echo "============================================"
echo "Start time: $(date)"
echo ""

# Aggregate metrics for support_groups_full_164
echo "Dataset: support_groups_full_164"
python scripts/utils/aggregate_results_balanced.py \
    --dataset support_groups_full_164

echo ""
echo "---"
echo ""

# Aggregate metrics for support_groups_full_164_loo
echo "Dataset: support_groups_full_164_loo"
python scripts/utils/aggregate_results_balanced.py \
    --dataset support_groups_full_164_loo

echo ""
echo "============================================"
echo "Performance Metrics Aggregation Completed"
echo "End time: $(date)"
echo "============================================"
echo ""

# Show results
echo "Performance metrics saved to:"
echo "  results/support_groups_full_164/performance_metrics.csv"
echo "  results/support_groups_full_164/performance_metrics_all_seeds.csv"
echo "  results/support_groups_full_164_loo/performance_metrics.csv"
echo "  results/support_groups_full_164_loo/performance_metrics_all_seeds.csv"
