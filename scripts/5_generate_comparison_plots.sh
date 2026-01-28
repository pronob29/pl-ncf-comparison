#!/bin/bash
#SBATCH --job-name=comparison_plots
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --cluster=chip-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=logs/analysis/comparison_plots_%j.out
#SBATCH --error=logs/analysis/comparison_plots_%j.err

source ~/.bashrc 2>/dev/null || true
export PYTHON_ENV="/umbc/rs/pi_jfoulds/users/pbarman1/conda_envs/testenv"
export PATH="${PYTHON_ENV}/bin:$PATH"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:$PYTHONPATH"
cd $SLURM_SUBMIT_DIR

mkdir -p logs/analysis

echo "============================================"
echo "STEP 5: Generating Comparison Plots"
echo "============================================"
echo "Start time: $(date)"
echo ""

# Generate comparison plots for support_groups_full_164
echo "Dataset: support_groups_full_164"
python src/generate_comparison_plots.py \
    --dataset support_groups_full_164

echo ""
echo "---"
echo ""

# Generate comparison plots for support_groups_full_164_loo
echo "Dataset: support_groups_full_164_loo"
python src/generate_comparison_plots.py \
    --dataset support_groups_full_164_loo

echo ""
echo "============================================"
echo "Comparison Plots Completed"
echo "End time: $(date)"
echo "============================================"
echo ""

# Show results
echo "Comparison plots saved to:"
echo "  results/support_groups_full_164/plots/"
echo "  results/support_groups_full_164_loo/plots/"
echo ""
ls -1 results/support_groups_full_164/plots/*.png 2>/dev/null | wc -l | xargs echo "support_groups_full_164:"
ls -1 results/support_groups_full_164_loo/plots/*.png 2>/dev/null | wc -l | xargs echo "support_groups_full_164_loo:"
