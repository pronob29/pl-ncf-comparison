#!/bin/bash
#SBATCH --job-name=extract_emb
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --cluster=chip-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=logs/analysis/extract_embeddings_%j.out
#SBATCH --error=logs/analysis/extract_embeddings_%j.err

source ~/.bashrc 2>/dev/null || true
export PYTHON_ENV="/umbc/rs/pi_jfoulds/users/pbarman1/conda_envs/testenv"
export PATH="${PYTHON_ENV}/bin:$PATH"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:$PYTHONPATH"
cd $SLURM_SUBMIT_DIR

mkdir -p logs/analysis

echo "============================================"
echo "STEP 1: Extracting User Embeddings"
echo "============================================"
echo "Start time: $(date)"
echo ""

# Extract embeddings for support_groups_full_164
echo "Dataset: support_groups_full_164"
python src/extract_embeddings.py \
    --dataset support_groups_full_164 \
    --all-models

echo ""
echo "---"
echo ""

# Extract embeddings for support_groups_full_164_loo
echo "Dataset: support_groups_full_164_loo"
python src/extract_embeddings.py \
    --dataset support_groups_full_164_loo \
    --all-models

echo ""
echo "============================================"
echo "Embedding Extraction Completed"
echo "End time: $(date)"
echo "============================================"
echo ""

# Show results
echo "Embeddings saved to:"
echo "  results/support_groups_full_164/embeddings/"
echo "  results/support_groups_full_164_loo/embeddings/"
echo ""
ls -lh results/support_groups_full_164/embeddings/*.npy 2>/dev/null | wc -l | xargs echo "support_groups_full_164:"
ls -lh results/support_groups_full_164_loo/embeddings/*.npy 2>/dev/null | wc -l | xargs echo "support_groups_full_164_loo:"
