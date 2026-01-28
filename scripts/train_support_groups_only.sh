#!/bin/bash
#SBATCH --job-name=train_sg
#SBATCH --time=72:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=4
#SBATCH --cluster=chip-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=logs/training/train_sg_%A_%a.out
#SBATCH --error=logs/training/train_sg_%A_%a.err
#SBATCH --array=0-59

source ~/.bashrc 2>/dev/null || true
export PYTHON_ENV="/umbc/rs/pi_jfoulds/users/pbarman1/conda_envs/testenv"
export PATH="${PYTHON_ENV}/bin:$PATH"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:$PYTHONPATH"

# Performance optimizations
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_LAUNCH_BLOCKING=0

cd $SLURM_SUBMIT_DIR
mkdir -p logs/training

DATASETS=("support_groups_full_164" "support_groups_full_164_loo")
MODELS=("mf_baseline" "mf_pl" "mlp_baseline" "mlp_pl" "neumf_baseline" "neumf_pl")

# Use same seeds for all models (42, 52, 62)
SEEDS=(42 52 62 122 232)

task_id=${SLURM_ARRAY_TASK_ID}
n_models=${#MODELS[@]}
n_seeds=5

seed_idx=$((task_id % n_seeds))
model_idx=$(((task_id / n_seeds) % n_models))
dataset_idx=$((task_id / (n_seeds * n_models)))

dataset=${DATASETS[$dataset_idx]}
model=${MODELS[$model_idx]}
seed=${SEEDS[$seed_idx]}

echo "Training $model on $dataset (seed=$seed)"

python src/train_val_test.py \
    --model $model \
    --dataset $dataset \
    --seed $seed \
    --epochs 20

echo "✅ Completed: $model on $dataset (seed=$seed)"
