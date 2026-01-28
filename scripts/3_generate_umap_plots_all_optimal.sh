#!/bin/bash
#SBATCH --job-name=umap_plots_all_optimal
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --cluster=chip-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=logs/analysis/umap_plots_all_optimal_%j.out
#SBATCH --error=logs/analysis/umap_plots_all_optimal_%j.err

source ~/.bashrc 2>/dev/null || true
export PYTHON_ENV="/umbc/rs/pi_jfoulds/users/pbarman1/conda_envs/testenv"
export PATH="${PYTHON_ENV}/bin:$PATH"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:$PYTHONPATH"
cd $SLURM_SUBMIT_DIR

mkdir -p logs/analysis

echo "============================================"
echo "STEP 3: Generating t-SNE Visualizations with Optimal K for ALL Plots"
echo "============================================"
echo "Start time: $(date)"
echo ""
echo "⚠️  Using t-SNE for clearer visual cluster separation"
echo "   - Method: t-SNE (better visual clustering than UMAP)"
echo "   - Perplexity: 15 (tuned for small dataset ~166 users / ~498 items)"
echo "   - Cluster labels computed on HD embeddings (--cluster-space hd)"
echo "   - Professional color palette (tab10)"
echo ""
echo "   Settings: --method tsne --perplexity 15 --normalize --cluster-space hd"
echo ""
echo "🔧 NEW: ALL plots (including baselines) use silhouette-selected optimal k"
echo "   Candidate k values: {3, 4, 5, 6, 7, 8, 10}"
echo "   Selection: maximize cosine silhouette (tie-break: smallest k)"
echo "   Fallback: k=5 if silhouette data is missing"
echo ""

DATASETS=("support_groups_full_164" "support_groups_full_164_loo")
MODELS=("mf_baseline" "mf_pl" "mlp_baseline" "mlp_pl" "neumf_baseline" "neumf_pl")
SEEDS=(42 52 62 122 232)

# t-SNE settings for clear visual cluster separation
# t-SNE often produces cleaner, more distinct visual clusters than UMAP
# Cluster labels are computed in HD space (--cluster-space hd), then overlaid on t-SNE
#   - method=tsne: Use t-SNE instead of UMAP
#   - perplexity=15: Lower perplexity for small dataset (focuses on local structure)
#   - normalize: L2-normalize embeddings before t-SNE
PRESENTATION_FLAGS="--method tsne --perplexity 15 --normalize --cluster-space hd"

# Helper function to get optimal k from silhouette analysis
# Arguments: dataset model seed entity repr_type
get_optimal_k() {
    local dataset=$1
    local model=$2
    local seed=$3
    local entity=$4
    local repr_type=$5

    # Call Python helper to get optimal k from silhouette CSV
    local optimal_k=$(python scripts/utils/get_optimal_k.py "$dataset" "$model" "$seed" "$entity" --repr "$repr_type" 2>/dev/null || echo "5")
    echo "$optimal_k"
}

total_plots=0
success_plots=0

for dataset in "${DATASETS[@]}"; do
    echo "================================================================================"
    echo "Dataset: $dataset"
    echo "================================================================================"
    echo ""

    for model in "${MODELS[@]}"; do
        echo "  Model: $model"

        for seed in "${SEEDS[@]}"; do
            # Generate plots for USER embeddings

            # 1. Main embeddings (for ALL models) - use OPTIMAL k from silhouette
            k_main=$(get_optimal_k "$dataset" "$model" "$seed" "user" "main")
            echo "    [User/Main] ${model}_seed${seed} (optimal k=$k_main)"
            python src/generate_umap_plots.py \
                --model "$model" \
                --dataset "$dataset" \
                --seed "$seed" \
                --entity user \
                --repr main \
                --n-clusters "$k_main" \
                $PRESENTATION_FLAGS > /dev/null 2>&1

            if [ $? -eq 0 ]; then
                ((success_plots++))
            fi
            ((total_plots++))

            # 2. PL embeddings (only for PL models) - use optimal k from PL silhouette
            if [[ "$model" == *"_pl" ]]; then
                k_pl=$(get_optimal_k "$dataset" "$model" "$seed" "user" "pl")
                echo "    [User/PL]   ${model}_seed${seed} (optimal k=$k_pl)"
                python src/generate_umap_plots.py \
                    --model "$model" \
                    --dataset "$dataset" \
                    --seed "$seed" \
                    --entity user \
                    --repr pl \
                    --n-clusters "$k_pl" \
                    $PRESENTATION_FLAGS > /dev/null 2>&1

                if [ $? -eq 0 ]; then
                    ((success_plots++))
                fi
                ((total_plots++))
            fi

            # Generate plots for ITEM embeddings

            # 3. Main embeddings (for ALL models) - use OPTIMAL k from silhouette
            k_main=$(get_optimal_k "$dataset" "$model" "$seed" "item" "main")
            echo "    [Item/Main] ${model}_seed${seed} (optimal k=$k_main)"
            python src/generate_umap_plots.py \
                --model "$model" \
                --dataset "$dataset" \
                --seed "$seed" \
                --entity item \
                --repr main \
                --n-clusters "$k_main" \
                $PRESENTATION_FLAGS > /dev/null 2>&1

            if [ $? -eq 0 ]; then
                ((success_plots++))
            fi
            ((total_plots++))

            # 4. PL embeddings (only for PL models) - use optimal k from PL silhouette
            if [[ "$model" == *"_pl" ]]; then
                k_pl=$(get_optimal_k "$dataset" "$model" "$seed" "item" "pl")
                echo "    [Item/PL]   ${model}_seed${seed} (optimal k=$k_pl)"
                python src/generate_umap_plots.py \
                    --model "$model" \
                    --dataset "$dataset" \
                    --seed "$seed" \
                    --entity item \
                    --repr pl \
                    --n-clusters "$k_pl" \
                    $PRESENTATION_FLAGS > /dev/null 2>&1

                if [ $? -eq 0 ]; then
                    ((success_plots++))
                fi
                ((total_plots++))
            fi
        done
        echo ""
    done
done

echo ""
echo "================================================================================"
echo "t-SNE Visualization Completed (All Optimal K)"
echo "================================================================================"
echo "End time: $(date)"
echo ""
echo "Summary:"
echo "  Total plots attempted: $total_plots"
echo "  Successful plots: $success_plots"
echo "  Failed plots: $((total_plots - success_plots))"
echo ""
echo "t-SNE plots saved to:"
echo "  results/support_groups_full_164/umap_plots/"
echo "  results/support_groups_full_164_loo/umap_plots/"
echo ""

# Count files
count_164=$(ls -1 results/support_groups_full_164/umap_plots/*.png 2>/dev/null | wc -l)
count_164_loo=$(ls -1 results/support_groups_full_164_loo/umap_plots/*.png 2>/dev/null | wc -l)

echo "Plot counts:"
echo "  support_groups_full_164: $count_164 plots"
echo "  support_groups_full_164_loo: $count_164_loo plots"
echo ""
echo "================================================================================"
