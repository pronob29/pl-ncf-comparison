#!/bin/bash
# Complete Analysis Pipeline with Optimal K for PL Embeddings
# This script chains all analysis steps: embeddings → clustering → UMAP → aggregation → plots

set -e  # Exit on error

echo "================================================================================"
echo "COMPLETE ANALYSIS PIPELINE - PL-NCF Comparison"
echo "================================================================================"
echo "Start time: $(date)"
echo ""
echo "This pipeline will:"
echo "  1. Extract embeddings (main + PL-specific)"
echo "  2. Compute clustering metrics (silhouette scores for k=3,4,5,6,7,8,10)"
echo "  3. Aggregate fair clustering summaries (fixed-k and optimal-k)"
echo "  4. Generate UMAP plots (with optimal k for PL embeddings)"
echo "  5. Aggregate performance metrics"
echo "  6. Create comparison plots"
echo ""
echo "================================================================================"
echo ""

# Check if we're in SLURM environment
if [ -n "$SLURM_JOB_ID" ]; then
    echo "Running in SLURM environment (Job ID: $SLURM_JOB_ID)"
    IN_SLURM=true
else
    echo "Running in local environment"
    IN_SLURM=false
fi

# Step 1: Extract Embeddings
echo ""
echo "================================================================================"
echo "STEP 1: Extracting Embeddings"
echo "================================================================================"
if [ "$IN_SLURM" = true ]; then
    JOB1=$(sbatch --parsable scripts/1_extract_embeddings.sh)
    echo "Submitted job: $JOB1"
    # Wait for completion
    while squeue -j $JOB1 2>/dev/null | grep -q $JOB1; do
        sleep 30
    done
    echo "Step 1 completed"
else
    bash scripts/1_extract_embeddings.sh
fi

# Step 2: Compute Clustering Metrics
echo ""
echo "================================================================================"
echo "STEP 2: Computing Clustering Metrics"
echo "================================================================================"
if [ "$IN_SLURM" = true ]; then
    JOB2=$(sbatch --parsable scripts/2_compute_clustering_metrics.sh)
    echo "Submitted job: $JOB2"
    while squeue -j $JOB2 2>/dev/null | grep -q $JOB2; do
        sleep 30
    done
    echo "Step 2 completed"
else
    bash scripts/2_compute_clustering_metrics.sh
fi

# Step 3: Aggregate Fair Clustering Summaries
echo ""
echo "================================================================================"
echo "STEP 3: Aggregating Fair Clustering Summaries"
echo "================================================================================"
python scripts/utils/aggregate_clustering_fair.py

# Step 4: Generate UMAP Plots with Optimal K for ALL plots (including baselines)
echo ""
echo "================================================================================"
echo "STEP 4: Generating UMAP Plots (Optimal K for ALL Embeddings)"
echo "================================================================================"
if [ "$IN_SLURM" = true ]; then
    JOB4=$(sbatch --parsable scripts/3_generate_umap_plots_all_optimal.sh)
    echo "Submitted job: $JOB4"
    while squeue -j $JOB4 2>/dev/null | grep -q $JOB4; do
        sleep 30
    done
    echo "Step 4 completed"
else
    bash scripts/3_generate_umap_plots_all_optimal.sh
fi

# Step 5: Aggregate Performance Metrics
echo ""
echo "================================================================================"
echo "STEP 5: Aggregating Performance Metrics"
echo "================================================================================"
if [ "$IN_SLURM" = true ]; then
    JOB5=$(sbatch --parsable scripts/4_aggregate_performance_metrics.sh)
    echo "Submitted job: $JOB5"
    while squeue -j $JOB5 2>/dev/null | grep -q $JOB5; do
        sleep 30
    done
    echo "Step 5 completed"
else
    bash scripts/4_aggregate_performance_metrics.sh
fi

# Step 6: Generate Comparison Plots
echo ""
echo "================================================================================"
echo "STEP 6: Generating Comparison Plots"
echo "================================================================================"
if [ "$IN_SLURM" = true ]; then
    JOB6=$(sbatch --parsable scripts/5_generate_comparison_plots.sh)
    echo "Submitted job: $JOB6"
    while squeue -j $JOB6 2>/dev/null | grep -q $JOB6; do
        sleep 30
    done
    echo "Step 6 completed"
else
    bash scripts/5_generate_comparison_plots.sh
fi

echo ""
echo "================================================================================"
echo "COMPLETE ANALYSIS PIPELINE FINISHED"
echo "================================================================================"
echo "End time: $(date)"
echo ""
echo "Results available in:"
echo "  - results/support_groups_full_164/"
echo "  - results/support_groups_full_164_loo/"
echo ""
echo "Key outputs:"
echo "  - Embeddings: results/{dataset}/embeddings/*.npy"
echo "  - Clustering: results/{dataset}/clustering/silhouette_*.csv"
echo "  - Fair clustering summaries: results/comprehensive_results/clustering_summary_{fixed_k5,optimal_k}.csv"
echo "  - UMAP plots: results/{dataset}/umap_plots/*.png (with optimal k for PL)"
echo "  - Metrics: results/{dataset}/performance_metrics.csv"
echo "  - Plots: results/{dataset}/plots/*.png"
echo ""
echo "================================================================================"
