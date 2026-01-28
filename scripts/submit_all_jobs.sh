#!/bin/bash
# Master script to submit all jobs in the correct order with dependencies
# Usage: ./scripts/submit_all_jobs.sh

set -e

cd /umbc/rs/pi_jfoulds/pbarman1/pronob/pl-ncf-comparison

echo "============================================"
echo "Submitting All Jobs with Dependencies"
echo "============================================"
echo "Start time: $(date)"
echo ""

# Create log directories
mkdir -p logs/training logs/analysis

# Step 0: Submit training job (array job)
echo "📤 Submitting Step 0: Training (array job 0-59)..."
TRAIN_JOB_FULL=$(sbatch --parsable scripts/train_support_groups_only.sh)
# Extract just the numeric job ID (remove cluster suffix like ";chip-gpu")
TRAIN_JOB=$(echo $TRAIN_JOB_FULL | cut -d';' -f1)
echo "   Job ID: $TRAIN_JOB"
echo ""

# Step 1: Submit embedding extraction (depends on training)
echo "📤 Submitting Step 1: Extract Embeddings (depends on $TRAIN_JOB)..."
EMBED_JOB_FULL=$(sbatch --parsable --dependency=afterok:$TRAIN_JOB scripts/1_extract_embeddings.sh)
EMBED_JOB=$(echo $EMBED_JOB_FULL | cut -d';' -f1)
echo "   Job ID: $EMBED_JOB"
echo ""

# Step 2: Submit clustering metrics (depends on embeddings)
echo "📤 Submitting Step 2: Compute Clustering Metrics (depends on $EMBED_JOB)..."
CLUSTER_JOB_FULL=$(sbatch --parsable --dependency=afterok:$EMBED_JOB scripts/2_compute_clustering_metrics.sh)
CLUSTER_JOB=$(echo $CLUSTER_JOB_FULL | cut -d';' -f1)
echo "   Job ID: $CLUSTER_JOB"
echo ""

# Step 3: Submit UMAP plots (depends on embeddings AND clustering for optimal k)
echo "📤 Submitting Step 3: Generate UMAP Plots with ALL Optimal K (depends on $EMBED_JOB and $CLUSTER_JOB)..."
UMAP_JOB_FULL=$(sbatch --parsable --dependency=afterok:$EMBED_JOB:$CLUSTER_JOB scripts/3_generate_umap_plots_all_optimal.sh)
UMAP_JOB=$(echo $UMAP_JOB_FULL | cut -d';' -f1)
echo "   Job ID: $UMAP_JOB"
echo ""

# Step 4: Submit performance aggregation (depends on training)
echo "📤 Submitting Step 4: Aggregate Performance Metrics (depends on $TRAIN_JOB)..."
AGG_JOB_FULL=$(sbatch --parsable --dependency=afterok:$TRAIN_JOB scripts/4_aggregate_performance_metrics.sh)
AGG_JOB=$(echo $AGG_JOB_FULL | cut -d';' -f1)
echo "   Job ID: $AGG_JOB"
echo ""

# Step 5: Submit comparison plots (depends on aggregation)
echo "📤 Submitting Step 5: Generate Comparison Plots (depends on $AGG_JOB)..."
PLOT_JOB_FULL=$(sbatch --parsable --dependency=afterok:$AGG_JOB scripts/5_generate_comparison_plots.sh)
PLOT_JOB=$(echo $PLOT_JOB_FULL | cut -d';' -f1)
echo "   Job ID: $PLOT_JOB"
echo ""

# Step 6: Submit comprehensive CSV (depends on clustering and aggregation)
echo "📤 Submitting Step 6: Create Comprehensive CSV (depends on $CLUSTER_JOB and $AGG_JOB)..."
CSV_JOB_FULL=$(sbatch --parsable --dependency=afterok:$CLUSTER_JOB:$AGG_JOB scripts/6_create_comprehensive_csv.sh)
CSV_JOB=$(echo $CSV_JOB_FULL | cut -d';' -f1)
echo "   Job ID: $CSV_JOB"
echo ""

echo "============================================"
echo "All Jobs Submitted Successfully!"
echo "============================================"
echo ""
echo "Job Pipeline:"
echo "  Training:      $TRAIN_JOB (array 0-59)"
echo "  ├─> Embeddings:  $EMBED_JOB"
echo "  │   ├─> Clustering: $CLUSTER_JOB"
echo "  │   │   └─> UMAP:  $UMAP_JOB (ALL plots use silhouette-optimal k)"
echo "  │   └─> (UMAP also depends on embeddings)"
echo "  └─> Aggregation: $AGG_JOB"
echo "      └─> Plots:     $PLOT_JOB"
echo "  Comprehensive CSV: $CSV_JOB (after Clustering + Aggregation)"
echo ""
echo "Monitor with: squeue -u $USER"
echo "View dependencies: squeue -u $USER -o '%.18i %.9P %.30j %.8T %.10M %.9l %.6D %R %E'"
echo ""
