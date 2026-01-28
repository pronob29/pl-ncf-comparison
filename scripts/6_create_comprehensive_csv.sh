#!/bin/bash
#SBATCH --job-name=comprehensive_csv
#SBATCH --time=00:15:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --cluster=chip-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=logs/analysis/comprehensive_csv_%j.out
#SBATCH --error=logs/analysis/comprehensive_csv_%j.err

source ~/.bashrc 2>/dev/null || true
export PYTHON_ENV="/umbc/rs/pi_jfoulds/users/pbarman1/conda_envs/testenv"
export PATH="${PYTHON_ENV}/bin:$PATH"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:$PYTHONPATH"
cd $SLURM_SUBMIT_DIR

mkdir -p logs/analysis

echo "============================================"
echo "STEP 6: Creating Comprehensive Results CSV"
echo "============================================"
echo "Start time: $(date)"
echo ""

python << 'PYTHON_EOF'
import pandas as pd
from pathlib import Path

datasets = ['support_groups_full_164', 'support_groups_full_164_loo']
all_results = []

print("Merging performance and clustering metrics...\n")

for dataset in datasets:
    print(f"Processing {dataset}...")

    # Load performance metrics
    perf_file = Path(f'results/{dataset}/performance_metrics.csv')
    if perf_file.exists():
        perf_df = pd.read_csv(perf_file)

        # Load clustering metrics
        sil_file = Path(f'results/{dataset}/clustering/silhouette_scores.csv')
        if sil_file.exists():
            sil_df = pd.read_csv(sil_file)

            # Merge on dataset, model, and seed
            merged = pd.merge(
                perf_df,
                sil_df,
                on=['dataset', 'model', 'seed'],
                how='outer'
            )
            all_results.append(merged)
            print(f"  ✓ Merged {len(merged)} rows")
        else:
            print(f"  ⚠ Clustering metrics not found")
            all_results.append(perf_df)
    else:
        print(f"  ✗ Performance metrics not found")

if not all_results:
    print("\n❌ No results to combine")
    exit(1)

# Combine all datasets
final_df = pd.concat(all_results, ignore_index=True)

# Define column order
col_order = [
    'dataset', 'model', 'seed',
    'test_hr', 'test_ndcg', 'test_auc',
    'val_hr', 'val_ndcg', 'val_auc',
    'silhouette_k3', 'silhouette_k4', 'silhouette_k5',
    'silhouette_k6', 'silhouette_k7', 'silhouette_k8', 'silhouette_k10',
    'epochs', 'val_epoch'
]
col_order = [c for c in col_order if c in final_df.columns]
final_df = final_df[col_order]

# Create output directory
output_dir = Path('results/comprehensive_results')
output_dir.mkdir(parents=True, exist_ok=True)

# Save complete results
output_file = output_dir / 'all_metrics_combined.csv'
final_df.to_csv(output_file, index=False, float_format='%.6f')
print(f"\n✅ Saved: {output_file}")
print(f"   Total rows: {len(final_df)}")

# Create summary with MEAN values
summary_mean_df = final_df[final_df['seed'] == 'mean'].copy()
summary_mean_file = output_dir / 'summary_mean_metrics.csv'
summary_mean_df.to_csv(summary_mean_file, index=False, float_format='%.6f')
print(f"✅ Saved: {summary_mean_file}")
print(f"   Total rows: {len(summary_mean_df)}")

# Create summary with MEDIAN values
summary_median_df = final_df[final_df['seed'] == 'median'].copy()
summary_median_file = output_dir / 'summary_median_metrics.csv'
summary_median_df.to_csv(summary_median_file, index=False, float_format='%.6f')
print(f"✅ Saved: {summary_median_file}")
print(f"   Total rows: {len(summary_median_df)}")

# Display summary for both mean and median
print("\n" + "="*80)
print("SUMMARY: Mean Test Metrics Across Seeds")
print("="*80)
for _, row in summary_mean_df.iterrows():
    print(f"\n{row['dataset']} | {row['model']}")
    print(f"  HR@5:  {row['test_hr']:.4f}")
    print(f"  NDCG@5: {row['test_ndcg']:.4f}")
    print(f"  AUC:    {row['test_auc']:.4f}")
    if 'silhouette_k5' in row and pd.notna(row['silhouette_k5']):
        print(f"  Silhouette (k=5): {row['silhouette_k5']:.4f}")

print("\n" + "="*80)
print("SUMMARY: Median Test Metrics Across Seeds")
print("="*80)
for _, row in summary_median_df.iterrows():
    print(f"\n{row['dataset']} | {row['model']}")
    print(f"  HR@5:  {row['test_hr']:.4f}")
    print(f"  NDCG@5: {row['test_ndcg']:.4f}")
    print(f"  AUC:    {row['test_auc']:.4f}")
    if 'silhouette_k5' in row and pd.notna(row['silhouette_k5']):
        print(f"  Silhouette (k=5): {row['silhouette_k5']:.4f}")

print("\n" + "="*80)
PYTHON_EOF

echo ""
echo "============================================"
echo "Comprehensive CSV Creation Completed"
echo "End time: $(date)"
echo "============================================"
echo ""

# Show final output
echo "Final results saved to:"
echo "  results/comprehensive_results/all_metrics_combined.csv"
echo "  results/comprehensive_results/summary_mean_metrics.csv"
echo "  results/comprehensive_results/summary_median_metrics.csv"
