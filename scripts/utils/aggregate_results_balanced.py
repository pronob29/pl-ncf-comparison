#!/usr/bin/env python3
"""
Aggregate Results for Balanced Experiments
===========================================

Aggregates training results (HR, NDCG, AUC) from all model/dataset/seed
combinations into comprehensive CSV files for analysis.

Usage:
    python scripts/aggregate_results_balanced.py --dataset support_groups_v7_bal50
    python scripts/aggregate_results_balanced.py --all  # All datasets

Output:
    results/{dataset}/performance_metrics.csv  # Median ± std across seeds
    results/{dataset}/performance_metrics_all_seeds.csv  # Individual seed results
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_results_file(results_file: Path) -> dict:
    """Load results from JSON file."""
    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"  ⚠️  Error loading {results_file.name}: {e}")
        return None


def aggregate_dataset_results(
    dataset: str,
    models: list = None,
    seeds: list = None
):
    """
    Aggregate results for a single dataset.

    Args:
        dataset: Dataset name
        models: List of model names
        seeds: List of random seeds
    """
    if models is None:
        models = ['mf_baseline', 'mf_pl', 'mlp_baseline', 'mlp_pl', 'neumf_baseline', 'neumf_pl']
    if seeds is None:
        seeds = [42, 52, 62, 122, 232]

    print(f"\n{'='*80}")
    print(f"Aggregating Results: {dataset}")
    print(f"{'='*80}\n")

    results_dir = Path(f"results/{dataset}/seeds")

    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return None

    all_results = []

    # Load all results files
    for model in models:
        for seed in seeds:
            results_file = results_dir / f"{model}_{dataset}_seed{seed}.json"

            if not results_file.exists():
                print(f"  ⚠️  Missing: {results_file.name}")
                continue

            data = load_results_file(results_file)
            if data is None:
                continue

            # Extract key metrics
            row = {
                'dataset': dataset,
                'model': model,
                'seed': seed,
                'val_hr': data.get('val_hr', np.nan),
                'val_ndcg': data.get('val_ndcg', np.nan),
                'val_auc': data.get('val_auc', np.nan),
                'test_hr': data.get('test_hr', np.nan),
                'test_ndcg': data.get('test_ndcg', np.nan),
                'test_auc': data.get('test_auc', np.nan),
                'epochs': data.get('epochs', np.nan),
                'val_epoch': data.get('val_epoch', np.nan),
            }

            all_results.append(row)
            print(f"  ✅ Loaded: {model} seed={seed} | Test HR={row['test_hr']:.4f}, NDCG={row['test_ndcg']:.4f}, AUC={row['test_auc']:.4f}")

    if not all_results:
        print(f"\n❌ No results found for {dataset}")
        return None

    # Create DataFrame
    df_all = pd.DataFrame(all_results)

    # Save all seeds results
    output_all = Path(f"results/{dataset}/performance_metrics_all_seeds.csv")
    output_all.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(output_all, index=False, float_format='%.4f')
    print(f"\n✅ Saved all seeds: {output_all}")

    # Compute aggregated statistics (median ± std and mean ± std)
    print("\nComputing aggregated statistics (median and mean ± std across seeds)...")

    agg_results = []

    for model in models:
        model_data = df_all[df_all['model'] == model]

        if len(model_data) == 0:
            print(f"  ⚠️  No data for {model}")
            continue

        # Median row (robust to outliers)
        row_median = {
            'dataset': dataset,
            'model': model,
            'seed': 'median',
            'val_hr': model_data['val_hr'].median(),
            'val_ndcg': model_data['val_ndcg'].median(),
            'val_auc': model_data['val_auc'].median(),
            'test_hr': model_data['test_hr'].median(),
            'test_ndcg': model_data['test_ndcg'].median(),
            'test_auc': model_data['test_auc'].median(),
            'epochs': model_data['epochs'].median(),
            'val_epoch': model_data['val_epoch'].median(),
        }
        agg_results.append(row_median)

        # Mean row (for comparison with plots)
        row_mean = {
            'dataset': dataset,
            'model': model,
            'seed': 'mean',
            'val_hr': model_data['val_hr'].mean(),
            'val_ndcg': model_data['val_ndcg'].mean(),
            'val_auc': model_data['val_auc'].mean(),
            'test_hr': model_data['test_hr'].mean(),
            'test_ndcg': model_data['test_ndcg'].mean(),
            'test_auc': model_data['test_auc'].mean(),
            'epochs': model_data['epochs'].mean(),
            'val_epoch': model_data['val_epoch'].mean(),
        }
        agg_results.append(row_mean)

        # Std row
        row_std = {
            'dataset': dataset,
            'model': model,
            'seed': 'std',
            'val_hr': model_data['val_hr'].std(),
            'val_ndcg': model_data['val_ndcg'].std(),
            'val_auc': model_data['val_auc'].std(),
            'test_hr': model_data['test_hr'].std(),
            'test_ndcg': model_data['test_ndcg'].std(),
            'test_auc': model_data['test_auc'].std(),
            'epochs': 0,
            'val_epoch': model_data['val_epoch'].std(),
        }
        agg_results.append(row_std)

    # Create aggregated DataFrame
    df_agg = pd.DataFrame(agg_results)

    # Combine all and aggregated
    df_combined = pd.concat([df_all, df_agg], ignore_index=True)

    # Save aggregated results
    output_agg = Path(f"results/{dataset}/performance_metrics.csv")
    df_combined.to_csv(output_agg, index=False, float_format='%.4f')
    print(f"✅ Saved aggregated: {output_agg}")

    # Print summary table
    print("\n" + "="*80)
    print(f"SUMMARY: Test Set Performance (Median ± Std)")
    print("="*80)

    summary_data = []
    for model in models:
        median_row = df_agg[(df_agg['model'] == model) & (df_agg['seed'] == 'median')]
        std_row = df_agg[(df_agg['model'] == model) & (df_agg['seed'] == 'std')]

        if len(median_row) == 0 or len(std_row) == 0:
            continue

        summary_data.append({
            'Model': model,
            'HR@5': f"{median_row['test_hr'].values[0]:.4f} ± {std_row['test_hr'].values[0]:.4f}",
            'NDCG@5': f"{median_row['test_ndcg'].values[0]:.4f} ± {std_row['test_ndcg'].values[0]:.4f}",
            'AUC': f"{median_row['test_auc'].values[0]:.4f} ± {std_row['test_auc'].values[0]:.4f}",
        })

    df_summary = pd.DataFrame(summary_data)
    print(df_summary.to_string(index=False))
    print()

    return df_combined


def aggregate_all_datasets(
    datasets: list = None,
    models: list = None,
    seeds: list = None
):
    """Aggregate results for all datasets."""
    if datasets is None:
        datasets = [
            'support_groups_full_164',
            'support_groups_full_164_loo'
        ]

    all_dfs = []

    for dataset in datasets:
        df = aggregate_dataset_results(dataset, models, seeds)
        if df is not None:
            all_dfs.append(df)

    if not all_dfs:
        print("\n❌ No results to aggregate")
        return

    # Create combined summary across all datasets
    print("\n" + "="*80)
    print("CROSS-DATASET SUMMARY: Test Set Performance")
    print("="*80 + "\n")

    for dataset in datasets:
        df = [d for d in all_dfs if d['dataset'].iloc[0] == dataset]
        if not df:
            continue

        df = df[0]
        df_median = df[df['seed'] == 'median']

        print(f"\n{dataset}:")
        print("-" * 80)

        for _, row in df_median.iterrows():
            print(f"  {row['model']:<20s} | HR={row['test_hr']:.4f}, NDCG={row['test_ndcg']:.4f}, AUC={row['test_auc']:.4f}")

    # Save combined results
    df_combined = pd.concat(all_dfs, ignore_index=True)
    output_combined = Path("results/balanced_experiments_combined.csv")
    output_combined.parent.mkdir(parents=True, exist_ok=True)
    df_combined.to_csv(output_combined, index=False, float_format='%.4f')

    print(f"\n✅ Saved combined results: {output_combined}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate training results for balanced experiments"
    )
    parser.add_argument(
        '--dataset',
        type=str,
        help='Dataset name (e.g., support_groups_v7_bal50)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Aggregate all balanced datasets'
    )

    args = parser.parse_args()

    if args.all:
        aggregate_all_datasets()
    elif args.dataset:
        aggregate_dataset_results(args.dataset)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
