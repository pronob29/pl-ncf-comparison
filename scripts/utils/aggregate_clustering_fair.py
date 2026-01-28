#!/usr/bin/env python3
"""
Fair Clustering Metrics Aggregation
====================================

Aggregates silhouette scores with methodologically fair k-selection protocols:
1. Fixed-k (k=5): All models/spaces evaluated at k=5 for strict comparability
2. Optimal-k: All models/spaces evaluated at their per-seed optimal k from grid

This ensures baseline vs PL comparisons are fair - both get the same k-selection rule.

Usage:
    python scripts/utils/aggregate_clustering_fair.py

Output:
    results/comprehensive_results/clustering_summary_fixed_k5.csv
    results/comprehensive_results/clustering_summary_optimal_k.csv
"""

import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

K_GRID = [3, 4, 5, 6, 7, 8, 10]


def read_silhouette_csv(csv_path: Path) -> List[Dict]:
    """Read silhouette CSV and return list of row dicts."""
    if not csv_path.exists():
        print(f"⚠️  Warning: {csv_path} not found")
        return []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)


def get_optimal_k_for_row(row: Dict, k_grid: List[int] = K_GRID) -> Tuple[int, float]:
    """
    Find optimal k (argmax cosine silhouette) for a single row.

    Returns:
        (optimal_k, optimal_score)
    """
    scores = {}
    for k in k_grid:
        col_name = f'cosine_k{k}'
        if col_name in row and row[col_name]:
            try:
                score = float(row[col_name])
                scores[k] = score
            except ValueError:
                pass

    if not scores:
        # Default to k=5 if no valid scores
        return 5, float('nan')

    optimal_k = max(scores, key=scores.get)
    optimal_score = scores[optimal_k]
    return optimal_k, optimal_score


def aggregate_dataset_entity_repr(
    dataset: str,
    entity: str,
    repr_type: str,
    clustering_dir: Path
) -> Tuple[List[Dict], List[Dict]]:
    """
    Aggregate clustering metrics for one dataset/entity/repr combination.

    Returns:
        (fixed_k_rows, optimal_k_rows)
    """
    csv_path = clustering_dir / f"silhouette_{entity}_{repr_type}.csv"
    rows = read_silhouette_csv(csv_path)

    if not rows:
        return [], []

    fixed_k_results = []
    optimal_k_results = []

    # Process each model/seed combination
    for row in rows:
        seed = row.get('seed', '')
        model = row.get('model', '')

        # Skip aggregated rows (mean/std)
        if seed in ['mean', 'std']:
            continue

        try:
            seed_int = int(seed)
        except ValueError:
            continue

        # Fixed-k (k=5) result
        cosine_k5 = float(row.get('cosine_k5', 'nan'))
        euclidean_k5 = float(row.get('euclidean_k5', 'nan'))

        fixed_k_results.append({
            'dataset': dataset,
            'model': model,
            'seed': seed_int,
            'entity': entity,
            'repr': repr_type,
            'k': 5,
            'cosine_silhouette': cosine_k5,
            'euclidean_silhouette': euclidean_k5,
            'protocol': 'fixed_k5'
        })

        # Optimal-k result
        optimal_k, optimal_score = get_optimal_k_for_row(row, K_GRID)

        # Get euclidean score for the same optimal k
        euclidean_col = f'euclidean_k{optimal_k}'
        euclidean_score = float(row.get(euclidean_col, 'nan'))

        optimal_k_results.append({
            'dataset': dataset,
            'model': model,
            'seed': seed_int,
            'entity': entity,
            'repr': repr_type,
            'k': optimal_k,
            'cosine_silhouette': optimal_score,
            'euclidean_silhouette': euclidean_score,
            'protocol': 'optimal_k'
        })

    return fixed_k_results, optimal_k_results


def compute_summary_stats(results: List[Dict]) -> List[Dict]:
    """
    Compute mean and std across seeds for each model/dataset/entity/repr.
    """
    # Group by (dataset, model, entity, repr)
    groups = {}
    for row in results:
        key = (row['dataset'], row['model'], row['entity'], row['repr'])
        if key not in groups:
            groups[key] = []
        groups[key].append(row)

    summary_rows = []
    for key, group_rows in groups.items():
        dataset, model, entity, repr_type = key
        protocol = group_rows[0]['protocol']

        # Compute mean
        cosine_scores = [r['cosine_silhouette'] for r in group_rows]
        euclidean_scores = [r['euclidean_silhouette'] for r in group_rows]
        k_values = [r['k'] for r in group_rows]

        # Filter out NaN values for stats
        cosine_valid = [s for s in cosine_scores if s == s]  # NaN != NaN
        euclidean_valid = [s for s in euclidean_scores if s == s]

        if cosine_valid:
            cosine_mean = sum(cosine_valid) / len(cosine_valid)
            cosine_std = (sum((s - cosine_mean)**2 for s in cosine_valid) / len(cosine_valid))**0.5 if len(cosine_valid) > 1 else 0.0
        else:
            cosine_mean = float('nan')
            cosine_std = float('nan')

        if euclidean_valid:
            euclidean_mean = sum(euclidean_valid) / len(euclidean_valid)
            euclidean_std = (sum((s - euclidean_mean)**2 for s in euclidean_valid) / len(euclidean_valid))**0.5 if len(euclidean_valid) > 1 else 0.0
        else:
            euclidean_mean = float('nan')
            euclidean_std = float('nan')

        # For optimal_k protocol, k varies by seed - report mode or mean
        if protocol == 'optimal_k':
            # Count frequency of each k
            k_freq = {}
            for k in k_values:
                k_freq[k] = k_freq.get(k, 0) + 1
            k_mode = max(k_freq, key=k_freq.get)
            k_display = f"{k_mode}"  # Mode (most common)
        else:
            k_display = "5"

        summary_rows.append({
            'dataset': dataset,
            'model': model,
            'seed': 'mean',
            'entity': entity,
            'repr': repr_type,
            'k': k_display,
            'cosine_silhouette': cosine_mean,
            'euclidean_silhouette': euclidean_mean,
            'protocol': protocol
        })

        summary_rows.append({
            'dataset': dataset,
            'model': model,
            'seed': 'std',
            'entity': entity,
            'repr': repr_type,
            'k': k_display,
            'cosine_silhouette': cosine_std,
            'euclidean_silhouette': euclidean_std,
            'protocol': protocol
        })

    return summary_rows


def main():
    print("="*80)
    print("Fair Clustering Metrics Aggregation")
    print("="*80)
    print()

    datasets = ['support_groups_full_164', 'support_groups_full_164_loo']
    entities = ['user', 'item']
    repr_types = ['main', 'pl']

    all_fixed_k = []
    all_optimal_k = []

    for dataset in datasets:
        print(f"Processing dataset: {dataset}")
        clustering_dir = Path(f"results/{dataset}/clustering")

        for entity in entities:
            for repr_type in repr_types:
                print(f"  {entity}/{repr_type}...", end=' ')

                fixed_k_rows, optimal_k_rows = aggregate_dataset_entity_repr(
                    dataset, entity, repr_type, clustering_dir
                )

                all_fixed_k.extend(fixed_k_rows)
                all_optimal_k.extend(optimal_k_rows)

                print(f"✓ ({len(fixed_k_rows)} seeds)")
        print()

    # Compute summary statistics
    print("Computing summary statistics...")
    fixed_k_summary = compute_summary_stats(all_fixed_k)
    optimal_k_summary = compute_summary_stats(all_optimal_k)

    # Combine per-seed and summary rows
    all_fixed_k_combined = all_fixed_k + fixed_k_summary
    all_optimal_k_combined = all_optimal_k + optimal_k_summary

    # Save results
    output_dir = Path("results/comprehensive_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fixed-k output
    fixed_k_path = output_dir / "clustering_summary_fixed_k5.csv"
    with open(fixed_k_path, 'w', newline='') as f:
        fieldnames = ['dataset', 'model', 'seed', 'entity', 'repr', 'k',
                      'cosine_silhouette', 'euclidean_silhouette', 'protocol']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # Sort by dataset, model, entity, repr, seed
        sorted_rows = sorted(all_fixed_k_combined, key=lambda r: (
            r['dataset'],
            r['model'],
            r['entity'],
            r['repr'],
            str(r['seed']) if isinstance(r['seed'], int) else ('z' + r['seed'])  # mean/std go last
        ))

        for row in sorted_rows:
            writer.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in row.items()})

    print(f"✅ Saved: {fixed_k_path}")
    print(f"   Rows: {len(all_fixed_k_combined)} ({len(all_fixed_k)} per-seed + {len(fixed_k_summary)} summary)")

    # Optimal-k output
    optimal_k_path = output_dir / "clustering_summary_optimal_k.csv"
    with open(optimal_k_path, 'w', newline='') as f:
        fieldnames = ['dataset', 'model', 'seed', 'entity', 'repr', 'k',
                      'cosine_silhouette', 'euclidean_silhouette', 'protocol']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        sorted_rows = sorted(all_optimal_k_combined, key=lambda r: (
            r['dataset'],
            r['model'],
            r['entity'],
            r['repr'],
            str(r['seed']) if isinstance(r['seed'], int) else ('z' + r['seed'])
        ))

        for row in sorted_rows:
            writer.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in row.items()})

    print(f"✅ Saved: {optimal_k_path}")
    print(f"   Rows: {len(all_optimal_k_combined)} ({len(all_optimal_k)} per-seed + {len(optimal_k_summary)} summary)")

    # Print comparison summary
    print()
    print("="*80)
    print("SUMMARY: Mean Cosine Silhouette Scores")
    print("="*80)
    print()

    # Get mean rows for each protocol
    fixed_means = [r for r in fixed_k_summary if r['seed'] == 'mean']
    optimal_means = [r for r in optimal_k_summary if r['seed'] == 'mean']

    print("FIXED-K (k=5) PROTOCOL:")
    print("-" * 80)
    print(f"{'Dataset':<30} {'Model':<15} {'Entity':<6} {'Repr':<6} {'Cosine':<10}")
    print("-" * 80)
    for row in fixed_means:
        print(f"{row['dataset']:<30} {row['model']:<15} {row['entity']:<6} {row['repr']:<6} {row['cosine_silhouette']:>10.6f}")

    print()
    print("OPTIMAL-K PROTOCOL:")
    print("-" * 80)
    print(f"{'Dataset':<30} {'Model':<15} {'Entity':<6} {'Repr':<6} {'K':<5} {'Cosine':<10}")
    print("-" * 80)
    for row in optimal_means:
        print(f"{row['dataset']:<30} {row['model']:<15} {row['entity']:<6} {row['repr']:<6} {row['k']:<5} {row['cosine_silhouette']:>10.6f}")

    print()
    print("="*80)
    print("✅ Aggregation complete!")
    print()
    print("Key findings:")
    print("  - Fixed-k (k=5): All models evaluated at same k for strict comparability")
    print("  - Optimal-k: All models evaluated at their per-seed optimal k from grid")
    print("  - Use optimal-k vs optimal-k for fair baseline vs PL comparisons")
    print("="*80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
