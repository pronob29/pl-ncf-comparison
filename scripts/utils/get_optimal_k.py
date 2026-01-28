#!/usr/bin/env python3
"""
Get Optimal K for UMAP Clustering from Silhouette Analysis
===========================================================

Reads silhouette CSV files and returns the optimal number of clusters (k)
for a given model/seed/entity/repr combination based on maximum cosine silhouette score.

Usage:
    python scripts/utils/get_optimal_k.py <dataset> <model> <seed> <entity> [--repr {main,pl}]

Examples:
    # Get optimal k for PL embeddings (default)
    python scripts/utils/get_optimal_k.py support_groups_full_164 mf_pl 42 user
    # Output: 4

    # Get optimal k for main embeddings
    python scripts/utils/get_optimal_k.py support_groups_full_164 mf_baseline 42 user --repr main
    # Output: 3

Returns:
    - The optimal k value (int) on stdout
    - Defaults to 5 if data is missing or on error
    - Logs warnings/info to stderr
"""

import argparse
import csv
import sys
from pathlib import Path


def get_optimal_k(
    dataset: str,
    model: str,
    seed: int,
    entity: str,
    repr_type: str = 'pl',
    k_values=None,
    default_k: int = 5
) -> int:
    """
    Get optimal k from silhouette analysis CSV.

    Args:
        dataset: Dataset name (e.g., 'support_groups_full_164')
        model: Model name (e.g., 'mf_pl', 'mf_baseline')
        seed: Random seed
        entity: 'user' or 'item'
        repr_type: 'main' or 'pl' (default: 'pl')
        k_values: List of k values to consider (default: [3, 4, 5, 6, 7, 8, 10])
        default_k: Default k if lookup fails (default: 5)

    Returns:
        Optimal k value (int)
    """
    if k_values is None:
        k_values = [3, 4, 5, 6, 7, 8, 10]

    # Build path to silhouette CSV based on repr_type
    csv_path = Path(f"results/{dataset}/clustering/silhouette_{entity}_{repr_type}.csv")

    if not csv_path.exists():
        print(f"Warning: Silhouette CSV not found: {csv_path}", file=sys.stderr)
        print(f"  Context: {dataset}/{model}/seed{seed}/{entity}/{repr_type}", file=sys.stderr)
        print(f"  Using default k={default_k}", file=sys.stderr)
        return default_k

    try:
        # Read CSV
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Find the matching row, skipping non-integer seeds (e.g., "mean", "std")
        matching_row = None
        for row in rows:
            # Skip rows where seed is not an integer (e.g., "mean", "std" summary rows)
            try:
                row_seed = int(row['seed'])
            except (ValueError, TypeError):
                continue

            if (row['dataset'] == dataset and
                row['model'] == model and
                row_seed == seed and
                row['entity'] == entity and
                row['repr'] == repr_type):
                matching_row = row
                break

        if matching_row is None:
            print(f"Warning: No data found for {dataset}/{model}/seed{seed}/{entity}/{repr_type}", file=sys.stderr)
            print(f"  Using default k={default_k}", file=sys.stderr)
            return default_k

        # Extract cosine silhouette scores for each k
        scores = {}
        for k in k_values:
            col_name = f'cosine_k{k}'
            if col_name in matching_row and matching_row[col_name]:
                try:
                    score = float(matching_row[col_name])
                    # Ignore NaN or invalid scores
                    if score == score:  # NaN check: NaN != NaN
                        scores[k] = score
                except (ValueError, TypeError):
                    pass

        if not scores:
            print(f"Warning: No valid cosine silhouette scores found", file=sys.stderr)
            print(f"  Context: {dataset}/{model}/seed{seed}/{entity}/{repr_type}", file=sys.stderr)
            print(f"  Using default k={default_k}", file=sys.stderr)
            return default_k

        # Find k with maximum cosine silhouette
        # Tie-break: choose smallest k if multiple have same max score
        max_score = max(scores.values())
        candidates = [k for k, s in scores.items() if s == max_score]
        optimal_k = min(candidates)  # smallest k wins ties
        optimal_score = scores[optimal_k]

        print(f"Optimal k={optimal_k} (cosine silhouette={optimal_score:.4f})", file=sys.stderr)
        return optimal_k

    except Exception as e:
        print(f"Error reading silhouette CSV: {e}", file=sys.stderr)
        print(f"  Context: {dataset}/{model}/seed{seed}/{entity}/{repr_type}", file=sys.stderr)
        print(f"  Using default k={default_k}", file=sys.stderr)
        return default_k


def main():
    parser = argparse.ArgumentParser(
        description="Get optimal k for UMAP clustering from silhouette analysis"
    )
    parser.add_argument('dataset', type=str, help='Dataset name')
    parser.add_argument('model', type=str, help='Model name (e.g., mf_pl, mf_baseline)')
    parser.add_argument('seed', type=int, help='Random seed')
    parser.add_argument('entity', type=str, choices=['user', 'item'], help='Entity type')
    parser.add_argument('--repr', type=str, choices=['main', 'pl'], default='pl',
                        dest='repr_type',
                        help='Representation type: main or pl (default: pl)')
    parser.add_argument('--default-k', type=int, default=5, help='Default k value (default: 5)')

    args = parser.parse_args()

    optimal_k = get_optimal_k(
        args.dataset,
        args.model,
        args.seed,
        args.entity,
        repr_type=args.repr_type,
        default_k=args.default_k
    )

    # Print only the k value to stdout (for shell script consumption)
    print(optimal_k)
    return 0


if __name__ == '__main__':
    sys.exit(main())
