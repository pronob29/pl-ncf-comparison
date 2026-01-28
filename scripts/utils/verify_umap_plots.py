#!/usr/bin/env python3
"""
Verify UMAP Plots and Metrics
==============================

Checks that all expected UMAP plots and metrics JSON files exist,
and validates that they have the correct representation type and
clustering parameters.

Usage:
    python scripts/utils/verify_umap_plots.py [--dataset DATASET] [--strict]

Example:
    python scripts/utils/verify_umap_plots.py
    python scripts/utils/verify_umap_plots.py --dataset support_groups_full_164
    python scripts/utils/verify_umap_plots.py --strict  # Also verify k matches silhouette optimal
"""

import argparse
import csv
import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

# Valid k values from silhouette analysis
VALID_K_VALUES = {3, 4, 5, 6, 7, 8, 10}


@lru_cache(maxsize=None)
def _load_silhouette_csv(csv_path: str) -> Dict[tuple, Dict[int, float]]:
    """
    Load and cache silhouette CSV data.

    Returns:
        Dict mapping (model, seed, entity, repr) -> {k: cosine_score}
    """
    path = Path(csv_path)
    if not path.exists():
        return {}

    result = {}
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip summary rows (mean, std)
            try:
                seed = int(row['seed'])
            except (ValueError, TypeError):
                continue

            key = (row['model'], seed, row['entity'], row['repr'])
            scores = {}
            for k in VALID_K_VALUES:
                col_name = f'cosine_k{k}'
                if col_name in row and row[col_name]:
                    try:
                        score = float(row[col_name])
                        if score == score:  # NaN check
                            scores[k] = score
                    except (ValueError, TypeError):
                        pass
            if scores:
                result[key] = scores

    return result


def get_expected_optimal_k(
    dataset: str,
    model: str,
    seed: int,
    entity: str,
    repr_type: str,
    default_k: int = 5
) -> int:
    """
    Compute the expected optimal k using the same logic as get_optimal_k.py.

    Args:
        dataset: Dataset name
        model: Model name
        seed: Random seed
        entity: 'user' or 'item'
        repr_type: 'main' or 'pl'
        default_k: Fallback k value

    Returns:
        Expected optimal k
    """
    csv_path = f"results/{dataset}/clustering/silhouette_{entity}_{repr_type}.csv"
    cache = _load_silhouette_csv(csv_path)

    key = (model, seed, entity, repr_type)
    scores = cache.get(key, {})

    if not scores:
        return default_k

    # Find k with max score, tie-break with smallest k
    max_score = max(scores.values())
    candidates = [k for k, s in scores.items() if s == max_score]
    return min(candidates)


def verify_umap_files(
    dataset: str,
    verbose: bool = True,
    strict: bool = False
) -> Tuple[int, int, List[str]]:
    """
    Verify that expected UMAP files exist and are valid.

    Args:
        dataset: Dataset name (e.g., 'support_groups_full_164')
        verbose: Print detailed information
        strict: If True, also verify n_clusters matches silhouette-optimal k

    Returns:
        (total_expected, total_found, list_of_errors)
    """
    models = ['mf_baseline', 'mf_pl', 'mlp_baseline', 'mlp_pl', 'neumf_baseline', 'neumf_pl']
    seeds = [42, 52, 62, 122, 232]
    entities = ['user', 'item']

    umap_dir = Path(f"results/{dataset}/umap_plots")
    errors = []
    total_expected = 0
    total_found = 0

    if verbose:
        print(f"\n{'='*80}")
        print(f"Verifying UMAP plots for: {dataset}")
        if strict:
            print("  (strict mode: verifying n_clusters matches silhouette-optimal k)")
        print(f"{'='*80}\n")

    # Check main representation files (all models)
    for model in models:
        for seed in seeds:
            for entity in entities:
                total_expected += 1

                png_file = umap_dir / f"{model}_seed{seed}_{entity}_main_umap.png"
                json_file = umap_dir / f"{model}_seed{seed}_{entity}_main_metrics.json"

                # Check PNG exists
                if not png_file.exists():
                    errors.append(f"Missing PNG: {png_file.name}")
                    continue

                # Check JSON exists
                if not json_file.exists():
                    errors.append(f"Missing JSON: {json_file.name}")
                    continue

                # Validate JSON content
                try:
                    with open(json_file, 'r') as f:
                        metrics = json.load(f)

                    # Verify repr field
                    if metrics.get('repr') != 'main':
                        errors.append(f"Invalid repr in {json_file.name}: expected 'main', got '{metrics.get('repr')}'")

                    # Verify n_clusters is in valid range {3,4,5,6,7,8,10}
                    n_clusters = metrics.get('clustering', {}).get('n_clusters')
                    if n_clusters not in VALID_K_VALUES:
                        errors.append(f"Invalid n_clusters in {json_file.name}: expected one of {sorted(VALID_K_VALUES)}, got {n_clusters}")

                    # Strict mode: verify n_clusters matches silhouette-optimal k
                    if strict and n_clusters in VALID_K_VALUES:
                        expected_k = get_expected_optimal_k(dataset, model, seed, entity, 'main')
                        if n_clusters != expected_k:
                            errors.append(f"n_clusters mismatch in {json_file.name}: expected {expected_k} (silhouette-optimal), got {n_clusters}")

                    total_found += 1

                except Exception as e:
                    errors.append(f"Error reading {json_file.name}: {e}")

    # Check PL representation files (only for *_pl models)
    pl_models = [m for m in models if m.endswith('_pl')]

    for model in pl_models:
        for seed in seeds:
            for entity in entities:
                total_expected += 1

                png_file = umap_dir / f"{model}_seed{seed}_{entity}_pl_umap.png"
                json_file = umap_dir / f"{model}_seed{seed}_{entity}_pl_metrics.json"

                # Check PNG exists
                if not png_file.exists():
                    errors.append(f"Missing PNG: {png_file.name}")
                    continue

                # Check JSON exists
                if not json_file.exists():
                    errors.append(f"Missing JSON: {json_file.name}")
                    continue

                # Validate JSON content
                try:
                    with open(json_file, 'r') as f:
                        metrics = json.load(f)

                    # Verify repr field
                    if metrics.get('repr') != 'pl':
                        errors.append(f"Invalid repr in {json_file.name}: expected 'pl', got '{metrics.get('repr')}'")

                    # Verify n_clusters is in valid range {3,4,5,6,7,8,10}
                    n_clusters = metrics.get('clustering', {}).get('n_clusters')
                    if n_clusters not in VALID_K_VALUES:
                        errors.append(f"Invalid n_clusters in {json_file.name}: expected one of {sorted(VALID_K_VALUES)}, got {n_clusters}")

                    # Strict mode: verify n_clusters matches silhouette-optimal k
                    if strict and n_clusters in VALID_K_VALUES:
                        expected_k = get_expected_optimal_k(dataset, model, seed, entity, 'pl')
                        if n_clusters != expected_k:
                            errors.append(f"n_clusters mismatch in {json_file.name}: expected {expected_k} (silhouette-optimal), got {n_clusters}")

                    # Verify embedding shape is smaller than main (PL embeddings are lower-dimensional)
                    emb_shape = metrics.get('embedding_shape')
                    if emb_shape and len(emb_shape) == 2:
                        pl_dim = emb_shape[1]
                        # PL embeddings should typically be 32-dim (vs 64-96 for main)
                        if pl_dim > 64:
                            errors.append(f"Suspicious PL embedding dim in {json_file.name}: {pl_dim} (expected < 64)")

                    total_found += 1

                except Exception as e:
                    errors.append(f"Error reading {json_file.name}: {e}")

    # Print summary
    if verbose:
        print(f"Expected files: {total_expected * 2} ({total_expected} PNG + {total_expected} JSON)")
        print(f"Found valid: {total_found * 2} ({total_found} PNG + {total_found} JSON)")
        print(f"Errors: {len(errors)}")

        if errors:
            print(f"\n{'='*80}")
            print("ERRORS:")
            print(f"{'='*80}")
            for error in errors:
                print(f"  ❌ {error}")
        else:
            print(f"\n✅ All UMAP plots validated successfully!")

        print(f"\n{'='*80}\n")

    return total_expected, total_found, errors


def main():
    parser = argparse.ArgumentParser(description="Verify UMAP plots and metrics")
    parser.add_argument('--dataset', type=str,
                        help='Dataset name (default: check both datasets)')
    parser.add_argument('--quiet', action='store_true',
                        help='Only print summary')
    parser.add_argument('--strict', action='store_true',
                        help='Also verify n_clusters matches silhouette-optimal k')

    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else ['support_groups_full_164', 'support_groups_full_164_loo']

    total_expected_all = 0
    total_found_all = 0
    all_errors = []

    for dataset in datasets:
        total_expected, total_found, errors = verify_umap_files(
            dataset, verbose=not args.quiet, strict=args.strict
        )
        total_expected_all += total_expected
        total_found_all += total_found
        all_errors.extend(errors)

    # Overall summary
    if len(datasets) > 1:
        print(f"\n{'='*80}")
        print("OVERALL SUMMARY")
        print(f"{'='*80}")
        print(f"Total expected files: {total_expected_all * 2}")
        print(f"Total found valid: {total_found_all * 2}")
        print(f"Total errors: {len(all_errors)}")

        if all_errors:
            print(f"\n❌ Validation FAILED with {len(all_errors)} errors")
            sys.exit(1)
        else:
            print(f"\n✅ All UMAP plots validated successfully across all datasets!")
            sys.exit(0)
    else:
        sys.exit(0 if not all_errors else 1)


if __name__ == '__main__':
    main()
