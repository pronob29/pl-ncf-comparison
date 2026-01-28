#!/usr/bin/env python3
"""
Compute Silhouette Scores for Clustering Quality Assessment
=============================================================

Computes silhouette scores for spherical K-means clustering of embeddings
with k=5,6,7,8 clusters. Supports both user/item entities and main/pl
representations. Computes both cosine and euclidean silhouette metrics.

Usage:
    # Compute for all entity×repr combinations
    python scripts/utils/compute_silhouette_scores.py --dataset support_groups_full_164

    # Compute for specific entity and representation
    python scripts/utils/compute_silhouette_scores.py --dataset support_groups_full_164 --entity user --repr pl

    # Process all datasets
    python scripts/utils/compute_silhouette_scores.py --all

Output:
    results/{dataset}/clustering/silhouette_{entity}_{repr}.csv
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize


def compute_silhouette_dual(
    embeddings: np.ndarray,
    k: int,
    random_state: int = 42
) -> Tuple[float, float]:
    """
    Compute both cosine and euclidean silhouette scores using spherical K-means.

    Spherical K-means normalizes embeddings before clustering, which is
    appropriate for cosine-based similarity measures (as in PL models).

    Args:
        embeddings: Embedding matrix (num_entities, embedding_dim)
        k: Number of clusters
        random_state: Random seed

    Returns:
        (silhouette_cosine, silhouette_euclidean)
    """
    if len(embeddings) < k:
        print(f"  ⚠️  Not enough samples ({len(embeddings)}) for k={k}")
        return np.nan, np.nan

    # Spherical K-means: L2-normalize before clustering
    embeddings_normalized = normalize(embeddings, norm='l2', axis=1)

    # Apply K-means on normalized embeddings
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(embeddings_normalized)

    # Silhouette score requires at least 2 clusters
    if len(np.unique(labels)) < 2:
        return np.nan, np.nan

    # Compute cosine silhouette on normalized embeddings
    try:
        score_cosine = silhouette_score(
            embeddings_normalized, labels, metric='cosine'
        )
    except Exception as e:
        print(f"  ⚠️  Cosine silhouette failed: {e}")
        score_cosine = np.nan

    # Compute euclidean silhouette on original embeddings
    try:
        score_euclidean = silhouette_score(
            embeddings, labels, metric='euclidean'
        )
    except Exception as e:
        print(f"  ⚠️  Euclidean silhouette failed: {e}")
        score_euclidean = np.nan

    return score_cosine, score_euclidean


def process_single_model(
    model: str,
    dataset: str,
    seed: int,
    entity: str = 'user',
    repr_type: str = 'main',
    k_values: list = [5, 6, 7, 8],
    embeddings_dir: Path = None
) -> Dict:
    """
    Compute silhouette scores for a single model/dataset/seed/entity/repr.

    Args:
        model: Model name
        dataset: Dataset name
        seed: Random seed
        entity: 'user' or 'item'
        repr_type: 'main' or 'pl'
        k_values: List of k values for clustering
        embeddings_dir: Custom embeddings directory

    Returns:
        Dictionary with results: {k: (cosine_score, euclidean_score)}
    """
    if embeddings_dir is None:
        embeddings_dir = Path(f"results/{dataset}/embeddings")

    # Build embedding path based on entity and repr_type
    if repr_type == 'pl':
        emb_path = embeddings_dir / f"{model}_seed{seed}_pl_{entity}_emb.npy"
    else:
        emb_path = embeddings_dir / f"{model}_seed{seed}_{entity}_emb.npy"

    if not emb_path.exists():
        print(f"  ⚠️  Embeddings not found: {emb_path}")
        return {k: (np.nan, np.nan) for k in k_values}

    # Load embeddings
    try:
        embeddings = np.load(emb_path)
        print(f"  Loaded: {emb_path.name} (shape: {embeddings.shape})")
    except Exception as e:
        print(f"  ❌ Error loading {emb_path}: {e}")
        return {k: (np.nan, np.nan) for k in k_values}

    # Compute dual silhouette scores for each k
    results = {}
    for k in k_values:
        score_cosine, score_euclidean = compute_silhouette_dual(
            embeddings, k, random_state=seed
        )
        results[k] = (score_cosine, score_euclidean)

        if not np.isnan(score_cosine):
            print(f"    k={k}: cosine={score_cosine:.4f}, euclidean={score_euclidean:.4f}")
        else:
            print(f"    k={k}: N/A")

    return results


def process_dataset(
    dataset: str,
    entity: str = 'user',
    repr_type: str = 'main',
    models: list = None,
    seeds: list = None,
    k_values: list = None
):
    """
    Compute silhouette scores for all models in a dataset for a specific entity×repr combination.

    Args:
        dataset: Dataset name (e.g., 'support_groups_full_164')
        entity: 'user' or 'item'
        repr_type: 'main' or 'pl'
        models: List of model names
        seeds: List of random seeds
        k_values: List of k values for clustering
    """
    if models is None:
        models = ['mf_baseline', 'mf_pl', 'mlp_baseline', 'mlp_pl',
                  'neumf_baseline', 'neumf_pl']
    if seeds is None:
        seeds = [42, 52, 62, 122, 232]
    if k_values is None:
        k_values = [5, 6, 7, 8]

    print(f"\n{'='*80}")
    print(f"Computing Silhouette Scores: {dataset}")
    print(f"Entity: {entity}, Representation: {repr_type}")
    print(f"{'='*80}\n")

    all_results = []

    for model in models:
        # Skip PL repr for non-PL models
        if repr_type == 'pl' and '_pl' not in model:
            print(f"{model} (seed=all): Skipping (not a PL model)\n")
            continue

        for seed in seeds:
            print(f"{model} (seed={seed}):")

            scores = process_single_model(
                model, dataset, seed, entity, repr_type, k_values
            )

            # Create row for CSV with both cosine and euclidean scores
            row = {
                'dataset': dataset,
                'model': model,
                'seed': seed,
                'entity': entity,
                'repr': repr_type,
            }

            for k in k_values:
                cosine_score, euclidean_score = scores.get(k, (np.nan, np.nan))
                row[f'cosine_k{k}'] = cosine_score
                row[f'euclidean_k{k}'] = euclidean_score

            all_results.append(row)
            print()

    if not all_results:
        print("⚠️  No results generated (no valid models for this repr type)")
        return None

    # Create DataFrame
    df = pd.DataFrame(all_results)

    # Add mean and std across seeds for each model
    print("\nComputing aggregated statistics...")

    # Get unique models that have data
    unique_models = df['model'].unique()

    agg_results = []
    for model in unique_models:
        model_data = df[df['model'] == model]

        # Mean row
        row_mean = {
            'dataset': dataset,
            'model': model,
            'seed': 'mean',
            'entity': entity,
            'repr': repr_type,
        }

        for k in k_values:
            row_mean[f'cosine_k{k}'] = model_data[f'cosine_k{k}'].mean()
            row_mean[f'euclidean_k{k}'] = model_data[f'euclidean_k{k}'].mean()

        agg_results.append(row_mean)

        # Std row
        row_std = {
            'dataset': dataset,
            'model': model,
            'seed': 'std',
            'entity': entity,
            'repr': repr_type,
        }

        for k in k_values:
            row_std[f'cosine_k{k}'] = model_data[f'cosine_k{k}'].std()
            row_std[f'euclidean_k{k}'] = model_data[f'euclidean_k{k}'].std()

        agg_results.append(row_std)

    # Append aggregated results
    df_agg = pd.DataFrame(agg_results)
    df_combined = pd.concat([df, df_agg], ignore_index=True)

    # Save results
    output_dir = Path(f"results/{dataset}/clustering")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"silhouette_{entity}_{repr_type}.csv"
    df_combined.to_csv(output_file, index=False, float_format='%.4f')

    print(f"\n✅ Saved: {output_file}")
    print(f"   Total rows: {len(df_combined)}")

    # Print summary
    print("\n" + "="*80)
    print(f"SUMMARY: Mean Silhouette Scores ({entity}, {repr_type})")
    print("="*80)
    print(df_agg[df_agg['seed'] == 'mean'].to_string(index=False))
    print()

    return df_combined


def process_all_datasets(
    datasets: list = None,
    entities: list = None,
    repr_types: list = None,
    models: list = None,
    seeds: list = None,
    k_values: list = None
):
    """Process all datasets for all entity×repr combinations."""
    if datasets is None:
        datasets = [
            'support_groups_full_164',
            'support_groups_full_164_loo'
        ]
    if entities is None:
        entities = ['user', 'item']
    if repr_types is None:
        repr_types = ['main', 'pl']

    for dataset in datasets:
        for entity in entities:
            for repr_type in repr_types:
                process_dataset(
                    dataset, entity, repr_type, models, seeds, k_values
                )


def main():
    parser = argparse.ArgumentParser(
        description="Compute silhouette scores for clustering quality"
    )
    parser.add_argument(
        '--dataset',
        type=str,
        help='Dataset name (e.g., support_groups_full_164)'
    )
    parser.add_argument(
        '--entity',
        type=str,
        default=None,
        choices=['user', 'item'],
        help='Entity type (default: process both user and item)'
    )
    parser.add_argument(
        '--repr',
        type=str,
        default=None,
        choices=['main', 'pl'],
        dest='repr_type',
        help='Representation type (default: process both main and pl)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Process all datasets and all entity×repr combinations'
    )
    parser.add_argument(
        '--k-values',
        type=int,
        nargs='+',
        default=[5, 6, 7, 8],
        help='K values for clustering (default: 5 6 7 8)'
    )

    args = parser.parse_args()

    if args.all:
        # Process all datasets and all combinations
        process_all_datasets(k_values=args.k_values)
    elif args.dataset:
        # Determine which entity×repr combinations to process
        entities = [args.entity] if args.entity else ['user', 'item']
        repr_types = [args.repr_type] if args.repr_type else ['main', 'pl']

        for entity in entities:
            for repr_type in repr_types:
                process_dataset(
                    args.dataset,
                    entity=entity,
                    repr_type=repr_type,
                    k_values=args.k_values
                )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
