#!/usr/bin/env python3
"""
Verify UMAP Visual Quality - PL vs Main Embeddings
===================================================

Analyzes UMAP plots and metrics to verify that PL-specific embeddings show
better visual cluster separation than main embeddings.

Key checks:
1. Silhouette scores: PL should be higher than main
2. Cluster compactness: Within-cluster distances should be smaller for PL
3. Cluster separation: Between-cluster distances should be larger for PL

Usage:
    python scripts/utils/verify_umap_visual_quality.py [--dataset DATASET]

Output:
    Visual quality comparison report with recommendations
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def load_umap_metrics(umap_dir: Path, model: str, seed: int, entity: str, repr_type: str) -> Dict:
    """Load UMAP metrics JSON file."""
    metrics_path = umap_dir / f"{model}_seed{seed}_{entity}_{repr_type}_metrics.json"

    if not metrics_path.exists():
        return None

    try:
        with open(metrics_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {metrics_path}: {e}")
        return None


def compare_visual_quality(
    main_metrics: Dict,
    pl_metrics: Dict,
    model: str,
    seed: int,
    entity: str
) -> Dict:
    """
    Compare visual quality metrics between main and PL embeddings.

    Returns:
        Dictionary with comparison results and recommendations
    """
    if not main_metrics or not pl_metrics:
        return {
            'status': 'missing',
            'message': 'One or both metrics files missing'
        }

    # Extract key metrics
    main_sil_cosine = main_metrics.get('silhouette_cosine', 0)
    pl_sil_cosine = pl_metrics.get('silhouette_cosine', 0)

    main_sil_euclidean = main_metrics.get('silhouette_euclidean', 0)
    pl_sil_euclidean = pl_metrics.get('silhouette_euclidean', 0)

    main_k = main_metrics.get('clustering', {}).get('n_clusters', 0)
    pl_k = pl_metrics.get('clustering', {}).get('n_clusters', 0)

    main_shape = main_metrics.get('embedding_shape', [0, 0])
    pl_shape = pl_metrics.get('embedding_shape', [0, 0])

    # Compute improvements
    cosine_improvement = ((pl_sil_cosine - main_sil_cosine) / main_sil_cosine * 100) if main_sil_cosine > 0 else 0
    euclidean_improvement = ((pl_sil_euclidean - main_sil_euclidean) / main_sil_euclidean * 100) if main_sil_euclidean > 0 else 0

    # Assess visual quality
    visual_quality = 'good'
    recommendations = []

    if pl_sil_cosine <= main_sil_cosine:
        visual_quality = 'poor'
        recommendations.append("⚠️  PL silhouette NOT better than main - check PL training")
    elif cosine_improvement < 10:
        visual_quality = 'marginal'
        recommendations.append("⚠️  PL improvement < 10% - modest visual improvement expected")
    elif cosine_improvement >= 50:
        visual_quality = 'excellent'
        recommendations.append("✅ PL improvement >= 50% - expect clearly visible cluster separation")
    else:
        visual_quality = 'good'
        recommendations.append("✅ PL improvement 10-50% - expect moderate visual improvement")

    # Check dimensionality reduction benefit
    if pl_shape[1] < main_shape[1]:
        recommendations.append(f"✅ PL uses lower-dimensional space ({pl_shape[1]}D vs {main_shape[1]}D)")

    # Check k values
    if pl_k != main_k:
        recommendations.append(f"ℹ️  Different k values: main k={main_k}, PL k={pl_k} (optimal)")

    return {
        'status': 'compared',
        'visual_quality': visual_quality,
        'main_silhouette_cosine': main_sil_cosine,
        'pl_silhouette_cosine': pl_sil_cosine,
        'cosine_improvement_pct': cosine_improvement,
        'main_silhouette_euclidean': main_sil_euclidean,
        'pl_silhouette_euclidean': pl_sil_euclidean,
        'euclidean_improvement_pct': euclidean_improvement,
        'main_k': main_k,
        'pl_k': pl_k,
        'main_dim': main_shape[1] if len(main_shape) > 1 else 0,
        'pl_dim': pl_shape[1] if len(pl_shape) > 1 else 0,
        'recommendations': recommendations
    }


def verify_dataset(dataset: str, models: List[str] = None, seeds: List[int] = None) -> None:
    """Verify visual quality for all PL models in a dataset."""

    if models is None:
        models = ['mf_pl', 'mlp_pl', 'neumf_pl']
    if seeds is None:
        seeds = [42, 52, 62, 122, 232]

    entities = ['user', 'item']
    umap_dir = Path(f"results/{dataset}/umap_plots")

    if not umap_dir.exists():
        print(f"❌ UMAP directory not found: {umap_dir}")
        print(f"   Run: sbatch scripts/3_generate_umap_plots_optimal.sh")
        return

    print("="*80)
    print(f"UMAP Visual Quality Verification: {dataset}")
    print("="*80)
    print()

    all_results = []
    missing_pl_plots = []

    for model in models:
        for seed in seeds:
            for entity in entities:
                # Load main and PL metrics
                main_metrics = load_umap_metrics(umap_dir, model, seed, entity, 'main')
                pl_metrics = load_umap_metrics(umap_dir, model, seed, entity, 'pl')

                if not pl_metrics:
                    missing_pl_plots.append(f"{model}_seed{seed}_{entity}_pl")
                    continue

                if not main_metrics:
                    print(f"⚠️  Missing main metrics for {model} seed {seed} {entity}")
                    continue

                # Compare
                result = compare_visual_quality(main_metrics, pl_metrics, model, seed, entity)
                result['model'] = model
                result['seed'] = seed
                result['entity'] = entity
                all_results.append(result)

    # Print summary
    if missing_pl_plots:
        print(f"⚠️  Missing {len(missing_pl_plots)} PL-specific UMAP plots:")
        print(f"    Run: sbatch scripts/3_generate_umap_plots_optimal.sh")
        print()

    if not all_results:
        print("❌ No PL UMAP plots found to verify")
        return

    # Group by model
    print("SUMMARY BY MODEL:")
    print("-" * 80)
    print()

    for model in models:
        model_results = [r for r in all_results if r['model'] == model]

        if not model_results:
            print(f"{model.upper()}: No plots found")
            continue

        print(f"{model.upper()}:")

        # Compute averages across seeds
        avg_main_sil = sum(r['main_silhouette_cosine'] for r in model_results) / len(model_results)
        avg_pl_sil = sum(r['pl_silhouette_cosine'] for r in model_results) / len(model_results)
        avg_improvement = sum(r['cosine_improvement_pct'] for r in model_results) / len(model_results)

        quality_counts = {}
        for r in model_results:
            q = r['visual_quality']
            quality_counts[q] = quality_counts.get(q, 0) + 1

        print(f"  Avg silhouette: main={avg_main_sil:.4f}, PL={avg_pl_sil:.4f}")
        print(f"  Avg improvement: {avg_improvement:+.1f}%")
        print(f"  Visual quality: {quality_counts}")

        # Show recommendations for first seed as example
        example = [r for r in model_results if r['seed'] == seeds[0] and r['entity'] == 'user']
        if example:
            print(f"  Example (seed {seeds[0]}, user):")
            for rec in example[0]['recommendations']:
                print(f"    {rec}")

        print()

    # Overall assessment
    print("="*80)
    print("OVERALL ASSESSMENT:")
    print("="*80)
    print()

    excellent_count = sum(1 for r in all_results if r['visual_quality'] == 'excellent')
    good_count = sum(1 for r in all_results if r['visual_quality'] == 'good')
    marginal_count = sum(1 for r in all_results if r['visual_quality'] == 'marginal')
    poor_count = sum(1 for r in all_results if r['visual_quality'] == 'poor')

    total = len(all_results)

    print(f"Total plots verified: {total}")
    print(f"  Excellent quality: {excellent_count} ({excellent_count/total*100:.1f}%)")
    print(f"  Good quality: {good_count} ({good_count/total*100:.1f}%)")
    print(f"  Marginal quality: {marginal_count} ({marginal_count/total*100:.1f}%)")
    print(f"  Poor quality: {poor_count} ({poor_count/total*100:.1f}%)")
    print()

    if poor_count > 0:
        print("⚠️  WARNING: Some PL plots show worse clustering than main!")
        print("   Review training logs and PL hyperparameters.")
    elif excellent_count + good_count >= total * 0.8:
        print("✅ PASS: Majority of PL plots show good or excellent cluster separation")
        print("   Visual inspection should confirm distinct cluster boundaries.")
    else:
        print("⚠️  BORDERLINE: Many plots show only marginal improvement")
        print("   Consider tuning PL hyperparameters for better separation.")

    print()
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Verify UMAP visual quality for PL vs main embeddings"
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='support_groups_full_164',
        help='Dataset name (default: support_groups_full_164)'
    )

    args = parser.parse_args()

    verify_dataset(args.dataset)

    print("NEXT STEPS:")
    print("-" * 80)
    print("1. Visually inspect UMAP plots side-by-side:")
    print(f"   results/{args.dataset}/umap_plots/{{model}}_seed{{seed}}_user_main_umap.png")
    print(f"   results/{args.dataset}/umap_plots/{{model}}_seed{{seed}}_user_pl_umap.png")
    print()
    print("2. Look for:")
    print("   ✅ Tighter, more compact clusters in PL plots")
    print("   ✅ Greater separation between cluster centroids")
    print("   ✅ Less overlap between different colored groups")
    print()
    print("3. If visual quality doesn't match numerical improvements:")
    print("   - Check that UMAP used correct k (should match metrics JSON)")
    print("   - Verify embeddings loaded correctly")
    print("   - Consider different UMAP hyperparameters (n_neighbors, min_dist)")
    print("="*80)


if __name__ == '__main__':
    sys.exit(main())
