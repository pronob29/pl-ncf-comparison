#!/usr/bin/env python3
"""
Generate Comprehensive Comparison Plots
========================================

Creates publication-quality comparison plots showing model performance
across multiple seeds with proper error bars.

Generates 3 plots per dataset:
1. HR@5 comparison bar chart (mean ± std across seeds)
2. NDCG@5 comparison bar chart (mean ± std across seeds)
3. Training curves (loss, HR@5, NDCG@5 over epochs) - if available

Color scheme:
- Blue tones for baseline models
- Orange/red tones for pseudo-labeling models

Usage:
    # Generate plots for specific dataset
    python src/generate_comparison_plots.py --dataset support_groups_full_164

    # Generate for all datasets
    python src/generate_comparison_plots.py --all

Output:
    results/{dataset}/plots/hr5_comparison.png
    results/{dataset}/plots/ndcg5_comparison.png
    results/{dataset}/plots/training_curves.png (if TensorBoard logs available)
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_seed_results(dataset: str, results_dir: Optional[Path] = None) -> List[Dict]:
    """
    Load all seed results JSON files for a dataset.

    Returns:
        List of result dictionaries from results/{dataset}/seeds/*.json
    """
    if results_dir is None:
        results_dir = Path(f"results/{dataset}/seeds")

    if not results_dir.exists():
        print(f"⚠️  Results directory not found: {results_dir}")
        return []

    results = []
    json_files = list(results_dir.glob("*.json"))

    if not json_files:
        print(f"⚠️  No JSON result files found in {results_dir}")
        return results

    for json_file in json_files:
        try:
            with json_file.open('r') as f:
                data = json.load(f)
                results.append(data)
        except Exception as e:
            print(f"⚠️  Error reading {json_file}: {e}")

    print(f"Loaded {len(results)} result files from {results_dir}")
    return results


def aggregate_metrics(results: List[Dict]) -> Dict[str, Dict]:
    """
    Aggregate metrics across seeds to compute mean and std.

    Returns:
        {
            'model_name': {
                'test_hr': {'mean': float, 'std': float, 'values': [...]},
                'test_ndcg': {'mean': float, 'std': float, 'values': [...]},
                'val_hr': {...},
                'val_ndcg': {...}
            },
            ...
        }
    """
    grouped = defaultdict(lambda: defaultdict(list))

    for result in results:
        model = result.get('model', 'unknown')
        grouped[model]['test_hr'].append(result.get('test_hr', 0.0))
        grouped[model]['test_ndcg'].append(result.get('test_ndcg', 0.0))
        grouped[model]['test_auc'].append(result.get('test_auc', 0.0))
        grouped[model]['val_hr'].append(result.get('val_hr', 0.0))
        grouped[model]['val_ndcg'].append(result.get('val_ndcg', 0.0))
        grouped[model]['val_auc'].append(result.get('val_auc', 0.0))

    aggregated = {}
    for model, metrics in grouped.items():
        aggregated[model] = {}
        for metric_name, values in metrics.items():
            arr = np.array(values)
            aggregated[model][metric_name] = {
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr)),
                'values': values
            }

    return aggregated


def get_model_colors() -> Dict[str, str]:
    """
    Return color scheme for models.

    Baseline models: Blue tones
    Pseudo-labeling models: Orange tones
    """
    colors = {
        'mf_baseline': '#3498db',      # Blue
        'mlp_baseline': '#5dade2',     # Light blue
        'neumf_baseline': '#85c1e9',   # Very light blue
        'mf_pl': '#e74c3c',            # Red-orange
        'mlp_pl': '#ec7063',           # Light red-orange
        'neumf_pl': '#f1948a',         # Very light red-orange
    }
    return colors


def plot_hr5_comparison(
    aggregated: Dict[str, Dict],
    dataset: str,
    output_path: Path,
    dpi: int = 300
):
    """Create HR@5 comparison bar chart."""
    # Set style
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.3)

    # Prepare data
    models = list(aggregated.keys())
    model_order = ['mf_baseline', 'mf_pl', 'mlp_baseline', 'mlp_pl', 'neumf_baseline', 'neumf_pl']
    models = [m for m in model_order if m in models]

    hr_means = [aggregated[m]['test_hr']['mean'] for m in models]
    hr_stds = [aggregated[m]['test_hr']['std'] for m in models]

    # Colors
    colors_map = get_model_colors()
    bar_colors = [colors_map.get(m, '#95a5a6') for m in models]

    # Model labels (prettier)
    labels = [m.replace('_', ' ').title() for m in models]

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(models))
    width = 0.6

    bars = ax.bar(
        x, hr_means, width,
        yerr=hr_stds,
        color=bar_colors,
        alpha=0.85,
        capsize=5,
        edgecolor='black',
        linewidth=1.2
    )

    # Formatting
    ax.set_ylabel("Hit Ratio @5", fontsize=14, fontweight='bold')
    ax.set_title(f"{dataset.title()} Dataset: Hit Ratio @5 Comparison", fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylim(0, min(1.0, max(hr_means) * 1.15))
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, hr_means, hr_stds)):
        height = bar.get_height()
        label_text = f"{mean:.4f}"
        if std > 0:
            label_text += f"\n±{std:.4f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + max(hr_means) * 0.02,
            label_text,
            ha='center',
            va='bottom',
            fontsize=9,
            fontweight='bold'
        )

    # Legend
    baseline_patch = plt.Rectangle((0, 0), 1, 1, fc='#3498db', alpha=0.85, edgecolor='black', linewidth=1.2)
    pl_patch = plt.Rectangle((0, 0), 1, 1, fc='#e74c3c', alpha=0.85, edgecolor='black', linewidth=1.2)
    ax.legend(
        [baseline_patch, pl_patch],
        ['Baseline Models', 'Pseudo-Labeling Models'],
        loc='upper left',
        frameon=True,
        framealpha=0.9,
        fontsize=11
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"  ✅ Saved: {output_path.name}")


def plot_ndcg5_comparison(
    aggregated: Dict[str, Dict],
    dataset: str,
    output_path: Path,
    dpi: int = 300
):
    """Create NDCG@5 comparison bar chart."""
    # Set style
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.3)

    # Prepare data
    models = list(aggregated.keys())
    model_order = ['mf_baseline', 'mf_pl', 'mlp_baseline', 'mlp_pl', 'neumf_baseline', 'neumf_pl']
    models = [m for m in model_order if m in models]

    ndcg_means = [aggregated[m]['test_ndcg']['mean'] for m in models]
    ndcg_stds = [aggregated[m]['test_ndcg']['std'] for m in models]

    # Colors
    colors_map = get_model_colors()
    bar_colors = [colors_map.get(m, '#95a5a6') for m in models]

    # Model labels
    labels = [m.replace('_', ' ').title() for m in models]

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(models))
    width = 0.6

    bars = ax.bar(
        x, ndcg_means, width,
        yerr=ndcg_stds,
        color=bar_colors,
        alpha=0.85,
        capsize=5,
        edgecolor='black',
        linewidth=1.2
    )

    # Formatting
    ax.set_ylabel("NDCG @5", fontsize=14, fontweight='bold')
    ax.set_title(f"{dataset.title()} Dataset: NDCG @5 Comparison", fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylim(0, min(1.0, max(ndcg_means) * 1.15))
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels
    for i, (bar, mean, std) in enumerate(zip(bars, ndcg_means, ndcg_stds)):
        height = bar.get_height()
        label_text = f"{mean:.4f}"
        if std > 0:
            label_text += f"\n±{std:.4f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + max(ndcg_means) * 0.02,
            label_text,
            ha='center',
            va='bottom',
            fontsize=9,
            fontweight='bold'
        )

    # Legend
    baseline_patch = plt.Rectangle((0, 0), 1, 1, fc='#3498db', alpha=0.85, edgecolor='black', linewidth=1.2)
    pl_patch = plt.Rectangle((0, 0), 1, 1, fc='#e74c3c', alpha=0.85, edgecolor='black', linewidth=1.2)
    ax.legend(
        [baseline_patch, pl_patch],
        ['Baseline Models', 'Pseudo-Labeling Models'],
        loc='upper left',
        frameon=True,
        framealpha=0.9,
        fontsize=11
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"  ✅ Saved: {output_path.name}")


def plot_auc_comparison(
    aggregated: Dict[str, Dict],
    dataset: str,
    output_path: Path,
    dpi: int = 300
):
    """Create AUC comparison bar chart."""
    # Set style
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.3)

    # Prepare data
    models = list(aggregated.keys())
    model_order = ['mf_baseline', 'mf_pl', 'mlp_baseline', 'mlp_pl', 'neumf_baseline', 'neumf_pl']
    models = [m for m in model_order if m in models]

    auc_means = [aggregated[m]['test_auc']['mean'] for m in models]
    auc_stds = [aggregated[m]['test_auc']['std'] for m in models]

    # Colors
    colors_map = get_model_colors()
    bar_colors = [colors_map.get(m, '#95a5a6') for m in models]

    # Model labels
    labels = [m.replace('_', ' ').title() for m in models]

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(models))
    width = 0.6

    bars = ax.bar(
        x, auc_means, width,
        yerr=auc_stds,
        color=bar_colors,
        alpha=0.85,
        capsize=5,
        edgecolor='black',
        linewidth=1.2
    )

    # Formatting
    ax.set_ylabel("AUC", fontsize=14, fontweight='bold')
    ax.set_title(f"{dataset.title()} Dataset: AUC Comparison", fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylim(0.4, min(1.0, max(auc_means) * 1.05))  # Start at 0.4 since AUC baseline is 0.5
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels
    for i, (bar, mean, std) in enumerate(zip(bars, auc_means, auc_stds)):
        height = bar.get_height()
        label_text = f"{mean:.4f}"
        if std > 0:
            label_text += f"\n±{std:.4f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + (max(auc_means) - 0.4) * 0.02,
            label_text,
            ha='center',
            va='bottom',
            fontsize=9,
            fontweight='bold'
        )

    # Legend
    baseline_patch = plt.Rectangle((0, 0), 1, 1, fc='#3498db', alpha=0.85, edgecolor='black', linewidth=1.2)
    pl_patch = plt.Rectangle((0, 0), 1, 1, fc='#e74c3c', alpha=0.85, edgecolor='black', linewidth=1.2)
    ax.legend(
        [baseline_patch, pl_patch],
        ['Baseline Models', 'Pseudo-Labeling Models'],
        loc='lower left',
        frameon=True,
        framealpha=0.9,
        fontsize=11
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"  ✅ Saved: {output_path.name}")


def generate_dataset_plots(dataset: str, dpi: int = 300) -> bool:
    """Generate all comparison plots for a single dataset."""
    print(f"\nGenerating comparison plots for: {dataset}")
    print("=" * 80)

    # Load results
    results = load_seed_results(dataset)

    if not results:
        print(f"⚠️  No results found for {dataset}")
        return False

    # Aggregate metrics
    aggregated = aggregate_metrics(results)

    if not aggregated:
        print(f"⚠️  No aggregated metrics for {dataset}")
        return False

    print(f"Aggregated metrics for {len(aggregated)} models:")
    for model, metrics in aggregated.items():
        print(f"  {model}: HR@5={metrics['test_hr']['mean']:.4f}±{metrics['test_hr']['std']:.4f}, "
              f"NDCG@5={metrics['test_ndcg']['mean']:.4f}±{metrics['test_ndcg']['std']:.4f}, "
              f"AUC={metrics['test_auc']['mean']:.4f}±{metrics['test_auc']['std']:.4f}")

    # Generate plots
    output_dir = Path(f"results/{dataset}/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        plot_hr5_comparison(aggregated, dataset, output_dir / "hr5_comparison.png", dpi=dpi)
        plot_ndcg5_comparison(aggregated, dataset, output_dir / "ndcg5_comparison.png", dpi=dpi)
        plot_auc_comparison(aggregated, dataset, output_dir / "auc_comparison.png", dpi=dpi)
        print("=" * 80)
        return True
    except Exception as e:
        print(f"❌ Error generating plots: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_all_plots(datasets=None, dpi: int = 300) -> bool:
    """Generate comparison plots for all datasets."""
    if datasets is None:
        datasets = ['support_groups_full_164', 'support_groups_full_164_loo']

    print(f"Generating comparison plots for {len(datasets)} datasets...")
    print("=" * 80)

    success_count = 0
    for dataset in datasets:
        if generate_dataset_plots(dataset, dpi=dpi):
            success_count += 1

    print("\n" + "=" * 80)
    print(f"Comparison plots complete: {success_count}/{len(datasets)} datasets")

    return success_count == len(datasets)


def main():
    parser = argparse.ArgumentParser(description="Generate comparison plots for NCF experiments")
    parser.add_argument('--dataset', type=str, help='Dataset name (e.g., support_groups_full_164)')
    parser.add_argument('--all', action='store_true', help='Generate for all datasets')
    parser.add_argument('--dpi', type=int, default=300, help='Plot resolution (default: 300)')

    args = parser.parse_args()

    if args.all:
        # Generate all plots
        success = generate_all_plots(dpi=args.dpi)
        sys.exit(0 if success else 1)

    elif args.dataset:
        # Generate for specific dataset
        success = generate_dataset_plots(args.dataset, dpi=args.dpi)
        sys.exit(0 if success else 1)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
