#!/usr/bin/env python3
"""
Extract User and Item Embeddings from Trained Models
=====================================================

Extracts learned embedding matrices from trained NCF model checkpoints
and saves them as numpy arrays for downstream analysis (UMAP, clustering).

Supports:
- MF models (user_emb, item_emb)
- MLP models (user_emb, item_emb before MLP layers)
- NeuMF models (both MF and MLP embedding components)
- PL models (main embeddings + separate PL-branch embeddings)

Usage:
    # Extract embeddings for specific model
    python src/extract_embeddings.py --model mf_baseline --dataset support_groups_full_164 --seed 42

    # Extract for all trained models
    python src/extract_embeddings.py --all

    # Extract for specific dataset
    python src/extract_embeddings.py --dataset support_groups_full_164 --all-models

Output:
    results/{dataset}/embeddings/{model}_seed{seed}_user_emb.npy
    results/{dataset}/embeddings/{model}_seed{seed}_item_emb.npy

    For PL models (*_pl), also extracts PL-branch embeddings:
    results/{dataset}/embeddings/{model}_seed{seed}_pl_user_emb.npy
    results/{dataset}/embeddings/{model}_seed{seed}_pl_item_emb.npy
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent))

from mf import MF
from mlp import MLP
from neumf import NeuMF


def extract_mf_embeddings(model: MF) -> Tuple[np.ndarray, np.ndarray]:
    """Extract user and item embeddings from MF model."""
    user_emb = model.user_emb.weight.detach().cpu().numpy()
    item_emb = model.item_emb.weight.detach().cpu().numpy()
    return user_emb, item_emb


def extract_mlp_embeddings(model: MLP) -> Tuple[np.ndarray, np.ndarray]:
    """Extract user and item embeddings from MLP model (before FC layers)."""
    user_emb = model.user_emb.weight.detach().cpu().numpy()
    item_emb = model.item_emb.weight.detach().cpu().numpy()
    return user_emb, item_emb


def extract_neumf_embeddings(model: NeuMF) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract user and item embeddings from NeuMF model.

    NeuMF has two paths (MF + MLP). We concatenate both for comprehensive representation.
    """
    # MF path embeddings
    user_emb_mf = model.user_emb_mf.weight.detach().cpu().numpy()
    item_emb_mf = model.item_emb_mf.weight.detach().cpu().numpy()

    # MLP path embeddings
    user_emb_mlp = model.user_emb_mlp.weight.detach().cpu().numpy()
    item_emb_mlp = model.item_emb_mlp.weight.detach().cpu().numpy()

    # Concatenate MF + MLP representations
    user_emb = np.concatenate([user_emb_mf, user_emb_mlp], axis=1)
    item_emb = np.concatenate([item_emb_mf, item_emb_mlp], axis=1)

    return user_emb, item_emb


def extract_pl_embeddings_from_checkpoint(checkpoint: Dict) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Extract PL-branch embeddings directly from checkpoint state dict.

    For PL models, the PL branch has separate embeddings that are used for
    cosine/feature alignment. These are stored under 'pl.user_emb.weight' and
    'pl.item_emb.weight' keys in the checkpoint.

    Args:
        checkpoint: Model state dict from torch.load()

    Returns:
        (pl_user_emb, pl_item_emb) as numpy arrays, or None if PL embeddings not found
    """
    if 'pl.user_emb.weight' in checkpoint and 'pl.item_emb.weight' in checkpoint:
        pl_user_emb = checkpoint['pl.user_emb.weight'].cpu().numpy()
        pl_item_emb = checkpoint['pl.item_emb.weight'].cpu().numpy()
        return pl_user_emb, pl_item_emb
    return None


def load_model_and_extract(
    model_path: Path,
    model_type: str,
    num_users: int,
    num_items: int,
    config: Optional[Dict] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a trained model checkpoint and extract embeddings.

    Args:
        model_path: Path to .model or .pth file
        model_type: One of ['mf_baseline', 'mf_pl', 'mlp_baseline', 'mlp_pl', 'neumf_baseline', 'neumf_pl']
        num_users: Number of users in dataset
        num_items: Number of items in dataset
        config: Optional model configuration dict

    Returns:
        (user_embeddings, item_embeddings) as numpy arrays
    """
    if config is None:
        config = {}

    # Load checkpoint first to avoid rebuilding model
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    # Create model instance with correct config including num_users/num_items
    config['num_users'] = num_users
    config['num_items'] = num_items

    # Add user_text_dim for PL models that were trained with text features
    if model_type in ['mf_pl', 'mlp_pl', 'neumf_pl']:
        # Check if the checkpoint has text_proj weights to determine if text was used
        if 'text_proj.weight' in checkpoint:
            config['user_text_dim'] = checkpoint['text_proj.weight'].shape[1]

    if model_type in ['mf_baseline', 'mf_pl']:
        model = MF(config)
        extract_fn = extract_mf_embeddings

    elif model_type in ['mlp_baseline', 'mlp_pl']:
        from mlp import MLP, MLPPL
        # Use MLPPL if pl_dim is set, otherwise use base MLP
        if config.get('pl_dim') is not None:
            model = MLPPL(config)
        else:
            model = MLP(config)
        extract_fn = extract_mlp_embeddings

    elif model_type in ['neumf_baseline', 'neumf_pl']:
        model = NeuMF(config)
        extract_fn = extract_neumf_embeddings
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load the checkpoint into the model
    model.load_state_dict(checkpoint, strict=False)
    model.eval()

    # Extract embeddings
    user_emb, item_emb = extract_fn(model)

    return user_emb, item_emb


def get_dataset_metadata(dataset: str) -> Dict:
    """Get dataset metadata (num_users, num_items) from splits."""
    from split_utils import prepare_leave_one_out

    try:
        _, _, _, _, metadata = prepare_leave_one_out(dataset)
        return metadata
    except Exception as e:
        print(f"Warning: Could not load metadata for {dataset}: {e}")
        # Return defaults
        return {'num_users': 1000, 'num_items': 1000}


def extract_single_model(model_name: str, dataset: str, seed: int, output_dir: Path):
    """Extract embeddings for a single model/dataset/seed combination."""
    from config import get_config

    print(f"Extracting embeddings: {model_name} on {dataset} (seed={seed})")

    # Get dataset metadata
    metadata = get_dataset_metadata(dataset)
    num_users = metadata['num_users']
    num_items = metadata['num_items']

    # Get model config
    config = get_config(model_name, dataset)
    config['num_users'] = num_users
    config['num_items'] = num_items

    # Find model checkpoint
    model_path = Path(f"models/trained/{model_name}_{dataset}_seed{seed}.model")

    if not model_path.exists():
        print(f"  ⚠️  Model not found: {model_path}")
        return False

    # Extract embeddings
    try:
        user_emb, item_emb = load_model_and_extract(
            model_path, model_name, num_users, num_items, config
        )

        # Save main embeddings
        output_dir.mkdir(parents=True, exist_ok=True)
        user_output = output_dir / f"{model_name}_seed{seed}_user_emb.npy"
        item_output = output_dir / f"{model_name}_seed{seed}_item_emb.npy"

        np.save(user_output, user_emb)
        np.save(item_output, item_emb)

        print(f"  ✅ Saved: {user_output.name} (shape: {user_emb.shape})")
        print(f"  ✅ Saved: {item_output.name} (shape: {item_emb.shape})")

        # Also extract PL-branch embeddings if this is a PL model
        if '_pl' in model_name:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            pl_embeddings = extract_pl_embeddings_from_checkpoint(checkpoint)

            if pl_embeddings is not None:
                pl_user_emb, pl_item_emb = pl_embeddings

                pl_user_output = output_dir / f"{model_name}_seed{seed}_pl_user_emb.npy"
                pl_item_output = output_dir / f"{model_name}_seed{seed}_pl_item_emb.npy"

                np.save(pl_user_output, pl_user_emb)
                np.save(pl_item_output, pl_item_emb)

                print(f"  ✅ Saved: {pl_user_output.name} (shape: {pl_user_emb.shape})")
                print(f"  ✅ Saved: {pl_item_output.name} (shape: {pl_item_emb.shape})")
            else:
                print(f"  ⚠️  No PL embeddings found in checkpoint (expected for {model_name})")

        return True

    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def extract_all_models(datasets=None, models=None, seeds=None):
    """Extract embeddings for all trained models."""
    if datasets is None:
        datasets = ['support_groups_full_164', 'support_groups_full_164_loo']
    if models is None:
        models = ['mf_baseline', 'mf_pl', 'mlp_baseline', 'mlp_pl', 'neumf_baseline', 'neumf_pl']
    if seeds is None:
        seeds = [42, 52, 62, 122, 232]

    total = len(datasets) * len(models) * len(seeds)
    success_count = 0

    print(f"Extracting embeddings for {total} models...")
    print("=" * 80)

    for dataset in datasets:
        output_dir = Path(f"results/{dataset}/embeddings")

        for model_name in models:
            for seed in seeds:
                if extract_single_model(model_name, dataset, seed, output_dir):
                    success_count += 1

    print("=" * 80)
    print(f"Extraction complete: {success_count}/{total} successful")

    if success_count < total:
        print(f"⚠️  {total - success_count} models failed or were not found")

    return success_count == total


def main():
    parser = argparse.ArgumentParser(description="Extract embeddings from trained NCF models")
    parser.add_argument('--model', type=str, help='Model name (e.g., mf_baseline)')
    parser.add_argument('--dataset', type=str, help='Dataset name (e.g., support_groups_full_164)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--all', action='store_true', help='Extract all models')
    parser.add_argument('--all-models', action='store_true', help='Extract all models for specified dataset')
    parser.add_argument('--output-dir', type=str, help='Custom output directory')

    args = parser.parse_args()

    if args.all:
        # Extract everything
        success = extract_all_models()
        sys.exit(0 if success else 1)

    elif args.all_models and args.dataset:
        # Extract all models for one dataset
        models = ['mf_baseline', 'mf_pl', 'mlp_baseline', 'mlp_pl', 'neumf_baseline', 'neumf_pl']
        seeds = [42, 52, 62, 122, 232]
        output_dir = Path(args.output_dir) if args.output_dir else Path(f"results/{args.dataset}/embeddings")

        total = len(models) * len(seeds)
        success_count = 0

        for model_name in models:
            for seed in seeds:
                if extract_single_model(model_name, args.dataset, seed, output_dir):
                    success_count += 1

        print(f"Extracted {success_count}/{total} models for {args.dataset}")
        sys.exit(0 if success_count == total else 1)

    elif args.model and args.dataset:
        # Extract single model
        output_dir = Path(args.output_dir) if args.output_dir else Path(f"results/{args.dataset}/embeddings")
        success = extract_single_model(args.model, args.dataset, args.seed, output_dir)
        sys.exit(0 if success else 1)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
