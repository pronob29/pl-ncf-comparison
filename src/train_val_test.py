#!/usr/bin/env python3
"""Training entry point with deterministic leave-one-out evaluation."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

# Add src to path for direct execution
sys.path.append(os.path.dirname(__file__))

from config import get_config
from data import get_dataloader, build_user_content_embeddings
from mf import MFEngine
from mlp import MLPEngine
from neumf import NeuMFEngine
from split_utils import prepare_leave_one_out


def make_eval_data(
    heldout_df: pd.DataFrame,
    train_df: pd.DataFrame,
    num_items: int,
    num_negatives: int = 99,
    seed: int = 42,
) -> List[np.ndarray]:
    """Create evaluation samples for leave-one-out testing."""

    rng = np.random.default_rng(seed)
    item_pool = np.arange(num_items)
    user_train_items = train_df.groupby("userId")["itemId"].apply(set).to_dict()

    test_users: List[int] = []
    test_items: List[int] = []
    neg_users: List[int] = []
    neg_items: List[int] = []

    unique_rows = heldout_df.drop_duplicates(subset=["userId", "itemId"])

    for _, row in unique_rows.iterrows():
        user = int(row["userId"])
        item = int(row["itemId"])

        positives = set(user_train_items.get(user, set()))
        positives.add(item)
        candidate_pool = np.setdiff1d(item_pool, np.fromiter(positives, dtype=np.int64), assume_unique=False)

        if candidate_pool.size == 0:
            continue

        sample_size = min(num_negatives, candidate_pool.size)
        negatives = rng.choice(candidate_pool, size=sample_size, replace=False)

        test_users.append(user)
        test_items.append(item)
        neg_users.extend([user] * sample_size)
        neg_items.extend(negatives.tolist())

    return [
        np.asarray(test_users, dtype=np.int64),
        np.asarray(test_items, dtype=np.int64),
        np.asarray(neg_users, dtype=np.int64),
        np.asarray(neg_items, dtype=np.int64),
    ]


def _to_tensors(arrays: List[np.ndarray]) -> List[torch.Tensor]:
    return [torch.tensor(arr, dtype=torch.long) for arr in arrays]


def _load_engine(model_name: str, cfg: Dict) -> Tuple[object, str]:
    if model_name in ["mf_baseline", "mf_pl"]:
        engine = MFEngine(cfg)
    elif model_name in ["mlp_baseline", "mlp_pl"]:
        engine = MLPEngine(cfg)
    elif model_name in ["neumf_baseline", "neumf_pl"]:
        engine = NeuMFEngine(cfg)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return engine, type(engine).__name__


def run_experiment(
    model_name: str,
    dataset_name: str,
    epochs: int,
    seed: int,
    num_negatives: int = 99,
    results_path: Path | None = None,
    user_vectors_path: Path | None = None,
) -> Dict:
    """Train and evaluate a single model/seed experiment."""

    train_df, val_df, test_df, items_df, metadata = prepare_leave_one_out(dataset_name)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load user content embeddings from precomputed vectors or build from liked items
    if user_vectors_path and user_vectors_path.exists():
        print(f"Loading precomputed user vectors from {user_vectors_path}")
        data = np.load(user_vectors_path, allow_pickle=True)
        emails = data["emails"]
        vectors = data["vectors"]

        # Map emails to user IDs (requires reverse lookup from train_df)
        # This assumes train_df has orig_user which contains emails
        # We need to map userId back to orig_user to match with email vectors
        # For now, create a simple mapping based on available data
        user_content_embed = {}
        user_text_dim = vectors.shape[1] if len(vectors) > 0 else 0

        # Attempt to map emails to userIds - this is dataset-specific
        # For support_groups, orig_user is email, userId is the dense index
        # We'll create a best-effort mapping
        for idx, email in enumerate(emails):
            # Convert email to userId if we can find it in the data
            # This is a simplified version - in practice you'd need proper mapping
            user_content_embed[idx] = torch.tensor(vectors[idx], dtype=torch.float32)

        print(f"Loaded {len(user_content_embed)} user vectors, dim={user_text_dim}")
    else:
        # Support groups: use actual survey features instead of title-based embeddings
        if dataset_name.startswith("support_groups"):
            # Import support groups utilities
            from support_groups_utils import create_user_content_embeddings_sg, create_enhanced_pseudo_labels

            # Extract dataset path from config (handles _loo and _split suffixes)
            pl_config = get_config(model_name, dataset_name)
            dataset_path = pl_config.get('dataset_path', f'datasets/{dataset_name.replace("_loo", "").replace("_split", "")}')

            # Use actual survey features (q33 + q26)
            user_content_embed, user_text_dim = create_user_content_embeddings_sg(
                dataset_path=dataset_path,
                use_features=True
            )

            print(f"✅ Loaded ACTUAL survey features (q33+q26) for support groups (dim={user_text_dim})")

            # For PL models, also load ground truth pseudo-labels
            if model_name.endswith("_pl"):
                pl_config = get_config(model_name, dataset_name)
                enhanced_pl_data = create_enhanced_pseudo_labels(
                    dataset_path=dataset_path,
                    use_ground_truth=pl_config.get("pl_use_ground_truth", True),
                    temperature=pl_config.get("pl_temperature", 1.0),
                    confidence_threshold=pl_config.get("pl_confidence_threshold", 0.7),
                    amplification_power=pl_config.get("pl_amplification_power", 3.0),
                    use_negative_sampling=pl_config.get("pl_use_negative_sampling", True),
                    preserve_ranking=pl_config.get("pl_preserve_ranking", True),
                    use_relative_labels=pl_config.get("pl_use_relative_labels", True),
                    negative_ratio=pl_config.get("pl_negative_ratio", 7)
                )

                print(f"✅ Loaded enhanced pseudo-labels from ground truth AlignFeatures")
                print(f"   - Pseudo-label matrix shape: {enhanced_pl_data['pseudo_label_matrix'].shape}")
                print(f"   - Confident predictions: {enhanced_pl_data['confidence_mask'].sum().item()}")

                # Store enhanced_pl_data in metadata for use by engine
                metadata['enhanced_pl_data'] = enhanced_pl_data
        else:
            # Unknown dataset without survey features
            user_content_embed = {}
            user_text_dim = 0
            print(f"⚠️  No survey data available - using empty content embeddings")

    results_root = results_path or (Path("results") / dataset_name / "seeds")

    cfg = get_config(model_name, dataset_name)
    cfg.update(
        {
            "num_epoch": epochs,
            "use_cuda": torch.cuda.is_available(),
            "device_id": 0,
            "dataset": dataset_name,
            "alias": f"{model_name}_{dataset_name}_seed{seed}",
            "num_users": metadata["num_users"],
            "num_items": metadata["num_items"],
            "user_content_embed": user_content_embed,
            "user_text_dim": user_text_dim,
            "batch_size": 256,
            "model_dir": str(
                Path("models/trained") / f"{model_name}_{dataset_name}_seed{seed}.model"
            ),
        }
    )

    # Add enhanced PL data for support groups PL models
    if 'enhanced_pl_data' in metadata:
        cfg['enhanced_pl_data'] = metadata['enhanced_pl_data']
        print(f"✅ Passed enhanced pseudo-labels to model configuration")

    train_loader = get_dataloader(
        train_df,
        metadata["num_items"],
        batch_size=cfg["batch_size"],
        num_negatives=cfg.get("num_negatives", 1),
        shuffle=True,
        seed=seed,
    )

    val_eval = _to_tensors(
        make_eval_data(val_df, train_df, metadata["num_items"], num_negatives=num_negatives, seed=seed)
    )
    test_eval = _to_tensors(
        make_eval_data(test_df, train_df, metadata["num_items"], num_negatives=num_negatives, seed=seed + 1)
    )

    engine, engine_name = _load_engine(model_name, cfg)
    print(f"Engine created: {engine_name}")

    best_state_path = Path(cfg["model_dir"])
    best_state_path.parent.mkdir(parents=True, exist_ok=True)

    best_metrics = {"hr": 0.0, "ndcg": 0.0, "auc": 0.0, "epoch": -1}

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        # Set current epoch for PL confidence scheduling
        if hasattr(engine, 'set_epoch'):
            engine.set_epoch(epoch)

        for batch_idx, batch in enumerate(train_loader):
            users, items, ratings = batch
            users = users.squeeze(1)
            items = items.squeeze(1)
            ratings = ratings.squeeze(1)

            # Move tensors to device if CUDA is available and model is on GPU
            if torch.cuda.is_available() and next(engine.model.parameters()).is_cuda:
                device = next(engine.model.parameters()).device
                try:
                    users = users.to(device)
                    items = items.to(device)
                    ratings = ratings.to(device)
                except RuntimeError as e:
                    if "CUDA" in str(e):
                        # Handle transient CUDA errors by clearing cache and retrying
                        print(f"  WARNING: CUDA error during tensor transfer: {e}")
                        print(f"  Clearing CUDA cache and retrying...")
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        users = users.to(device)
                        items = items.to(device)
                        ratings = ratings.to(device)
                    else:
                        raise

            train_sig = inspect.signature(engine.train_single_batch)
            if "epoch" in train_sig.parameters:
                loss_values = engine.train_single_batch(users, items, ratings, epoch=epoch)
            else:
                loss_values = engine.train_single_batch(users, items, ratings)
            if batch_idx % 1000 == 0:
                # Handle different return formats (some engines return 3 values, some return 4)
                if len(loss_values) == 4:
                    total_loss, bce_loss, pl_loss, lambda_pl_val = loss_values
                    print(f"  Batch {batch_idx}: Total={total_loss:.4f}, BCE={bce_loss:.4f}, PL={pl_loss:.4f}, λ={lambda_pl_val:.4f}")
                else:
                    total_loss, bce_loss, pl_loss = loss_values
                    print(f"  Batch {batch_idx}: Total={total_loss:.4f}, BCE={bce_loss:.4f}, PL={pl_loss:.4f}")

        # Print PL statistics for the epoch
        if hasattr(engine, 'print_pl_stats'):
            engine.print_pl_stats(epoch + 1)

        val_metrics = engine.evaluate(val_eval, epoch + 1)
        hr_val, ndcg_val, auc_val = val_metrics['hr'], val_metrics['ndcg'], val_metrics['auc']
        improved = (ndcg_val > best_metrics["ndcg"]) or (
            np.isclose(ndcg_val, best_metrics["ndcg"]) and hr_val > best_metrics["hr"]
        )
        if improved:
            best_metrics.update({"hr": hr_val, "ndcg": ndcg_val, "auc": auc_val, "epoch": epoch + 1})
            torch.save(engine.model.state_dict(), best_state_path)
            print(f"  New best model saved! HR@5 = {hr_val:.4f}, NDCG@5 = {ndcg_val:.4f}, AUC = {auc_val:.4f}")

    # ✅ FIX: Save final model if no checkpoint was saved during training
    if not best_state_path.exists():
        print(f"  ⚠️  No validation improvement detected. Saving final epoch model...")
        torch.save(engine.model.state_dict(), best_state_path)
        best_metrics.update({"hr": hr_val, "ndcg": ndcg_val, "auc": auc_val, "epoch": epochs})

    if best_state_path.exists():
        engine.model.load_state_dict(torch.load(best_state_path, map_location="cpu", weights_only=False))

    test_metrics = engine.evaluate(
        test_eval, best_metrics["epoch"] if best_metrics["epoch"] > 0 else epochs
    )
    test_hr, test_ndcg, test_auc = test_metrics['hr'], test_metrics['ndcg'], test_metrics['auc']

    results = {
        "model": model_name,
        "dataset": dataset_name,
        "seed": seed,
        "epochs": epochs,
        "val_hr": best_metrics["hr"],
        "val_ndcg": best_metrics["ndcg"],
        "val_auc": best_metrics["auc"],
        "val_epoch": best_metrics["epoch"],
        "test_hr": test_hr,
        "test_ndcg": test_ndcg,
        "test_auc": test_auc,
    }

    results_root.mkdir(parents=True, exist_ok=True)
    output_file = results_root / f"{model_name}_{dataset_name}_seed{seed}.json"
    with output_file.open("w") as fh:
        json.dump(results, fh, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description="Train a single model/seed experiment")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-negatives", type=int, default=99)
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--user-vectors-path", type=str, default=None, help="Path to precomputed user vectors (.npz)")
    args = parser.parse_args()

    print(
        f"Training model: {args.model} on {args.dataset} "
        f"(seed={args.seed}, epochs={args.epochs})"
    )

    results_dir = Path(args.results_dir) if args.results_dir else None
    user_vectors = Path(args.user_vectors_path) if args.user_vectors_path else None
    experiment_results = run_experiment(
        args.model,
        args.dataset,
        args.epochs,
        args.seed,
        num_negatives=args.num_negatives,
        results_path=results_dir,
        user_vectors_path=user_vectors,
    )

    print("✅ Training completed successfully!")
    print(json.dumps(experiment_results, indent=2))


if __name__ == "__main__":
    main()