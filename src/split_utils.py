"""Utility helpers for deterministic train/validation/test splits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple, Set

import numpy as np
import pandas as pd


MIN_INTERACTIONS = 3


def _use_leave_one_out(dataset: str) -> bool:
    """Check if dataset should use leave-one-out evaluation."""
    return dataset.endswith("_loo")


def _use_user_cold_start(dataset: str) -> bool:
    """Check if dataset should use user cold-start split."""
    return dataset.endswith("_coldstart") or dataset.endswith("_user_cold_start")


def _strip_dataset_suffixes(dataset: str) -> str:
    """Strip all split-related suffixes from dataset name."""
    base = dataset
    for suffix in ["_loo", "_split", "_coldstart", "_user_cold_start"]:
        base = base.replace(suffix, "")
    return base


def _get_min_interactions(dataset: str) -> int:
    """Get minimum interactions threshold for dataset."""
    # Support groups datasets have dense synthetic data, use lower threshold
    base_dataset = _strip_dataset_suffixes(dataset)
    if base_dataset.startswith("support_groups_full_"):
        return 3
    return MIN_INTERACTIONS


def _load_raw_datasets(dataset: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw interaction data and item metadata for supported datasets.

    Supported datasets:
    - support_groups_full_*: User-group interactions with support group features
    """
    # Handle dataset name suffixes (_loo, _split, _coldstart)
    base_dataset = _strip_dataset_suffixes(dataset)
    base = Path("datasets") / base_dataset

    if base_dataset.startswith("support_groups_full_"):
        interactions = pd.read_csv(base / "interactions.csv")
        groups = pd.read_csv(base / "groups.csv")

        # Add rating column (use AlignFeatures as rating if not present)
        if 'rating' not in interactions.columns and 'AlignFeatures' in interactions.columns:
            # Convert AlignFeatures [0,1] to rating scale [1,5]
            interactions['rating'] = 1 + (interactions['AlignFeatures'] * 4)

        interactions = interactions.rename(
            columns={
                "user_id": "orig_user",
                "group_id": "orig_item",
            }
        )
        groups = groups.rename(columns={"group_id": "orig_item"})
        return interactions, groups

    else:
        raise ValueError(f"Unsupported dataset: {base_dataset}. "
                        f"Supported datasets: support_groups_full_*")


def _filter_and_map_interactions(
    ratings: pd.DataFrame,
    items: pd.DataFrame,
    min_interactions: int = MIN_INTERACTIONS,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int], Dict[str, int]]:
    # Filter to users with sufficient interactions
    user_counts = ratings.groupby("orig_user").size()
    valid_users = user_counts[user_counts >= min_interactions].index
    filtered = ratings[ratings["orig_user"].isin(valid_users)].copy()

    # Build dense id mapping
    user_order = np.sort(valid_users)
    uid_map = {user: idx for idx, user in enumerate(user_order)}
    filtered["userId"] = filtered["orig_user"].map(uid_map)

    valid_items = filtered["orig_item"].unique()
    item_order = np.sort(valid_items)
    iid_map = {item: idx for idx, item in enumerate(item_order)}
    filtered["itemId"] = filtered["orig_item"].map(iid_map)

    items = items[items["orig_item"].isin(valid_items)].copy()
    items["itemId"] = items["orig_item"].map(iid_map)

    return filtered, items, uid_map, iid_map


def _assign_stratified_split(
    ratings: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Assign stratified train/val/test splits maintaining class balance.

    For balanced datasets (v7), we want each split to have the same positive/negative
    ratio as the full dataset. This ensures fair evaluation without ceiling effects.

    Args:
        ratings: DataFrame with userId, itemId, rating columns
        train_ratio: Proportion for training (default 0.70)
        val_ratio: Proportion for validation (default 0.15)
        test_ratio: Proportion for test (default 0.15)
        seed: Random seed for reproducibility

    Returns:
        train_df, val_df, test_df with maintained class balance
    """
    rng = np.random.RandomState(seed)

    # Binarize ratings (>=4 is positive, <=3 is negative)
    ratings = ratings.copy()
    ratings["is_positive"] = (ratings["rating"] >= 4).astype(int)

    # Split by class to maintain balance
    positive_interactions = ratings[ratings["is_positive"] == 1].copy()
    negative_interactions = ratings[ratings["is_positive"] == 0].copy()

    def _split_by_ratio(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split DataFrame into train/val/test maintaining ratio."""
        n = len(df)
        indices = rng.permutation(n)

        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

        return (
            df.iloc[train_idx].copy(),
            df.iloc[val_idx].copy(),
            df.iloc[test_idx].copy(),
        )

    # Split positive and negative separately
    pos_train, pos_val, pos_test = _split_by_ratio(positive_interactions)
    neg_train, neg_val, neg_test = _split_by_ratio(negative_interactions)

    # Combine positive and negative for each split
    train_df = pd.concat([pos_train, neg_train], ignore_index=True)
    val_df = pd.concat([pos_val, neg_val], ignore_index=True)
    test_df = pd.concat([pos_test, neg_test], ignore_index=True)

    # Shuffle each split
    train_df = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    val_df = val_df.sample(frac=1, random_state=seed + 1).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=seed + 2).reset_index(drop=True)

    # Remove helper column
    train_df = train_df.drop(columns=["is_positive"])
    val_df = val_df.drop(columns=["is_positive"])
    test_df = test_df.drop(columns=["is_positive"])

    return train_df, val_df, test_df


def _assign_leave_one_out_split(ratings: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if "timestamp" in ratings.columns:
        sort_cols = ["userId", "timestamp", "itemId"]
    else:
        ratings = ratings.reset_index(drop=True)
        ratings["sequence_index"] = ratings.index
        sort_cols = ["userId", "sequence_index", "itemId"]

    sorted_df = ratings.sort_values(sort_cols)
    sorted_df["position"] = sorted_df.groupby("userId").cumcount()
    max_position = sorted_df.groupby("userId")["position"].transform("max")
    sorted_df["rev_position"] = max_position - sorted_df["position"]

    test_df = sorted_df[sorted_df["rev_position"] == 0].drop(columns=["position", "rev_position"])
    val_df = sorted_df[sorted_df["rev_position"] == 1].drop(columns=["position", "rev_position"])
    train_df = sorted_df[sorted_df["rev_position"] >= 2].drop(columns=["position", "rev_position"])

    helper_cols = ["sequence_index"]
    missing_cols = [col for col in helper_cols if col in train_df.columns]
    if missing_cols:
        train_df = train_df.drop(columns=missing_cols)
        val_df = val_df.drop(columns=[c for c in missing_cols if c in val_df.columns])
        test_df = test_df.drop(columns=[c for c in missing_cols if c in test_df.columns])

    return train_df, val_df, test_df


def _assign_user_cold_start_split(
    ratings: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Set[int]]]:
    """
    Assign user-level cold-start split where train/val/test are disjoint by user.
    
    No test user (and no validation user) appears in training at all.
    This simulates the cold-start scenario where we must recommend to new users.
    
    Args:
        ratings: DataFrame with userId, itemId, rating columns
        train_ratio: Proportion of users for training (default 0.70)
        val_ratio: Proportion of users for validation (default 0.15)
        test_ratio: Proportion of users for test (default 0.15)
        seed: Random seed for reproducibility
        
    Returns:
        train_df, val_df, test_df: Split DataFrames
        user_splits: Dict with 'train', 'val', 'test' user ID sets
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
    
    rng = np.random.RandomState(seed)
    
    # Get unique users and shuffle
    unique_users = ratings['userId'].unique()
    n_users = len(unique_users)
    shuffled_users = rng.permutation(unique_users)
    
    # Split users into train/val/test
    n_train = int(n_users * train_ratio)
    n_val = int(n_users * val_ratio)
    
    train_users = set(shuffled_users[:n_train])
    val_users = set(shuffled_users[n_train:n_train + n_val])
    test_users = set(shuffled_users[n_train + n_val:])
    
    # Verify disjointness with assertions
    assert len(train_users & val_users) == 0, \
        f"Train/Val overlap: {len(train_users & val_users)} users"
    assert len(train_users & test_users) == 0, \
        f"Train/Test overlap: {len(train_users & test_users)} users"
    assert len(val_users & test_users) == 0, \
        f"Val/Test overlap: {len(val_users & test_users)} users"
    assert len(train_users) + len(val_users) + len(test_users) == n_users, \
        f"User count mismatch: {len(train_users)} + {len(val_users)} + {len(test_users)} != {n_users}"
    
    print(f"User Cold-Start Split (seed={seed}):")
    print(f"  - Train users: {len(train_users)} ({100*len(train_users)/n_users:.1f}%)")
    print(f"  - Val users: {len(val_users)} ({100*len(val_users)/n_users:.1f}%)")
    print(f"  - Test users: {len(test_users)} ({100*len(test_users)/n_users:.1f}%)")
    print(f"  - Disjointness verified ✓")
    
    # Filter interactions by user split
    train_df = ratings[ratings['userId'].isin(train_users)].copy()
    val_df = ratings[ratings['userId'].isin(val_users)].copy()
    test_df = ratings[ratings['userId'].isin(test_users)].copy()
    
    # Shuffle each split
    train_df = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    val_df = val_df.sample(frac=1, random_state=seed + 1).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=seed + 2).reset_index(drop=True)
    
    print(f"  - Train interactions: {len(train_df)}")
    print(f"  - Val interactions: {len(val_df)}")
    print(f"  - Test interactions: {len(test_df)}")
    
    user_splits = {
        'train': train_users,
        'val': val_users,
        'test': test_users
    }
    
    return train_df, val_df, test_df, user_splits


def verify_user_cold_start_disjointness(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> bool:
    """
    Verify that train/val/test splits have no overlapping users.
    
    Returns:
        True if splits are disjoint by user, raises AssertionError otherwise
    """
    train_users = set(train_df['userId'].unique())
    val_users = set(val_df['userId'].unique())
    test_users = set(test_df['userId'].unique())
    
    train_val_overlap = train_users & val_users
    train_test_overlap = train_users & test_users
    val_test_overlap = val_users & test_users
    
    assert len(train_val_overlap) == 0, \
        f"Train/Val user overlap: {len(train_val_overlap)} users: {list(train_val_overlap)[:5]}..."
    assert len(train_test_overlap) == 0, \
        f"Train/Test user overlap: {len(train_test_overlap)} users: {list(train_test_overlap)[:5]}..."
    assert len(val_test_overlap) == 0, \
        f"Val/Test user overlap: {len(val_test_overlap)} users: {list(val_test_overlap)[:5]}..."
    
    print("✓ User cold-start disjointness verified:")
    print(f"  - Train users: {len(train_users)}")
    print(f"  - Val users: {len(val_users)}")
    print(f"  - Test users: {len(test_users)}")
    print(f"  - No overlaps found")
    
    return True


def prepare_leave_one_out(dataset: str, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    """Return deterministic train/val/test splits for the requested dataset.
    
    Supports multiple split strategies based on dataset name suffix:
    - _loo: Leave-one-out split (by interaction timestamp)
    - _coldstart or _user_cold_start: User cold-start split (disjoint by user)
    - default: Stratified split (70/15/15 by interactions)
    
    Args:
        dataset: Dataset name with optional suffix (_loo, _coldstart)
        seed: Random seed for user_cold_start split (default: 42)
    
    Returns:
        train_df, val_df, test_df, items_df, metadata
    """

    splits_dir = Path("datasets") / "splits" / dataset
    train_path = splits_dir / "train.csv"
    val_path = splits_dir / "val.csv"
    test_path = splits_dir / "test.csv"
    items_path = splits_dir / "items.csv"
    meta_path = splits_dir / "meta.json"
    user_splits_path = splits_dir / "user_splits.json"

    # For user cold-start, include seed in cache path to allow different seeds
    if _use_user_cold_start(dataset):
        splits_dir = Path("datasets") / "splits" / f"{dataset}_seed{seed}"
        train_path = splits_dir / "train.csv"
        val_path = splits_dir / "val.csv"
        test_path = splits_dir / "test.csv"
        items_path = splits_dir / "items.csv"
        meta_path = splits_dir / "meta.json"
        user_splits_path = splits_dir / "user_splits.json"

    if train_path.exists() and val_path.exists() and test_path.exists() and items_path.exists() and meta_path.exists():
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)
        items_df = pd.read_csv(items_path)
        with meta_path.open("r") as fh:
            metadata = json.load(fh)
        
        # Verify cold-start disjointness on load
        if _use_user_cold_start(dataset):
            verify_user_cold_start_disjointness(train_df, val_df, test_df)
        
        return train_df, val_df, test_df, items_df, metadata

    min_interactions = _get_min_interactions(dataset)
    ratings_raw, items_raw = _load_raw_datasets(dataset)
    mapped_ratings, mapped_items, uid_map, iid_map = _filter_and_map_interactions(
        ratings_raw, items_raw, min_interactions=min_interactions
    )

    # Determine split method based on dataset
    user_splits = None
    if _use_user_cold_start(dataset):
        print(f"Using USER COLD-START split for {dataset} (seed={seed})")
        train_df, val_df, test_df, user_splits = _assign_user_cold_start_split(
            mapped_ratings, seed=seed
        )
        # Verify disjointness
        verify_user_cold_start_disjointness(train_df, val_df, test_df)
    elif _use_leave_one_out(dataset):
        print(f"Using LEAVE-ONE-OUT split for {dataset}")
        train_df, val_df, test_df = _assign_leave_one_out_split(mapped_ratings)
    else:
        print(f"Using TRAIN/VAL/TEST (70/15/15) stratified split for {dataset}")
        train_df, val_df, test_df = _assign_stratified_split(mapped_ratings)

    splits_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    mapped_items.to_csv(items_path, index=False)

    metadata = {
        "dataset": dataset,
        "num_users": len(uid_map),
        "num_items": len(iid_map),
        "min_interactions": min_interactions,
        "columns": list(train_df.columns),
        "split_type": "user_cold_start" if _use_user_cold_start(dataset) else (
            "leave_one_out" if _use_leave_one_out(dataset) else "stratified"
        ),
        "seed": seed if _use_user_cold_start(dataset) else None,
    }

    with meta_path.open("w") as fh:
        json.dump(metadata, fh, indent=2)
    
    # Save user splits for cold-start
    if user_splits is not None:
        with user_splits_path.open("w") as fh:
            # Convert numpy int64 to Python int for JSON serialization
            json.dump({
                'train_users': [int(u) for u in user_splits['train']],
                'val_users': [int(u) for u in user_splits['val']],
                'test_users': [int(u) for u in user_splits['test']],
            }, fh, indent=2)

    return train_df, val_df, test_df, mapped_items, metadata
