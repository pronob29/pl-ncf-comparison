#!/usr/bin/env python3
"""
support_groups_utils.py
=======================

Utilities for loading and processing Support Groups dataset with survey features.

This module provides functions to:
1. Load user survey features (q33, q26 responses)
2. Load group feature weights (g_w1-10)
3. Generate pseudo-labels from cosine similarity
4. Create user-group compatibility matrices
"""

import pandas as pd
import numpy as np
import torch
from typing import Dict, Tuple, Optional
from pathlib import Path


def load_user_features(dataset_path: str) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Load user survey features from users.csv.

    Args:
        dataset_path: Path to dataset directory (e.g., "datasets/support_groups_original")

    Returns:
        user_features: Array of shape [num_users, feature_dim]
        user_id_map: Dictionary mapping original user_id to dense index
    """
    users_df = pd.read_csv(Path(dataset_path) / "users.csv")

    # Extract q33 and q26 features
    q33_cols = [f'q33_w{i}' for i in range(1, 7)]  # 6 features
    q26_cols = [f'q26_w{i}' for i in range(1, 11)]  # 10 features

    feature_cols = q33_cols + q26_cols  # Total: 16 features

    # Create user ID mapping
    user_id_map = {uid: idx for idx, uid in enumerate(users_df['user_id'].values)}

    # Extract features
    user_features = users_df[feature_cols].values.astype(np.float32)

    # Normalize features (important for cosine similarity)
    norms = np.linalg.norm(user_features, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # Avoid division by zero
    user_features = user_features / norms

    print(f"Loaded {len(user_features)} user feature vectors (dim={user_features.shape[1]})")

    return user_features, user_id_map


def load_group_features(dataset_path: str) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Load group feature weights from groups.csv.

    Args:
        dataset_path: Path to dataset directory

    Returns:
        group_features: Array of shape [num_groups, feature_dim]
        group_id_map: Dictionary mapping original group_id to dense index
    """
    groups_df = pd.read_csv(Path(dataset_path) / "groups.csv")

    # Extract g_w features
    g_w_cols = [f'g_w{i}' for i in range(1, 11)]  # 10 features

    # Create group ID mapping
    group_id_map = {gid: idx for idx, gid in enumerate(groups_df['group_id'].values)}

    # Extract features
    group_features = groups_df[g_w_cols].values.astype(np.float32)

    # Normalize features
    norms = np.linalg.norm(group_features, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    group_features = group_features / norms

    print(f"Loaded {len(group_features)} group feature vectors (dim={group_features.shape[1]})")

    return group_features, group_id_map


def generate_pseudo_labels_from_features(
    user_features: np.ndarray,
    group_features: np.ndarray,
    temperature: float = 1.0,
    confidence_threshold: float = 0.7
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate pseudo-labels using cosine similarity between user and group features.

    Args:
        user_features: Normalized user feature vectors [num_users, user_dim]
        group_features: Normalized group feature vectors [num_groups, group_dim]
        temperature: Temperature for softmax normalization
        confidence_threshold: Minimum similarity score to consider

    Returns:
        pseudo_labels: Similarity matrix [num_users, num_groups]
        confidence_mask: Boolean mask indicating confident predictions
    """
    num_users = user_features.shape[0]
    num_groups = group_features.shape[0]

    # Pad features to same dimension (or use learned projection)
    user_dim = user_features.shape[1]
    group_dim = group_features.shape[1]

    if user_dim > group_dim:
        # Pad group features
        padding = np.zeros((num_groups, user_dim - group_dim), dtype=np.float32)
        group_features = np.concatenate([group_features, padding], axis=1)
    elif group_dim > user_dim:
        # Pad user features
        padding = np.zeros((num_users, group_dim - user_dim), dtype=np.float32)
        user_features = np.concatenate([user_features, padding], axis=1)

    # Compute cosine similarity: already normalized, so just dot product
    pseudo_labels = user_features @ group_features.T  # [num_users, num_groups]

    # Apply temperature scaling
    pseudo_labels = pseudo_labels / temperature

    # Create confidence mask
    confidence_mask = pseudo_labels >= confidence_threshold

    # Apply softmax normalization per user
    exp_labels = np.exp(pseudo_labels)
    pseudo_labels = exp_labels / (exp_labels.sum(axis=1, keepdims=True) + 1e-8)

    print(f"Generated pseudo-labels: shape={pseudo_labels.shape}")
    print(f"Confident predictions: {confidence_mask.sum()} / {confidence_mask.size} "
          f"({100*confidence_mask.sum()/confidence_mask.size:.1f}%)")
    print(f"Pseudo-label statistics: min={pseudo_labels.min():.4f}, "
          f"max={pseudo_labels.max():.4f}, mean={pseudo_labels.mean():.4f}")

    return pseudo_labels.astype(np.float32), confidence_mask


def load_ground_truth_similarities(dataset_path: str) -> Tuple[Dict, Dict]:
    """
    Load ground truth AlignFeatures and SimCondition from interactions.

    Args:
        dataset_path: Path to dataset directory

    Returns:
        align_features: Dict[(user_id, group_id)] -> align_score
        sim_condition: Dict[(user_id, group_id)] -> sim_score
    """
    interactions_df = pd.read_csv(Path(dataset_path) / "interactions.csv")

    align_features = {}
    sim_condition = {}

    for _, row in interactions_df.iterrows():
        key = (int(row['user_id']), int(row['group_id']))
        align_features[key] = float(row['AlignFeatures'])
        sim_condition[key] = float(row['SimCondition'])

    print(f"Loaded ground truth similarities for {len(align_features)} interactions")

    return align_features, sim_condition


def create_enhanced_pseudo_labels(
    dataset_path: str,
    use_ground_truth: bool = True,
    temperature: float = 1.0,
    confidence_threshold: float = 0.7,
    amplification_power: float = 10.0,
    use_negative_sampling: bool = True,
    preserve_ranking: bool = True,
    use_relative_labels: bool = True,
    negative_ratio: int = 7
) -> Dict:
    """
    Create comprehensive pseudo-label package for PL models.

    Args:
        dataset_path: Path to dataset directory
        use_ground_truth: If True, use AlignFeatures as pseudo-labels
        temperature: Temperature for feature-based pseudo-labels
        confidence_threshold: Confidence threshold
        amplification_power: Power to amplify small differences in AlignFeatures
        use_negative_sampling: If True, add negative pseudo-labels for non-interactions

    Returns:
        Dictionary containing:
        - pseudo_label_matrix: [num_users, num_groups] tensor
        - confidence_mask: Boolean mask
        - user_features: User feature vectors
        - group_features: Group feature vectors
    """
    # Load features
    user_features, user_id_map = load_user_features(dataset_path)
    group_features, group_id_map = load_group_features(dataset_path)

    if use_ground_truth:
        # Use AlignFeatures as ground truth pseudo-labels
        align_features, sim_condition = load_ground_truth_similarities(dataset_path)

        num_users = len(user_id_map)
        num_groups = len(group_id_map)

        # Initialize pseudo-label matrix
        pseudo_labels = np.zeros((num_users, num_groups), dtype=np.float32)
        confidence_mask = np.zeros((num_users, num_groups), dtype=bool)

        # Collect all AlignFeatures values for normalization
        align_values = np.array(list(align_features.values()))

        # Transform AlignFeatures to amplify small differences while preserving ranking
        align_min = align_values.min()
        align_max = align_values.max()

        print(f"Original AlignFeatures: min={align_min:.4f}, max={align_max:.4f}, std={align_values.std():.4f}")

        # Fill in positive interactions with transformed scores
        transformed_scores = []
        user_groups_map = {}  # Track which groups each user interacted with

        for (orig_uid, orig_gid), align_score in align_features.items():
            if orig_uid in user_id_map and orig_gid in group_id_map:
                uid = user_id_map[orig_uid]
                gid = group_id_map[orig_gid]

                # Min-max normalize to [0, 1]
                normalized = (align_score - align_min) / (align_max - align_min + 1e-8)

                if preserve_ranking:
                    # Use softer amplification to preserve fine-grained ranking
                    # Lower power = softer transformation = more gradation
                    effective_power = max(amplification_power / 3.0, 3.0)
                    transformed = normalized ** (1.0 / effective_power)
                else:
                    # Original aggressive amplification
                    transformed = normalized ** (1.0 / amplification_power)

                # Store original align score for relative ranking
                if uid not in user_groups_map:
                    user_groups_map[uid] = []
                user_groups_map[uid].append((gid, align_score, transformed))

                pseudo_labels[uid, gid] = transformed
                confidence_mask[uid, gid] = True
                transformed_scores.append(transformed)

        transformed_scores = np.array(transformed_scores)
        print(f"Transformed pseudo-labels: min={transformed_scores.min():.4f}, max={transformed_scores.max():.4f}, std={transformed_scores.std():.4f}")
        print(f"Amplification increased variance by {transformed_scores.std() / align_values.std():.2f}x")

        # Use relative (ranking-aware) pseudo-labels per user
        if use_relative_labels:
            print("Converting to relative rankings within each user...")
            for uid, group_scores in user_groups_map.items():
                if len(group_scores) > 1:
                    # Sort by original AlignFeatures score
                    sorted_groups = sorted(group_scores, key=lambda x: x[1], reverse=True)

                    # Create ranking-based scores: best=1.0, worst=0.0, linear interpolation
                    n_groups = len(sorted_groups)
                    for rank, (gid, _, _) in enumerate(sorted_groups):
                        # Linear ranking: rank 0 (best) -> 1.0, rank n-1 (worst) -> 0.5
                        # This preserves relative ordering while keeping positives > 0.5
                        ranking_score = 1.0 - (rank / (2 * n_groups))
                        pseudo_labels[uid, gid] = ranking_score

            # Recompute statistics
            confident_values = pseudo_labels[confidence_mask]
            print(f"After relative ranking: min={confident_values.min():.4f}, max={confident_values.max():.4f}, std={confident_values.std():.4f}")

        # Add negative pseudo-labels for non-interactions
        if use_negative_sampling:
            # For each user, add low-confidence negative labels for non-interacted groups
            neg_label_value = 0.0  # Low score for negative samples
            neg_count = 0

            for uid in range(num_users):
                # Get interacted groups
                interacted = confidence_mask[uid].nonzero()[0]

                # Sample some non-interacted groups as negatives
                all_groups = np.arange(num_groups)
                non_interacted = np.setdiff1d(all_groups, interacted)

                if len(non_interacted) > 0:
                    # Sample up to negative_ratio x the positive samples as negatives
                    # v11: Increased from 3x to 7x for stronger PL signal
                    n_negatives = min(len(non_interacted), len(interacted) * negative_ratio)
                    neg_samples = np.random.choice(non_interacted, size=n_negatives, replace=False)

                    for gid in neg_samples:
                        pseudo_labels[uid, gid] = neg_label_value
                        confidence_mask[uid, gid] = True
                        neg_count += 1

            print(f"Added {neg_count} negative pseudo-labels (ratio: {neg_count}/{confidence_mask.sum()-neg_count})")

        print(f"Using ground truth AlignFeatures as pseudo-labels (amplified)")
        print(f"Total confident predictions: {confidence_mask.sum()}")
    else:
        # Generate pseudo-labels from features
        pseudo_labels, confidence_mask = generate_pseudo_labels_from_features(
            user_features, group_features, temperature, confidence_threshold
        )

    # Convert to torch tensors
    return {
        'pseudo_label_matrix': torch.from_numpy(pseudo_labels),
        'confidence_mask': torch.from_numpy(confidence_mask),
        'user_features': torch.from_numpy(user_features),
        'group_features': torch.from_numpy(group_features),
        'user_id_map': user_id_map,
        'group_id_map': group_id_map,
        'num_users': len(user_id_map),
        'num_groups': len(group_id_map)
    }


def create_user_content_embeddings_sg(
    dataset_path: str,
    use_features: bool = True
) -> Tuple[Dict[int, torch.Tensor], int]:
    """
    Create user content embeddings for Support Groups dataset.

    Args:
        dataset_path: Path to dataset directory
        use_features: If True, use actual survey features; else use dummy

    Returns:
        user_content_embed: Dict[userId, feature_tensor]
        embed_dim: Dimension of embeddings
    """
    if use_features:
        user_features, user_id_map = load_user_features(dataset_path)

        user_content_embed = {
            idx: torch.from_numpy(feat)
            for idx, feat in enumerate(user_features)
        }

        embed_dim = user_features.shape[1]

        print(f"Created user content embeddings from survey features (dim={embed_dim})")
    else:
        # Dummy embeddings (fallback)
        user_content_embed = {}
        embed_dim = 16

        print(f"Created dummy user content embeddings (dim={embed_dim})")

    return user_content_embed, embed_dim
