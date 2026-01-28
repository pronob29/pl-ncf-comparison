'''
data.py
'''

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random
from typing import Dict, Optional, List, Set

class OnTheFlyNCFDataset(Dataset):
    """
    Efficient NCF Dataset: yields (user, item, label) with on-the-fly negative sampling.
    """
    def __init__(
        self,
        ratings: pd.DataFrame,
        num_items: int,
        num_negatives: int = 1,
        seed: Optional[int] = None,
    ):
        self.ratings = ratings.reset_index(drop=True)
        self.num_items = num_items
        self.num_negatives = num_negatives
        self._rng = random.Random(seed) if seed is not None else random.Random()

        self.user_positive_items = ratings.groupby('userId')['itemId'].apply(set).to_dict()
        self.users = self.ratings['userId'].values
        self.items = self.ratings['itemId'].values
        self.labels = self.ratings['rating'].values

    def __len__(self):
        return len(self.ratings) * (1 + self.num_negatives)

    def __getitem__(self, idx):
        pos_idx = idx // (1 + self.num_negatives)
        sub_idx = idx % (1 + self.num_negatives)
        user = self.users[pos_idx]
        if sub_idx == 0:
            item = self.items[pos_idx]
            label = 1.0
        else:
            while True:
                neg_item = self._rng.randint(0, self.num_items - 1)
                if neg_item not in self.user_positive_items[user]:
                    break
            item = neg_item
            label = 0.0
        return torch.LongTensor([user]), torch.LongTensor([item]), torch.FloatTensor([label])

def get_dataloader(
    ratings_df,
    num_items,
    batch_size=256,
    num_negatives=1,
    shuffle=True,
    seed: Optional[int] = None,
    num_workers: int = 0,  # Changed from 4 to 0 to avoid multiprocessing deadlocks
):
    """
    Create DataLoader for training.
    
    Note: num_workers set to 0 by default to avoid multiprocessing issues
    with on-the-fly negative sampling. For large datasets, the bottleneck
    is typically in TF-IDF/model computation, not data loading.
    """
    dataset = OnTheFlyNCFDataset(ratings_df, num_items, num_negatives, seed=seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    return loader


# =============================================================================
# Hard Negative Sampling Dataset
# =============================================================================

class HardNegativeNCFDataset(Dataset):
    """
    NCF Dataset with hard negative sampling based on teacher similarity.
    
    For each positive, samples negatives from top-M teacher-similar groups
    (excluding positives), forcing the model to learn fine-grained distinctions.
    
    This addresses the limitation of uniform random negatives which are often
    "too easy" - items that are obviously dissimilar to the user.
    
    Args:
        ratings: DataFrame with userId, itemId, rating columns
        num_items: Total number of items
        teacher_similarity: [num_users, num_items] teacher similarity matrix
        num_negatives: Number of negative samples per positive
        hard_negative_pool_size: Top M similar non-positives to sample from
        seed: Random seed for reproducibility
    """
    
    def __init__(
        self,
        ratings: pd.DataFrame,
        num_items: int,
        teacher_similarity: np.ndarray,
        num_negatives: int = 1,
        hard_negative_pool_size: int = 50,
        seed: Optional[int] = None,
    ):
        self.ratings = ratings.reset_index(drop=True)
        self.num_items = num_items
        self.num_negatives = num_negatives
        self.hard_negative_pool_size = hard_negative_pool_size
        self._rng = random.Random(seed) if seed is not None else random.Random()
        self.teacher_similarity = np.asarray(teacher_similarity)
        
        # Build positive item sets per user
        self.user_positive_items = ratings.groupby('userId')['itemId'].apply(set).to_dict()
        self.users = self.ratings['userId'].values
        self.items = self.ratings['itemId'].values
        
        # Precompute hard negative pools for efficiency
        self._build_hard_negative_pools()
    
    def _build_hard_negative_pools(self):
        """Precompute top-M similar non-positive items for each user."""
        self.hard_negative_pools: Dict[int, np.ndarray] = {}
        unique_users = np.unique(self.users)
        
        print(f"Building hard negative pools for {len(unique_users)} users...")
        
        for user_id in unique_users:
            positives = self.user_positive_items.get(user_id, set())
            user_sims = self.teacher_similarity[user_id]  # [num_items]
            
            # Get indices of non-positive items
            all_items = np.arange(self.num_items)
            non_positive_mask = ~np.isin(all_items, list(positives))
            non_positive_items = all_items[non_positive_mask]
            non_positive_sims = user_sims[non_positive_mask]
            
            # Top M similar non-positives (hard negatives)
            if len(non_positive_items) > 0:
                pool_size = min(self.hard_negative_pool_size, len(non_positive_items))
                top_m_indices = np.argsort(non_positive_sims)[-pool_size:]
                self.hard_negative_pools[user_id] = non_positive_items[top_m_indices]
            else:
                self.hard_negative_pools[user_id] = np.array([], dtype=np.int64)
        
        print(f"Hard negative pools built. Avg pool size: "
              f"{np.mean([len(p) for p in self.hard_negative_pools.values()]):.1f}")
    
    def __len__(self):
        return len(self.ratings) * (1 + self.num_negatives)
    
    def __getitem__(self, idx):
        pos_idx = idx // (1 + self.num_negatives)
        sub_idx = idx % (1 + self.num_negatives)
        user = self.users[pos_idx]
        
        if sub_idx == 0:
            # Positive sample
            item = self.items[pos_idx]
            label = 1.0
        else:
            # Hard negative sample
            pool = self.hard_negative_pools.get(user, None)
            if pool is not None and len(pool) > 0:
                item = int(self._rng.choice(pool))
            else:
                # Fallback to uniform sampling if no hard negatives available
                while True:
                    item = self._rng.randint(0, self.num_items - 1)
                    if item not in self.user_positive_items.get(user, set()):
                        break
            label = 0.0
        
        return torch.LongTensor([user]), torch.LongTensor([item]), torch.FloatTensor([label])


def get_hard_negative_dataloader(
    ratings_df: pd.DataFrame,
    num_items: int,
    teacher_similarity: np.ndarray,
    batch_size: int = 256,
    num_negatives: int = 1,
    hard_negative_pool_size: int = 50,
    shuffle: bool = True,
    seed: Optional[int] = None,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create DataLoader with hard negative sampling.
    
    Args:
        ratings_df: Training DataFrame with userId, itemId, rating
        num_items: Total number of items
        teacher_similarity: [num_users, num_items] teacher similarity matrix
        batch_size: Batch size
        num_negatives: Negatives per positive
        hard_negative_pool_size: Top-M similar items to sample negatives from
        shuffle: Whether to shuffle
        seed: Random seed
        num_workers: DataLoader workers
    
    Returns:
        DataLoader with hard negative sampling
    """
    dataset = HardNegativeNCFDataset(
        ratings_df, num_items, teacher_similarity,
        num_negatives, hard_negative_pool_size, seed
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


# =============================================================================
# Inductive Dataset (for cold-start models)
# =============================================================================

class InductiveNCFDataset(Dataset):
    """
    NCF Dataset for inductive models that use user features instead of IDs.
    
    Yields (user_features, item_id, label) tuples for training inductive
    models that can generalize to unseen users.
    
    Args:
        ratings: DataFrame with userId, itemId, rating columns
        user_features: [num_users, feature_dim] user feature matrix
        num_items: Total number of items
        num_negatives: Number of negative samples per positive
        seed: Random seed for reproducibility
    """
    
    def __init__(
        self,
        ratings: pd.DataFrame,
        user_features: np.ndarray,
        num_items: int,
        num_negatives: int = 1,
        seed: Optional[int] = None,
    ):
        self.ratings = ratings.reset_index(drop=True)
        self.user_features = torch.tensor(user_features, dtype=torch.float32)
        self.num_items = num_items
        self.num_negatives = num_negatives
        self._rng = random.Random(seed) if seed is not None else random.Random()
        
        self.user_positive_items = ratings.groupby('userId')['itemId'].apply(set).to_dict()
        self.users = self.ratings['userId'].values
        self.items = self.ratings['itemId'].values
        
        self.feature_dim = self.user_features.shape[1]
    
    def __len__(self):
        return len(self.ratings) * (1 + self.num_negatives)
    
    def __getitem__(self, idx):
        pos_idx = idx // (1 + self.num_negatives)
        sub_idx = idx % (1 + self.num_negatives)
        user_id = self.users[pos_idx]
        
        if sub_idx == 0:
            item = self.items[pos_idx]
            label = 1.0
        else:
            while True:
                item = self._rng.randint(0, self.num_items - 1)
                if item not in self.user_positive_items.get(user_id, set()):
                    break
            label = 0.0
        
        user_feat = self.user_features[user_id]  # [feature_dim]
        return user_feat, torch.LongTensor([item]), torch.FloatTensor([label])


class InductiveHardNegativeDataset(Dataset):
    """
    Inductive dataset with hard negative sampling.
    
    Combines user features (for inductive inference) with hard negatives
    (for better discrimination learning).
    """
    
    def __init__(
        self,
        ratings: pd.DataFrame,
        user_features: np.ndarray,
        num_items: int,
        teacher_similarity: np.ndarray,
        num_negatives: int = 1,
        hard_negative_pool_size: int = 50,
        seed: Optional[int] = None,
    ):
        self.ratings = ratings.reset_index(drop=True)
        self.user_features = torch.tensor(user_features, dtype=torch.float32)
        self.num_items = num_items
        self.num_negatives = num_negatives
        self.hard_negative_pool_size = hard_negative_pool_size
        self._rng = random.Random(seed) if seed is not None else random.Random()
        self.teacher_similarity = np.asarray(teacher_similarity)
        
        self.user_positive_items = ratings.groupby('userId')['itemId'].apply(set).to_dict()
        self.users = self.ratings['userId'].values
        self.items = self.ratings['itemId'].values
        
        self._build_hard_negative_pools()
    
    def _build_hard_negative_pools(self):
        """Build hard negative pools (same as HardNegativeNCFDataset)."""
        self.hard_negative_pools: Dict[int, np.ndarray] = {}
        unique_users = np.unique(self.users)
        
        for user_id in unique_users:
            positives = self.user_positive_items.get(user_id, set())
            user_sims = self.teacher_similarity[user_id]
            
            all_items = np.arange(self.num_items)
            non_positive_mask = ~np.isin(all_items, list(positives))
            non_positive_items = all_items[non_positive_mask]
            non_positive_sims = user_sims[non_positive_mask]
            
            if len(non_positive_items) > 0:
                pool_size = min(self.hard_negative_pool_size, len(non_positive_items))
                top_m_indices = np.argsort(non_positive_sims)[-pool_size:]
                self.hard_negative_pools[user_id] = non_positive_items[top_m_indices]
            else:
                self.hard_negative_pools[user_id] = np.array([], dtype=np.int64)
    
    def __len__(self):
        return len(self.ratings) * (1 + self.num_negatives)
    
    def __getitem__(self, idx):
        pos_idx = idx // (1 + self.num_negatives)
        sub_idx = idx % (1 + self.num_negatives)
        user_id = self.users[pos_idx]
        
        if sub_idx == 0:
            item = self.items[pos_idx]
            label = 1.0
        else:
            pool = self.hard_negative_pools.get(user_id, None)
            if pool is not None and len(pool) > 0:
                item = int(self._rng.choice(pool))
            else:
                while True:
                    item = self._rng.randint(0, self.num_items - 1)
                    if item not in self.user_positive_items.get(user_id, set()):
                        break
            label = 0.0
        
        user_feat = self.user_features[user_id]
        return user_feat, torch.LongTensor([item]), torch.FloatTensor([label])


def get_inductive_dataloader(
    ratings_df: pd.DataFrame,
    user_features: np.ndarray,
    num_items: int,
    batch_size: int = 256,
    num_negatives: int = 1,
    teacher_similarity: Optional[np.ndarray] = None,
    hard_negative_pool_size: int = 50,
    shuffle: bool = True,
    seed: Optional[int] = None,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create DataLoader for inductive models.
    
    Args:
        ratings_df: Training DataFrame
        user_features: [num_users, feature_dim] user feature matrix
        num_items: Total number of items
        batch_size: Batch size
        num_negatives: Negatives per positive
        teacher_similarity: If provided, use hard negatives
        hard_negative_pool_size: Pool size for hard negatives
        shuffle: Whether to shuffle
        seed: Random seed
        num_workers: DataLoader workers
    
    Returns:
        DataLoader for inductive training
    """
    if teacher_similarity is not None:
        dataset = InductiveHardNegativeDataset(
            ratings_df, user_features, num_items, teacher_similarity,
            num_negatives, hard_negative_pool_size, seed
        )
    else:
        dataset = InductiveNCFDataset(
            ratings_df, user_features, num_items, num_negatives, seed
        )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )

# ---- (Optional) User-content embedding for user text profiles ----

def build_user_content_embeddings(ratings: pd.DataFrame, books: pd.DataFrame, like_threshold=4.0, tfidf_max_feats=300):
    """
    Builds TF-IDF embedding per user from liked books' titles.
    Returns: Dict[userId, torch.Tensor], tfidf_dim (int)
    """
    import time
    start_time = time.time()
    
    print(f"[build_user_content_embeddings] Starting TF-IDF computation...")
    print(f"  - Total ratings: {len(ratings)}")
    print(f"  - Like threshold: {like_threshold}")
    print(f"  - TF-IDF max features: {tfidf_max_feats}")
    
    # Merge ratings with book titles for liked books (if not already merged)
    if "title" not in ratings.columns:
        merged = (
            ratings.merge(books[["itemId", "title"]], on="itemId", how="left")
            .query("rating >= @like_threshold")
        )
    else:
        # Title already present from previous merge
        merged = ratings.query("rating >= @like_threshold")
    
    print(f"  - Liked items: {len(merged)}")
    
    user_text = (
        merged.groupby("userId")["title"]
        .apply(lambda titles: " ".join(titles.astype(str)))
        .reset_index()
    )
    
    print(f"  - Users with liked items: {len(user_text)}")
    print(f"  - Computing TF-IDF vectors... (this may take a while for large datasets)")
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(max_features=tfidf_max_feats)
    tfidf_matrix = vectorizer.fit_transform(user_text["title"]).toarray()
    
    print(f"  - TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"  - Converting to PyTorch tensors...")
    
    user_content_embed: Dict[int, torch.Tensor] = {
        int(uid): torch.tensor(vec, dtype=torch.float32)
        for uid, vec in zip(user_text["userId"], tfidf_matrix)
    }
    tfidf_dim = tfidf_matrix.shape[1]
    
    elapsed = time.time() - start_time
    print(f"  ✅ TF-IDF computation completed in {elapsed:.2f}s")
    
    return user_content_embed, tfidf_dim


def generate_title_based_pseudo_labels(
    train_df: pd.DataFrame,
    items_df: pd.DataFrame,
    num_users: int,
    num_items: int,
    like_threshold: float = 4.0,
    tfidf_max_feats: int = 300,
    temperature: float = 1.0,
    confidence_threshold: float = 0.5,
    amplification_power: float = 3.0,
    use_negative_sampling: bool = True
) -> Dict:
    """
    Generate pseudo-labels for MovieLens/Goodbooks using title-based TF-IDF similarity.

    Algorithm:
    1. Build user profiles from liked items (rating >= like_threshold)
    2. Build item profiles from titles (TF-IDF)
    3. Compute cosine similarity: pseudo_label[u,i] = cos(user_profile[u], item_profile[i])
    4. Apply confidence thresholding and amplification

    Args:
        train_df: Training ratings DataFrame with userId, itemId, rating
        items_df: Items DataFrame with itemId, title
        num_users: Total number of users
        num_items: Total number of items
        like_threshold: Rating threshold for "liked" items (default 4.0)
        tfidf_max_feats: Max TF-IDF features (default 300)
        temperature: Temperature scaling parameter
        confidence_threshold: Minimum similarity to consider confident
        amplification_power: Power to amplify similarity scores
        use_negative_sampling: Whether to add negative pseudo-labels

    Returns:
        Dictionary with:
        - pseudo_label_matrix: [num_users, num_items] tensor
        - confidence_mask: Boolean mask for confident predictions
        - user_profiles: User TF-IDF profiles (dict)
        - item_profiles: Item TF-IDF profiles (dict)
    """
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer

    print(f"\n{'='*80}")
    print(f"GENERATING TITLE-BASED PSEUDO-LABELS")
    print(f"{'='*80}")

    # Step 1: Build item title profiles using TF-IDF
    print(f"\n[1/5] Building item title profiles...")
    if 'title' not in items_df.columns:
        raise ValueError("items_df must have 'title' column for pseudo-label generation")

    # Create TF-IDF vectorizer for item titles
    item_titles = items_df.set_index('itemId')['title'].fillna('').astype(str)
    vectorizer = TfidfVectorizer(max_features=tfidf_max_feats, min_df=1, stop_words='english')

    # Fit on all item titles
    all_titles = item_titles.tolist()
    item_tfidf_matrix = vectorizer.fit_transform(all_titles).toarray().astype(np.float32)

    # Create item profile dictionary
    item_profiles = {
        int(item_id): item_tfidf_matrix[idx]
        for idx, item_id in enumerate(items_df['itemId'])
    }

    print(f"   Created {len(item_profiles)} item profiles (dim={item_tfidf_matrix.shape[1]})")

    # Step 2: Build user profiles from liked items
    print(f"\n[2/5] Building user profiles from liked items (rating >= {like_threshold})...")

    # Merge with item titles
    train_with_titles = train_df.merge(items_df[['itemId', 'title']], on='itemId', how='left')
    liked_items = train_with_titles[train_with_titles['rating'] >= like_threshold]

    # Aggregate liked titles per user
    user_texts = (
        liked_items.groupby('userId')['title']
        .apply(lambda titles: ' '.join(titles.fillna('').astype(str)))
        .to_dict()
    )

    # Build user TF-IDF profiles using same vectorizer
    user_profiles = {}
    for user_id in range(num_users):
        if user_id in user_texts and user_texts[user_id].strip():
            # Transform using fitted vectorizer
            user_tfidf = vectorizer.transform([user_texts[user_id]]).toarray()[0].astype(np.float32)
            user_profiles[user_id] = user_tfidf
        else:
            # User with no liked items or no valid titles
            user_profiles[user_id] = np.zeros(item_tfidf_matrix.shape[1], dtype=np.float32)

    print(f"   Created {len(user_profiles)} user profiles")
    print(f"   Users with liked items: {sum(1 for p in user_profiles.values() if p.sum() > 0)}")

    # Step 3: Compute cosine similarity matrix
    print(f"\n[3/5] Computing user-item cosine similarities...")
    print(f"   Matrix dimensions: {num_users} users × {num_items} items = {num_users * num_items:,} entries")
    print(f"   This may take several minutes for large datasets...")

    import time
    start_time = time.time()

    pseudo_labels = np.zeros((num_users, num_items), dtype=np.float32)
    confidence_mask = np.zeros((num_users, num_items), dtype=bool)

    # Build full user and item matrices for efficient computation
    print(f"   Building user matrix...")
    user_matrix = np.stack([user_profiles[uid] for uid in range(num_users)])
    
    print(f"   Building item matrix...")
    item_matrix = np.stack([item_profiles.get(iid, np.zeros(item_tfidf_matrix.shape[1], dtype=np.float32))
                           for iid in range(num_items)])

    # Normalize for cosine similarity
    print(f"   Normalizing matrices...")
    user_norms = np.linalg.norm(user_matrix, axis=1, keepdims=True)
    user_norms[user_norms == 0] = 1.0
    user_matrix_norm = user_matrix / user_norms

    item_norms = np.linalg.norm(item_matrix, axis=1, keepdims=True)
    item_norms[item_norms == 0] = 1.0
    item_matrix_norm = item_matrix / item_norms

    # Compute cosine similarity: [num_users, num_items]
    print(f"   Computing similarity matrix ({num_users}×{num_items})...")
    print(f"   (Matrix multiply: {user_matrix_norm.shape} @ {item_matrix_norm.T.shape})")
    similarity_matrix = user_matrix_norm @ item_matrix_norm.T
    
    elapsed = time.time() - start_time
    print(f"   ✅ Similarity computation completed in {elapsed:.2f}s")

    print(f"   Similarity matrix shape: {similarity_matrix.shape}")
    print(f"   Similarity range: [{similarity_matrix.min():.4f}, {similarity_matrix.max():.4f}]")
    print(f"   Mean similarity: {similarity_matrix.mean():.4f}")

    # Step 4: Apply confidence thresholding and amplification
    print(f"\n[4/5] Applying confidence thresholding (threshold={confidence_threshold})...")

    # Mark confident predictions
    confidence_mask = similarity_matrix >= confidence_threshold
    confident_count = confidence_mask.sum()

    print(f"   Confident predictions: {confident_count} / {confidence_mask.size} ({100*confident_count/confidence_mask.size:.2f}%)")

    # Apply temperature scaling and amplification
    print(f"\n[5/5] Applying amplification (power={amplification_power}, temperature={temperature})...")

    # Min-max normalize similarities to [0, 1]
    sim_min = similarity_matrix.min()
    sim_max = similarity_matrix.max()
    if sim_max > sim_min:
        normalized_sim = (similarity_matrix - sim_min) / (sim_max - sim_min)
    else:
        normalized_sim = similarity_matrix

    # Amplify using power transform
    amplified_sim = normalized_sim ** (1.0 / amplification_power)

    # Apply temperature scaling
    pseudo_labels = amplified_sim / temperature

    print(f"   Amplified pseudo-labels: [{pseudo_labels.min():.4f}, {pseudo_labels.max():.4f}]")
    print(f"   Mean: {pseudo_labels.mean():.4f}, Std: {pseudo_labels.std():.4f}")

    # Add negative samples if requested
    if use_negative_sampling:
        print(f"\n[6/5] Adding negative pseudo-labels...")

        # Get actual interactions from training data
        user_positive_items = train_df.groupby('userId')['itemId'].apply(set).to_dict()

        neg_count = 0
        for uid in range(num_users):
            positive_items = user_positive_items.get(uid, set())

            # Sample negative items (not in training)
            all_items = set(range(num_items))
            negative_items = all_items - positive_items

            if len(negative_items) > 0:
                # Sample up to 3x positive samples as negatives
                n_negatives = min(len(negative_items), len(positive_items) * 3)
                if n_negatives > 0:
                    neg_samples = np.random.choice(list(negative_items), size=n_negatives, replace=False)

                    for iid in neg_samples:
                        # Only add negative label if not already confident
                        if not confidence_mask[uid, iid]:
                            pseudo_labels[uid, iid] = 0.0
                            confidence_mask[uid, iid] = True
                            neg_count += 1

        print(f"   Added {neg_count} negative pseudo-labels")

    print(f"\n{'='*80}")
    print(f"PSEUDO-LABEL GENERATION COMPLETE")
    print(f"{'='*80}")
    print(f"Final statistics:")
    print(f"  Total confident predictions: {confidence_mask.sum()}")
    print(f"  Pseudo-label range: [{pseudo_labels.min():.4f}, {pseudo_labels.max():.4f}]")
    print(f"  Coverage: {100*confidence_mask.sum()/(num_users*num_items):.2f}% of user-item pairs")
    print(f"{'='*80}\n")

    return {
        'pseudo_label_matrix': torch.from_numpy(pseudo_labels),
        'confidence_mask': torch.from_numpy(confidence_mask),
        'user_profiles': {k: torch.from_numpy(v) for k, v in user_profiles.items()},
        'item_profiles': {k: torch.from_numpy(v) for k, v in item_profiles.items()},
        'num_users': num_users,
        'num_items': num_items
    }
