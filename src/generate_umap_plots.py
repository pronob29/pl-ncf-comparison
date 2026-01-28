#!/usr/bin/env python3
"""
Generate UMAP Visualizations of User/Item Embeddings
=====================================================

Creates 2D UMAP projections of learned embeddings for **visualization only**.
Cluster labels are always computed in the original high-dimensional (HD)
embedding space using spherical K-means, then overlaid on the UMAP projection.

**IMPORTANT**: UMAP is a non-linear dimensionality reduction technique that
distorts distances and densities. Clustering decisions and silhouette metrics
must ALWAYS be computed on the original HD embeddings, NOT on UMAP coordinates.
The 2D scatter plot is purely for visual inspection of cluster structure.

Features:
- UMAP dimensionality reduction (2D visualization only)
- Spherical K-means clustering (on HD embeddings, overlaid on UMAP)
- Dual silhouette metrics (cosine + euclidean, computed on HD embeddings)
- Publication-ready 300 DPI plots
- Support for both main and PL-branch representations
- Configurable entity (user/item) and representation (main/pl)

Usage:
    # Generate UMAP for specific model (default: user, main)
    python src/generate_umap_plots.py --model mf_baseline --dataset support_groups_full_164 --seed 42

    # Generate for PL representation of PL model
    python src/generate_umap_plots.py --model neumf_pl --dataset support_groups_full_164 --seed 42 --entity user --repr pl

    # Generate for all models
    python src/generate_umap_plots.py --all

Output:
    results/{dataset}/umap_plots/{model}_seed{seed}_{entity}_{repr}_umap.png
    results/{dataset}/umap_plots/{model}_seed{seed}_{entity}_{repr}_metrics.json
"""

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

# Suppress UMAP warnings
warnings.filterwarnings('ignore')


# =============================================================================
# Cluster Visualization Constants
# =============================================================================

# Marker shapes for cluster visualization (cycled if more clusters than markers)
# Deterministic assignment: unique labels are sorted, then mapped by index
CLUSTER_MARKERS = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*", "h", "8"]


# =============================================================================
# Dataset and Model Naming Utilities
# =============================================================================

# Canonical dataset ID → display name mapping
DATASET_DISPLAY_NAMES = {
    'support_groups_full_164': 'Support Groups (Stratified Split)',
    'support_groups_full_164_loo': 'Support Groups (Leave-One-Out)',
}

# Model variant descriptions
MODEL_VARIANTS = {
    'mf_baseline': ('Matrix Factorization', 'Baseline'),
    'mf_pl': ('Matrix Factorization', 'PL'),
    'mlp_baseline': ('MLP', 'Baseline'),
    'mlp_pl': ('MLP', 'PL'),
    'neumf_baseline': ('NeuMF', 'Baseline'),
    'neumf_pl': ('NeuMF', 'PL'),
}


def get_dataset_display_name(dataset_id: str) -> str:
    """
    Convert dataset ID to human-readable display name.

    Args:
        dataset_id: Internal dataset identifier (e.g., 'support_groups_full_164_loo')

    Returns:
        Human-readable display name with protocol information
    """
    return DATASET_DISPLAY_NAMES.get(dataset_id, dataset_id.replace('_', ' ').title())


def get_model_display_info(model_name: str) -> Tuple[str, str]:
    """
    Get model architecture and variant (Baseline/PL) from model name.

    Args:
        model_name: Model identifier (e.g., 'neumf_pl')

    Returns:
        Tuple of (architecture_name, variant_name)
    """
    return MODEL_VARIANTS.get(model_name, (model_name.replace('_', ' ').title(), ''))


def get_protocol_from_dataset(dataset_id: str) -> str:
    """
    Extract evaluation protocol from dataset ID.

    Args:
        dataset_id: Dataset identifier

    Returns:
        'LOO' for leave-one-out, 'Stratified' otherwise
    """
    return 'LOO' if '_loo' in dataset_id.lower() else 'Stratified'


def load_embeddings(
    embeddings_dir: Path,
    model_name: str,
    seed: int,
    entity: str = 'user',
    repr_type: str = 'main'
) -> Optional[np.ndarray]:
    """
    Load embeddings from .npy file.

    Args:
        embeddings_dir: Directory containing embedding files
        model_name: Model name (e.g., 'mf_baseline', 'neumf_pl')
        seed: Random seed
        entity: 'user' or 'item'
        repr_type: 'main' or 'pl'

    Returns:
        Embeddings array or None if not found
    """
    # Build filename based on entity and repr_type
    if repr_type == 'pl':
        # For PL representation, load from pl_user_emb.npy or pl_item_emb.npy
        embedding_path = embeddings_dir / f"{model_name}_seed{seed}_pl_{entity}_emb.npy"
    else:
        # For main representation, load from user_emb.npy or item_emb.npy
        embedding_path = embeddings_dir / f"{model_name}_seed{seed}_{entity}_emb.npy"

    if not embedding_path.exists():
        print(f"  ⚠️  Embedding file not found: {embedding_path}")
        return None

    try:
        embeddings = np.load(embedding_path)
        return embeddings
    except Exception as e:
        print(f"  ❌ Error loading {embedding_path}: {e}")
        return None


def preprocess_embeddings(
    embeddings: np.ndarray,
    normalize_l2: bool = False,
    pca_dim: Optional[int] = None,
    random_state: int = 42
) -> np.ndarray:
    """
    Optionally preprocess embeddings before UMAP.

    Args:
        embeddings: Original embedding matrix (num_entities, embedding_dim)
        normalize_l2: If True, L2-normalize embeddings to unit length
        pca_dim: If specified, apply PCA to reduce to this dimensionality
        random_state: Random seed for PCA

    Returns:
        Preprocessed embeddings
    """
    processed = embeddings.copy()

    # L2 normalization
    if normalize_l2:
        processed = normalize(processed, norm='l2', axis=1)
        print(f"  Applied L2 normalization")

    # PCA dimensionality reduction
    if pca_dim is not None and pca_dim < processed.shape[1]:
        pca = PCA(n_components=pca_dim, random_state=random_state)
        processed = pca.fit_transform(processed)
        variance_explained = np.sum(pca.explained_variance_ratio_)
        print(f"  Applied PCA: {embeddings.shape[1]}D → {pca_dim}D (variance retained: {variance_explained:.2%})")
    elif pca_dim is not None:
        print(f"  ⚠️  Skipping PCA: requested dim ({pca_dim}) >= embedding dim ({processed.shape[1]})")

    return processed


def apply_umap(
    embeddings: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    spread: float = 1.0,
    repulsion_strength: float = 1.0,
    metric: str = 'cosine',
    random_state: int = 42
) -> np.ndarray:
    """
    Apply UMAP dimensionality reduction to 2D.

    Args:
        embeddings: User embedding matrix (num_users, embedding_dim)
        n_neighbors: UMAP n_neighbors parameter (default: 15)
        min_dist: UMAP min_dist parameter (default: 0.1)
        spread: UMAP spread parameter (default: 1.0)
        repulsion_strength: UMAP repulsion_strength parameter (default: 1.0)
        metric: Distance metric for UMAP (default: 'cosine')
        random_state: Random seed for reproducibility

    Returns:
        2D UMAP projection (num_users, 2)
    """
    try:
        import umap  # type: ignore
    except Exception as exc:  # ImportError + numba cache issues
        raise RuntimeError(
            "UMAP projection requested but `umap-learn` could not be imported. "
            "Install `umap-learn` and ensure numba has a writable cache directory."
        ) from exc

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        spread=spread,
        repulsion_strength=repulsion_strength,
        n_components=2,
        metric=metric,
        random_state=random_state,
        verbose=False
    )

    embedding_2d = reducer.fit_transform(embeddings)
    return embedding_2d


def apply_tsne(
    embeddings: np.ndarray,
    perplexity: float = 30.0,
    learning_rate: float = 200.0,
    max_iter: int = 1000,
    metric: str = 'cosine',
    random_state: int = 42
) -> np.ndarray:
    """
    Apply t-SNE dimensionality reduction to 2D.

    t-SNE often produces clearer visual cluster separation than UMAP,
    especially for data with local cluster structure.

    Args:
        embeddings: Embedding matrix (num_entities, embedding_dim)
        perplexity: t-SNE perplexity (default: 30.0)
        learning_rate: t-SNE learning rate (default: 200.0)
        max_iter: Number of iterations (default: 1000)
        metric: Distance metric (default: 'cosine')
        random_state: Random seed for reproducibility

    Returns:
        2D t-SNE projection (num_entities, 2)
    """
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=learning_rate,
        max_iter=max_iter,
        metric=metric,
        random_state=random_state,
        init='pca'
    )

    embedding_2d = tsne.fit_transform(embeddings)
    return embedding_2d


def cluster_embeddings_spherical(
    embeddings: np.ndarray,
    n_clusters: int = 5,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply spherical K-means clustering to embeddings.

    Spherical K-means normalizes embeddings to unit length before clustering,
    making it appropriate for cosine-based similarity measures. This is the
    recommended approach for PL models that use cosine/feature alignment.

    Args:
        embeddings: Embedding matrix (can be original high-dim or 2D UMAP)
        n_clusters: Number of clusters
        random_state: Random seed

    Returns:
        (cluster_labels, normalized_embeddings)
    """
    # L2-normalize embeddings to unit length (spherical k-means)
    embeddings_normalized = normalize(embeddings, norm='l2', axis=1)

    # Apply K-means on normalized embeddings
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(embeddings_normalized)

    return labels, embeddings_normalized


def compute_silhouette_scores(
    embeddings: np.ndarray,
    embeddings_normalized: np.ndarray,
    labels: np.ndarray
) -> Dict[str, float]:
    """
    Compute both cosine and euclidean silhouette scores.

    Args:
        embeddings: Original (non-normalized) embeddings
        embeddings_normalized: L2-normalized embeddings
        labels: Cluster labels

    Returns:
        Dictionary with 'cosine' and 'euclidean' silhouette scores
    """
    scores = {}

    # Cosine silhouette: computed on normalized embeddings with cosine metric
    # (Note: cosine distance on normalized vectors = euclidean distance / 2)
    try:
        scores['cosine'] = float(silhouette_score(embeddings_normalized, labels, metric='cosine'))
    except Exception as e:
        print(f"  Warning: Could not compute cosine silhouette: {e}")
        scores['cosine'] = None

    # Euclidean silhouette: computed on original embeddings
    try:
        scores['euclidean'] = float(silhouette_score(embeddings, labels, metric='euclidean'))
    except Exception as e:
        print(f"  Warning: Could not compute euclidean silhouette: {e}")
        scores['euclidean'] = None

    return scores


def compute_cluster_medoids(
    embeddings_hd: np.ndarray,
    labels: np.ndarray,
    metric: str = "cosine"
) -> Dict[int, int]:
    """
    Compute medoid index for each cluster in the original HD embedding space.

    The medoid is the point that minimizes the sum of distances to all other
    points in the same cluster. This is computed in the ORIGINAL high-dimensional
    embedding space (not the 2D UMAP coordinates) to preserve true cluster
    relationships.

    Medoid computation details:
    - For each cluster c, find the point p such that sum of distances from p
      to all other points in c is minimized.
    - Distance metric matches clustering: if embeddings are L2-normalized and
      clustering uses spherical k-means (cosine), use cosine distance.
      Otherwise, use Euclidean distance.

    Args:
        embeddings_hd: High-dimensional embeddings (num_entities, embedding_dim)
        labels: Cluster labels (num_entities,)
        metric: Distance metric - "cosine" (1 - cosine_similarity) or "euclidean"

    Returns:
        Dictionary mapping cluster label (int) to medoid index (int)
    """
    from scipy.spatial.distance import cdist

    medoids = {}
    unique_labels = np.unique(labels)

    for label in unique_labels:
        mask = labels == label
        cluster_indices = np.where(mask)[0]
        cluster_embeddings = embeddings_hd[mask]

        # Single-point cluster: that point is trivially the medoid
        if len(cluster_indices) == 1:
            medoids[int(label)] = int(cluster_indices[0])
            continue

        # Compute pairwise distances within the cluster
        if metric == "cosine":
            # cosine distance = 1 - cosine_similarity
            # Appropriate when using L2-normalized embeddings with spherical k-means
            distances = cdist(cluster_embeddings, cluster_embeddings, metric='cosine')
        else:
            distances = cdist(cluster_embeddings, cluster_embeddings, metric='euclidean')

        # Medoid: point with minimum sum of distances to all other points
        sum_distances = distances.sum(axis=1)
        medoid_local_idx = np.argmin(sum_distances)
        medoids[int(label)] = int(cluster_indices[medoid_local_idx])

    return medoids


def plot_umap(
    embedding_2d: np.ndarray,
    cluster_labels: np.ndarray,
    model_name: str,
    dataset: str,
    seed: int,
    entity: str,
    repr_type: str,
    output_path: Path,
    method: str = "umap",
    n_clusters: int = 5,
    dpi: int = 300,
    embeddings_hd: Optional[np.ndarray] = None,
    embeddings_hd_normalized: Optional[np.ndarray] = None
):
    """
    Create UMAP scatter plot with cluster coloring, marker shapes, and medoid overlays.

    Cluster labels are computed on the original high-dimensional embeddings
    and overlaid on the 2D UMAP projection for visualization.

    Visualization features:
    - Each cluster uses a unique (color, marker shape) combination
    - Marker assignment: unique labels are sorted, then mapped deterministically
      to CLUSTER_MARKERS by index (cycling if more clusters than shapes)
    - Medoid overlay: one representative point per cluster shown at larger size
      with black edge, computed in HD space (not UMAP 2D)
    - Plot titles do NOT include seed numbers (seed kept only in filename)

    Args:
        embedding_2d: 2D UMAP embeddings (num_entities, 2)
        cluster_labels: Cluster assignments (num_entities,) - computed on HD embeddings
        model_name: Model name for title
        dataset: Dataset identifier
        seed: Random seed (used for filename only, not shown in title)
        entity: 'user' or 'item'
        repr_type: 'main' or 'pl'
        output_path: Where to save the plot
        n_clusters: Number of clusters (K) used
        dpi: Resolution (default: 300)
        embeddings_hd: Original high-dimensional embeddings for medoid computation
        embeddings_hd_normalized: L2-normalized HD embeddings (for cosine medoid metric)
    """
    # Set clean, professional style
    sns.set_style("white")
    sns.set_context("paper", font_scale=1.1)

    # Create figure with white background
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
    ax.set_facecolor('white')

    # Get sorted unique cluster labels for deterministic marker assignment
    unique_labels = np.sort(np.unique(cluster_labels))
    n_unique = len(unique_labels)

    # Professional color palette - distinct but not garish
    # Use tab10 for professional, distinguishable colors
    palette = sns.color_palette("tab10", n_unique)

    # Create deterministic (color, marker) mapping for each cluster label
    # Marker assignment: sorted labels are mapped by index, cycling if necessary
    label_to_color = {label: palette[i] for i, label in enumerate(unique_labels)}
    label_to_marker = {label: CLUSTER_MARKERS[i % len(CLUSTER_MARKERS)]
                       for i, label in enumerate(unique_labels)}

    # Compute medoids in HD space if embeddings are provided
    # Medoid = point minimizing sum of distances to other cluster members
    # Computed in ORIGINAL HD space, plotted at its UMAP 2D location
    medoids = {}
    if embeddings_hd_normalized is not None:
        # Use normalized embeddings with cosine metric (matches spherical k-means)
        medoids = compute_cluster_medoids(embeddings_hd_normalized, cluster_labels, metric="cosine")
    elif embeddings_hd is not None:
        # Fall back to unnormalized embeddings with euclidean metric
        medoids = compute_cluster_medoids(embeddings_hd, cluster_labels, metric="euclidean")

    # Clean scatter plot with professional styling
    # Each cluster gets unique (color, marker) combination
    # HD cluster labels are overlaid on 2D projection
    for label in unique_labels:
        mask = cluster_labels == label
        color = label_to_color[label]
        marker = label_to_marker[label]

        ax.scatter(
            embedding_2d[mask, 0],
            embedding_2d[mask, 1],
            c=[color],
            marker=marker,
            label=f'Cluster {int(label) + 1}',
            alpha=0.75,
            s=60,
            edgecolors='white',
            linewidths=0.3
        )

    # Overlay medoids with distinct visual style:
    # - Larger marker size for visibility
    # - Black edgecolor with thick linewidth
    # - Small text annotation with cluster ID
    if medoids:
        for label, medoid_idx in medoids.items():
            color = label_to_color[label]
            marker = label_to_marker[label]
            x, y = embedding_2d[medoid_idx, 0], embedding_2d[medoid_idx, 1]

            # Draw medoid point (larger, with black edge)
            ax.scatter(
                x, y,
                c=[color],
                marker=marker,
                s=200,  # Larger size for medoid
                edgecolors='black',
                linewidths=2.0,
                zorder=10  # Draw on top
            )

            # Small, unobtrusive cluster ID annotation near medoid
            ax.annotate(
                f'{int(label) + 1}',
                (x, y),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8,
                fontweight='bold',
                color='black',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none'),
                zorder=11
            )

    # Get display names using utility functions
    dataset_display = get_dataset_display_name(dataset)
    model_arch, model_variant = get_model_display_info(model_name)

    # In healthcare context, "item" refers to support groups, so display as "Group"
    entity_display = "Group" if entity == 'item' else "User"
    repr_display = "PL-Branch" if repr_type == 'pl' else "Main"

    # Title formatting: seed is REMOVED from plot title for paper-facing figures
    # Seed value is retained only in filename for bookkeeping purposes
    # Format: Line 1 = Model info, Line 2 = Dataset + K (no seed)
    title_line1 = f"{model_arch} ({model_variant}) - {entity_display} Embeddings [{repr_display}]"
    title_line2 = f"{dataset_display} | K={n_clusters}"

    ax.set_title(
        f"{title_line1}\n{title_line2}",
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    axis_prefix = "t-SNE" if method == "tsne" else "UMAP"
    ax.set_xlabel(f"{axis_prefix} Dimension 1", fontsize=14)
    ax.set_ylabel(f"{axis_prefix} Dimension 2", fontsize=14)

    # Legend: reflects both color and marker for each cluster
    ax.legend(
        loc='best',
        frameon=True,
        framealpha=0.9,
        fontsize=10,
        markerscale=2
    )

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')

    # Tight layout
    plt.tight_layout()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"  ✅ Saved: {output_path.name}")


def generate_umap_single(
    model_name: str,
    dataset: str,
    seed: int,
    entity: str = 'user',
    repr_type: str = 'main',
    embeddings_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    n_clusters: int = 5,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    spread: float = 1.0,
    repulsion_strength: float = 1.0,
    normalize_l2: bool = False,
    pca_dim: Optional[int] = None,
    cluster_space: str = 'hd',
    umap_random_state: Optional[int] = None,
    include_k_in_filename: bool = False,
    method: str = 'umap',
    perplexity: float = 30.0
) -> bool:
    """
    Generate 2D visualization plot for a single model/dataset/seed/entity/repr combination.

    Args:
        model_name: Model name (e.g., 'mf_baseline', 'neumf_pl')
        dataset: Dataset name
        seed: Random seed
        entity: 'user' or 'item' (default: 'user')
        repr_type: 'main' or 'pl' (default: 'main')
        embeddings_dir: Custom embeddings directory
        output_dir: Custom output directory
        n_clusters: Number of clusters for K-means
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        spread: UMAP spread parameter
        repulsion_strength: UMAP repulsion_strength parameter
        normalize_l2: If True, L2-normalize before dimensionality reduction
        pca_dim: If specified, apply PCA to this dimensionality first
        cluster_space: 'hd' (cluster on original embeddings) or 'umap' (cluster on 2D)
        umap_random_state: Random state for UMAP/t-SNE (defaults to seed if None)
        include_k_in_filename: If True, include k value in output filename
        method: 'umap' or 'tsne' (default: 'umap')
        perplexity: t-SNE perplexity parameter (default: 30.0)

    Returns:
        True if successful, False otherwise
    """
    # GUARDRAIL: Coerce deprecated 'umap' cluster_space to 'hd'
    # UMAP is visualization only - cluster labels must be computed on HD embeddings
    if cluster_space == 'umap':
        warnings.warn(
            "DEPRECATED: cluster_space='umap' is deprecated and will be removed in a future version. "
            "Clustering in 2D UMAP space is methodologically unsound because UMAP distorts distances "
            "and densities. Coercing to cluster_space='hd' (high-dimensional). "
            "Cluster labels are now computed on the original HD embeddings and overlaid on UMAP.",
            DeprecationWarning,
            stacklevel=2
        )
        cluster_space = 'hd'

    method_display = method.upper()
    print(f"Generating {method_display}: {model_name} on {dataset} (seed={seed}, entity={entity}, repr={repr_type})")

    # Set paths
    if embeddings_dir is None:
        embeddings_dir = Path(f"results/{dataset}/embeddings")
    if output_dir is None:
        output_dir = Path(f"results/{dataset}/umap_plots")

    # Load embeddings
    embeddings = load_embeddings(embeddings_dir, model_name, seed, entity, repr_type)

    if embeddings is None:
        return False

    print(f"  Loaded embeddings: {embeddings.shape}")

    # Store original embeddings shape for metadata
    original_shape = embeddings.shape

    # Preprocessing (optional: L2 normalization and/or PCA)
    embeddings_processed = preprocess_embeddings(
        embeddings,
        normalize_l2=normalize_l2,
        pca_dim=pca_dim,
        random_state=seed
    )

    # Apply dimensionality reduction (UMAP or t-SNE)
    try:
        dr_seed = umap_random_state if umap_random_state is not None else seed
        if method == 'tsne':
            embedding_2d = apply_tsne(
                embeddings_processed,
                perplexity=perplexity,
                metric='cosine',
                random_state=dr_seed
            )
            print(f"  Applied t-SNE: {embedding_2d.shape}")
        else:
            embedding_2d = apply_umap(
                embeddings_processed,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                spread=spread,
                repulsion_strength=repulsion_strength,
                metric='cosine',
                random_state=dr_seed
            )
            print(f"  Applied UMAP: {embedding_2d.shape}")
    except Exception as e:
        print(f"  ❌ {method.upper()} failed: {e}")
        return False

    # Cluster: choose between high-dimensional or UMAP space
    try:
        if cluster_space == 'umap':
            # Cluster on 2D UMAP coordinates
            cluster_labels, embeddings_normalized = cluster_embeddings_spherical(
                embedding_2d, n_clusters=n_clusters, random_state=seed
            )
            print(f"  Clustered into {n_clusters} groups (spherical k-means on 2D UMAP)")
            # For silhouette: use the 2D UMAP coordinates
            silhouette_embeddings = embedding_2d
            silhouette_embeddings_norm = embeddings_normalized
        else:
            # Cluster on original high-dimensional embeddings (default behavior)
            cluster_labels, embeddings_normalized = cluster_embeddings_spherical(
                embeddings, n_clusters=n_clusters, random_state=seed
            )
            print(f"  Clustered into {n_clusters} groups (spherical k-means on high-D)")
            # For silhouette: use the original embeddings
            silhouette_embeddings = embeddings
            silhouette_embeddings_norm = embeddings_normalized
    except Exception as e:
        print(f"  ❌ Clustering failed: {e}")
        return False

    # Compute silhouette scores
    try:
        silhouette_scores = compute_silhouette_scores(
            silhouette_embeddings, silhouette_embeddings_norm, cluster_labels
        )
        print(f"  Silhouette scores: cosine={silhouette_scores['cosine']:.4f}, euclidean={silhouette_scores['euclidean']:.4f}")
    except Exception as e:
        print(f"  ⚠️  Silhouette computation failed: {e}")
        silhouette_scores = {'cosine': None, 'euclidean': None}

    # Build output filenames
    if include_k_in_filename:
        base_name = f"{model_name}_seed{seed}_{entity}_{repr_type}_k{n_clusters}"
    else:
        base_name = f"{model_name}_seed{seed}_{entity}_{repr_type}"

    output_path = output_dir / f"{base_name}_umap.png"
    metrics_path = output_dir / f"{base_name}_metrics.json"

    # Plot with HD embeddings for medoid computation
    # Pass both original and normalized embeddings so medoids can be computed
    # in HD space using the same metric as clustering (cosine for spherical k-means)
    try:
        plot_umap(
            embedding_2d, cluster_labels, model_name, dataset, seed,
            entity, repr_type, output_path, n_clusters=n_clusters,
            method=method,
            embeddings_hd=silhouette_embeddings,
            embeddings_hd_normalized=silhouette_embeddings_norm
        )
    except Exception as e:
        print(f"  ❌ Plotting failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Save metrics to JSON sidecar
    try:
        metrics = {
            'dataset': dataset,
            'model': model_name,
            'seed': seed,
            'entity': entity,
            'repr': repr_type,
            'projection_method': method,
            'preprocessing': {
                'normalize_l2': normalize_l2,
                'pca_dim': pca_dim
            },
            'umap_params': {
                'n_neighbors': n_neighbors,
                'min_dist': min_dist,
                'spread': spread,
                'repulsion_strength': repulsion_strength,
                'metric': 'cosine',
                'random_state': dr_seed
            },
            'tsne_params': {
                'perplexity': perplexity,
                'learning_rate': 200.0,
                'max_iter': 1000,
                'metric': 'cosine',
                'random_state': dr_seed,
                'init': 'pca'
            },
            'clustering': {
                'n_clusters': n_clusters,
                'method': 'spherical_kmeans',
                'cluster_space': cluster_space
            },
            'silhouette_cosine': silhouette_scores['cosine'],
            'silhouette_euclidean': silhouette_scores['euclidean'],
            'embedding_shape': list(original_shape)
        }

        output_dir.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"  ✅ Saved: {metrics_path.name}")

    except Exception as e:
        print(f"  ⚠️  Could not save metrics: {e}")

    return True


def generate_all_umaps(
    datasets=None,
    models=None,
    seeds=None,
    n_clusters: int = 5,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    spread: float = 1.0,
    repulsion_strength: float = 1.0,
    normalize_l2: bool = False,
    pca_dim: Optional[int] = None,
    cluster_space: str = 'hd',
    umap_random_state: Optional[int] = None
):
    """Generate UMAP plots for all models."""
    if datasets is None:
        datasets = ['support_groups_full_164', 'support_groups_full_164_loo']
    if models is None:
        models = ['mf_baseline', 'mf_pl', 'mlp_baseline', 'mlp_pl', 'neumf_baseline', 'neumf_pl']
    if seeds is None:
        seeds = [42, 52, 62, 122, 232]

    total = len(datasets) * len(models) * len(seeds)
    success_count = 0

    print(f"Generating UMAP plots for {total} models...")
    print("=" * 80)

    for dataset in datasets:
        for model_name in models:
            for seed in seeds:
                if generate_umap_single(
                    model_name, dataset, seed,
                    n_clusters=n_clusters,
                    n_neighbors=n_neighbors,
                    min_dist=min_dist,
                    spread=spread,
                    repulsion_strength=repulsion_strength,
                    normalize_l2=normalize_l2,
                    pca_dim=pca_dim,
                    cluster_space=cluster_space,
                    umap_random_state=umap_random_state
                ):
                    success_count += 1

    print("=" * 80)
    print(f"UMAP generation complete: {success_count}/{total} successful")

    if success_count < total:
        print(f"⚠️  {total - success_count} plots failed or embeddings were not found")

    return success_count == total


def main():
    parser = argparse.ArgumentParser(
        description="Generate UMAP visualizations of user/item embeddings",
        epilog="Use --presentation for recommended settings for publication-quality plots with clearer separation."
    )
    parser.add_argument('--model', type=str, help='Model name (e.g., mf_baseline, neumf_pl)')
    parser.add_argument('--dataset', type=str, help='Dataset name (e.g., support_groups_full_164)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--entity', type=str, default='user', choices=['user', 'item'],
                        help='Entity type: user or item (default: user)')
    parser.add_argument('--repr', type=str, default='main', choices=['main', 'pl'], dest='repr_type',
                        help='Representation type: main or pl (default: main). Use "pl" for PL-branch embeddings.')
    parser.add_argument('--all', action='store_true', help='Generate for all models')
    parser.add_argument('--all-models', action='store_true', help='Generate for all models in dataset')

    # Clustering parameters
    parser.add_argument('--n-clusters', type=int, default=5, help='Number of K-means clusters (default: 5)')
    parser.add_argument('--cluster-space', type=str, default='hd', choices=['hd', 'umap'],
                        help='Cluster on high-dimensional embeddings. Default: hd. '
                             'NOTE: "umap" is DEPRECATED and will be coerced to "hd". '
                             'UMAP is visualization only; cluster labels are computed on HD embeddings.')

    # Preprocessing options
    parser.add_argument('--normalize', action='store_true',
                        help='L2-normalize embeddings before UMAP (presentation mode)')
    parser.add_argument('--pca-dim', type=int, default=None,
                        help='Apply PCA to reduce to N dimensions before UMAP (e.g., 50)')

    # Dimensionality reduction method
    parser.add_argument('--method', type=str, default='umap', choices=['umap', 'tsne'],
                        help='Dimensionality reduction method: umap or tsne (default: umap). '
                             't-SNE often produces clearer visual clusters.')

    # UMAP hyperparameters
    parser.add_argument('--n-neighbors', type=int, default=15, help='UMAP n_neighbors (default: 15)')
    parser.add_argument('--min-dist', type=float, default=0.1, help='UMAP min_dist (default: 0.1)')
    parser.add_argument('--spread', type=float, default=1.0, help='UMAP spread (default: 1.0)')
    parser.add_argument('--repulsion-strength', type=float, default=1.0,
                        help='UMAP repulsion_strength (default: 1.0)')

    # t-SNE hyperparameters
    parser.add_argument('--perplexity', type=float, default=30.0, help='t-SNE perplexity (default: 30.0)')

    parser.add_argument('--random-state', type=int, default=None,
                        help='Random state (defaults to --seed if not specified)')

    # Presentation mode
    parser.add_argument('--presentation', action='store_true',
                        help='Presentation mode: Use recommended settings for clearer visual separation '
                             '(normalize=True, pca-dim=50, n-neighbors=50, min-dist=0.0, repulsion-strength=2.0). '
                             'NOTE: Cluster labels are always computed on HD embeddings and overlaid on UMAP. '
                             'UMAP is visualization only and does not affect model training or evaluation.')

    # I/O options
    parser.add_argument('--embeddings-dir', type=str, help='Custom embeddings directory')
    parser.add_argument('--output-dir', type=str, help='Custom output directory')
    parser.add_argument('--include-k', action='store_true', help='Include k value in filename (avoids race conditions)')

    args = parser.parse_args()

    # Apply presentation mode defaults if requested
    if args.presentation:
        print("=" * 80)
        print("PRESENTATION MODE ENABLED")
        print("Using recommended settings for clearer visual separation:")
        print("  --normalize (L2 normalization)")
        print("  --pca-dim 50 (PCA to 50 dimensions)")
        print("  --n-neighbors 50 (larger neighborhood)")
        print("  --min-dist 0.0 (tighter clusters)")
        print("  --repulsion-strength 2.0 (stronger repulsion)")
        print("  --cluster-space hd (cluster labels from HD embeddings, overlaid on UMAP)")
        print()
        print("NOTE: UMAP is visualization only. Cluster labels are computed on")
        print("      the original high-dimensional embeddings (HD) and overlaid on UMAP.")
        print("      Model training and evaluation metrics (HR/NDCG/AUC) are unchanged.")
        print("=" * 80)
        print()

        # Override parameters with presentation defaults (only if not explicitly set)
        if not hasattr(args, 'normalize') or not args.normalize:
            args.normalize = True
        if args.pca_dim is None:
            args.pca_dim = 50
        if args.n_neighbors == 15:  # default value
            args.n_neighbors = 50
        if args.min_dist == 0.1:  # default value
            args.min_dist = 0.0
        if args.repulsion_strength == 1.0:  # default value
            args.repulsion_strength = 2.0
        # NOTE: cluster_space always stays 'hd' - UMAP is visualization only

    if args.all:
        # Generate all UMAP plots
        success = generate_all_umaps(
            n_clusters=args.n_clusters,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            spread=args.spread,
            repulsion_strength=args.repulsion_strength,
            normalize_l2=args.normalize,
            pca_dim=args.pca_dim,
            cluster_space=args.cluster_space,
            umap_random_state=args.random_state
        )
        sys.exit(0 if success else 1)

    elif args.all_models and args.dataset:
        # Generate for all models in one dataset
        models = ['mf_baseline', 'mf_pl', 'mlp_baseline', 'mlp_pl', 'neumf_baseline', 'neumf_pl']
        seeds = [42, 52, 62, 122, 232]

        total = len(models) * len(seeds)
        success_count = 0

        for model_name in models:
            for seed in seeds:
                if generate_umap_single(
                    model_name, args.dataset, seed,
                    embeddings_dir=Path(args.embeddings_dir) if args.embeddings_dir else None,
                    output_dir=Path(args.output_dir) if args.output_dir else None,
                    n_clusters=args.n_clusters,
                    n_neighbors=args.n_neighbors,
                    min_dist=args.min_dist,
                    spread=args.spread,
                    repulsion_strength=args.repulsion_strength,
                    normalize_l2=args.normalize,
                    pca_dim=args.pca_dim,
                    cluster_space=args.cluster_space,
                    umap_random_state=args.random_state
                ):
                    success_count += 1

        print(f"Generated {success_count}/{total} UMAP plots for {args.dataset}")
        sys.exit(0 if success_count == total else 1)

    elif args.model and args.dataset:
        # Generate single visualization
        success = generate_umap_single(
            args.model, args.dataset, args.seed,
            entity=args.entity,
            repr_type=args.repr_type,
            embeddings_dir=Path(args.embeddings_dir) if args.embeddings_dir else None,
            output_dir=Path(args.output_dir) if args.output_dir else None,
            n_clusters=args.n_clusters,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            spread=args.spread,
            repulsion_strength=args.repulsion_strength,
            normalize_l2=args.normalize,
            pca_dim=args.pca_dim,
            cluster_space=args.cluster_space,
            umap_random_state=args.random_state,
            include_k_in_filename=args.include_k,
            method=args.method,
            perplexity=args.perplexity
        )
        sys.exit(0 if success else 1)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
