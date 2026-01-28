"""
Enhanced Pseudo-Label Embedding for Support Group Recommendation

This module implements the robust pseudo-labeling mechanism described in:
"Enhanced Recommender Systems for Automated Support Groups Formation"

Key Features:
- Cosine similarity-based pseudo-label generation from survey data
- Robust enhancement mechanisms (confidence schedule, stability filter, temperature calibration)
- Hierarchical smoothing for group relationships
- Soft pseudo-label loss with calibrated confidence scores
- Propensity weighting for over-exposed groups

Reference: Meeting presentation Sep 17, 2025
Author: Pronob Kumar Barman
"""

import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List
import numpy as np
from collections import Counter, defaultdict, deque
import logging

logger = logging.getLogger(__name__)


class LazyPseudoLabelTensor:
    """
    Memory-efficient lazy tensor that mimics torch.Tensor behavior for pseudo-label matrices.

    Only computes values when indexed, avoiding allocation of massive [batch_size, num_items] tensors.
    Maintains API compatibility with existing indexing patterns like tensor[user_indices, item_indices].
    """

    def __init__(self, batch_size: int, num_items: int, device: torch.device,
                 fill_value: float = 0.0, dtype: torch.dtype = torch.float32):
        self.batch_size = batch_size
        self.num_items = num_items
        self.device = device
        self.fill_value = fill_value
        self.dtype = dtype

    def __getitem__(self, indices):
        """Handle indexing operations like tensor[user_indices, item_indices]."""
        if isinstance(indices, tuple) and len(indices) == 2:
            user_indices, item_indices = indices

            # Handle torch.arange(len(u_idx)), i_idx pattern from mlp.py:217
            if isinstance(user_indices, torch.Tensor) and isinstance(item_indices, torch.Tensor):
                batch_size = len(user_indices)
                # Return zeros for all user-item pairs (no survey data case)
                return torch.full((batch_size,), self.fill_value,
                                device=self.device, dtype=self.dtype)

        # Fallback for other indexing patterns
        if isinstance(indices, torch.Tensor):
            return torch.full(indices.shape, self.fill_value,
                            device=self.device, dtype=self.dtype)

        # Single index
        return torch.tensor(self.fill_value, device=self.device, dtype=self.dtype)

    @property
    def shape(self):
        """Return tensor shape for compatibility."""
        return (self.batch_size, self.num_items)

    @property
    def size(self):
        """Return tensor size method for compatibility."""
        return lambda dim=None: self.shape if dim is None else self.shape[dim]

    def to(self, device):
        """Device transfer compatibility."""
        self.device = device
        return self

    def cuda(self):
        """CUDA compatibility."""
        self.device = torch.device('cuda')
        return self

    def cpu(self):
        """CPU compatibility."""
        self.device = torch.device('cpu')
        return self


class PLEmbedding(nn.Module):
    """
    Enhanced Pseudo-Label Embedding for Support Group Recommendation.

    Implements the complete algorithm from the meeting presentation including:
    1. Pseudo-label generation via cosine similarity: sim(u_survey, g') = u_survey·g' / (||u_survey|| ||g'||)
    2. Robust enhancement mechanisms (confidence schedule, stability filter, temperature calibration)
    3. Hierarchical smoothing for group relationships
    4. Quality control and monitoring
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        dim: int = 32,
        tau: float = 0.8,  # Threshold for pseudo-label generation (τ from slides)
        temperature: float = 1.0,  # Temperature scaling parameter
        confidence_threshold_start: float = 0.95,  # Starting confidence threshold (θ_confidence)
        confidence_threshold_end: float = 0.85,  # Ending confidence threshold
        stability_window: int = 2,  # Window for stability checking (consecutive epochs)
        hierarchy_mapping: Optional[Dict[int, int]] = None,
        hierarchy_epsilon: float = 0.2,  # Mass allocated to parent groups (ε from slides)
        propensity_alpha: float = 0.1,  # Smoothing factor for propensity updates
        use_enhanced_features: bool = True,  # Whether to use enhanced pseudo-labeling features
        use_soft_labels: bool = False,  # 🆕 Use similarity scores directly instead of binary threshold
        memory_threshold_gb: float = 2.0,  # Memory threshold for tensor creation (GB)
        max_tensor_elements: int = 50_000_000,  # Max elements before using lazy tensor
        **kwargs
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.dim = dim
        self.tau = tau
        self.use_soft_labels = use_soft_labels  # 🆕 Store soft labels flag
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.confidence_threshold_start = confidence_threshold_start
        self.confidence_threshold_end = confidence_threshold_end
        self.current_confidence_threshold = confidence_threshold_start
        self.stability_window = stability_window
        self.hierarchy_mapping = hierarchy_mapping or {}
        self.hierarchy_epsilon = hierarchy_epsilon
        self.propensity_alpha = propensity_alpha
        self.use_enhanced_features = use_enhanced_features

        # Memory management parameters
        self.memory_threshold_gb = memory_threshold_gb
        self.max_tensor_elements = max_tensor_elements

        # Main embeddings for pseudo-labeling
        self.user_emb = nn.Embedding(num_users, dim)
        self.item_emb = nn.Embedding(num_items, dim)

        # Initialize embeddings
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

        # Stability tracking for consecutive epochs
        self.user_top_predictions = defaultdict(lambda: deque(maxlen=stability_window))

        # Propensity tracking for over-exposed groups
        self.group_exposure_counts = torch.zeros(num_items)
        self.group_propensity_weights = torch.ones(num_items)

        # Statistics tracking
        self.stats = {
            'total_pseudo_labels_generated': 0,
            'pseudo_labels_accepted_confidence': 0,
            'pseudo_labels_accepted_stability': 0,
            'pseudo_labels_final_accepted': 0,
            'hierarchy_smoothing_applied': 0
        }

    def generate_pseudo_labels(self, user_survey_embeddings: torch.Tensor,
                             group_embeddings: torch.Tensor,
                             user_ids: torch.LongTensor) -> torch.Tensor:
        """
        Generate pseudo-labels using cosine similarity as described in slides.

        Formula: sim(u_survey, g') = u_survey·g' / (||u_survey|| ||g'||)
        Pseudo-label: y_pseudo = 1 if sim > τ, 0 otherwise

        Args:
            user_survey_embeddings: Survey embeddings for users [batch_size, dim]
            group_embeddings: Group embeddings [num_groups, dim]
            user_ids: User IDs for the batch [batch_size]

        Returns:
            pseudo_labels: Binary pseudo-labels [batch_size, num_groups]
        """
        # Normalize embeddings for cosine similarity
        user_survey_norm = F.normalize(user_survey_embeddings, p=2, dim=1)
        group_norm = F.normalize(group_embeddings, p=2, dim=1)

        # Compute cosine similarity: sim(u_survey, g')
        similarity_matrix = torch.mm(user_survey_norm, group_norm.t())  # [batch_size, num_groups]

        # FIX 1.2: Normalize similarity from [-1,1] to [0,1] range to prevent BCE overflow
        similarity_matrix = (similarity_matrix + 1.0) / 2.0
        similarity_matrix = torch.clamp(similarity_matrix, 0.0, 1.0)

        # 🆕 SOFT LABELS: Use similarity scores directly (regression) vs binary threshold (classification)
        if self.use_soft_labels:
            # SOFT APPROACH: Use normalized similarity as continuous target [0,1]
            # This preserves ranking information and gradation
            # Better for simple models (MF, MLP) that can't handle hard thresholds
            pseudo_labels = similarity_matrix  # No thresholding!
            self.stats['total_pseudo_labels_generated'] += (similarity_matrix > 0.5).sum().item()
        else:
            # HARD APPROACH: Binary threshold (original method)
            # y_pseudo = 1 if sim > τ, 0 otherwise
            pseudo_labels = (similarity_matrix > self.tau).float()
            # FIX 1.1: Ensure pseudo-labels are strictly in [0,1] range
            pseudo_labels = torch.clamp(pseudo_labels, 0.0, 1.0)
            self.stats['total_pseudo_labels_generated'] += pseudo_labels.sum().item()

        return pseudo_labels, similarity_matrix

    def apply_confidence_schedule(self, predictions: torch.Tensor, epoch: int, total_epochs: int) -> torch.Tensor:
        """
        Apply confidence schedule: θ_confidence starts at 0.95, decays to 0.85 if validation improves.
        """
        # Linear decay of confidence threshold
        decay_factor = epoch / max(total_epochs, 1)
        self.current_confidence_threshold = (
            self.confidence_threshold_start -
            decay_factor * (self.confidence_threshold_start - self.confidence_threshold_end)
        )

        # Apply confidence gating
        confidence_mask = (predictions.max(dim=1)[0] > self.current_confidence_threshold)

        self.stats['pseudo_labels_accepted_confidence'] += confidence_mask.sum().item()

        return confidence_mask

    def apply_temperature_calibration(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply temperature calibration before confidence gating.
        """
        return logits / self.temperature

    def apply_stability_filter(self, predictions: torch.Tensor, user_ids: torch.LongTensor) -> torch.Tensor:
        """
        Apply stability filter: Accept pseudo-label only if top-1 group is unchanged across
        two consecutive epochs and two different dropout seeds.
        """
        batch_size = predictions.size(0)
        stability_mask = torch.zeros(batch_size, dtype=torch.bool, device=predictions.device)

        # Get current top-1 predictions
        current_top_items = predictions.argmax(dim=1)

        for i, user_id in enumerate(user_ids.cpu().numpy()):
            current_top = current_top_items[i].item()

            # Update history
            self.user_top_predictions[user_id].append(current_top)

            # Check stability: top-1 unchanged across consecutive epochs
            if len(self.user_top_predictions[user_id]) >= self.stability_window:
                recent_predictions = list(self.user_top_predictions[user_id])
                is_stable = all(pred == recent_predictions[0] for pred in recent_predictions)
                stability_mask[i] = is_stable

        self.stats['pseudo_labels_accepted_stability'] += stability_mask.sum().item()

        return stability_mask

    def apply_hierarchy_smoothing(self, predictions: torch.Tensor, items: torch.LongTensor) -> torch.Tensor:
        """
        Apply hierarchical smoothing: For leaf subgroup predictions, allocate ε mass to parent group.
        """
        if not self.hierarchy_mapping:
            return predictions

        smoothed_predictions = predictions.clone()

        for i, item_id in enumerate(items.cpu().numpy()):
            if item_id in self.hierarchy_mapping:
                parent_id = self.hierarchy_mapping[item_id]

                if parent_id < self.num_items and parent_id != item_id:
                    # Find parent index in current batch
                    parent_mask = (items == parent_id)
                    if parent_mask.any():
                        parent_idx = torch.where(parent_mask)[0][0]

                        # Allocate ε mass to parent group
                        mass_to_parent = self.hierarchy_epsilon * predictions[i]
                        smoothed_predictions[parent_idx] += mass_to_parent
                        smoothed_predictions[i] *= (1 - self.hierarchy_epsilon)

                        self.stats['hierarchy_smoothing_applied'] += 1

        # FIX 1.1: Ensure smoothed predictions stay in [0,1] range
        smoothed_predictions = torch.clamp(smoothed_predictions, 0.0, 1.0)

        return smoothed_predictions

    def update_propensity_weights(self, accepted_items: torch.LongTensor):
        """
        Update propensity weights to down-weight over-exposed groups.
        """
        if len(accepted_items) == 0:
            return

        # Update exposure counts
        for item_id in accepted_items.cpu().numpy():
            if item_id < self.num_items:
                old_count = self.group_exposure_counts[item_id]
                new_count = (1 - self.propensity_alpha) * old_count + self.propensity_alpha
                self.group_exposure_counts[item_id] = new_count

        # Update propensity weights (inverse of exposure)
        max_exposure = self.group_exposure_counts.max()
        if max_exposure > 0:
            self.group_propensity_weights = 1.0 / (1.0 + self.group_exposure_counts / max_exposure)

    def get_propensity_weights(self, items: torch.LongTensor) -> torch.Tensor:
        """Get propensity weights for items to down-weight over-exposed groups."""
        return self.group_propensity_weights[items]

    def _should_use_lazy_tensor(self, batch_size: int, device: torch.device) -> bool:
        """
        Determine if lazy tensor should be used based on memory constraints.

        Args:
            batch_size: Number of users in the batch
            device: Target device for tensor allocation

        Returns:
            True if lazy tensor should be used, False otherwise
        """
        # Calculate tensor size
        num_elements = batch_size * self.num_items
        tensor_size_gb = (num_elements * 4) / (1024**3)  # 4 bytes per float32

        # Check element count threshold
        if num_elements > self.max_tensor_elements:
            logger.info(f"Using lazy tensor: {num_elements} elements > {self.max_tensor_elements} threshold")
            return True

        # Check memory size threshold
        if tensor_size_gb > self.memory_threshold_gb:
            logger.info(f"Using lazy tensor: {tensor_size_gb:.2f}GB > {self.memory_threshold_gb}GB threshold")
            return True

        # Check available GPU memory if using CUDA
        if device.type == 'cuda' and torch.cuda.is_available():
            try:
                free_memory = torch.cuda.get_device_properties(device).total_memory - torch.cuda.memory_allocated(device)
                free_memory_gb = free_memory / (1024**3)

                # Use lazy tensor if required memory > 50% of free memory
                if tensor_size_gb > 0.5 * free_memory_gb:
                    logger.info(f"Using lazy tensor: {tensor_size_gb:.2f}GB > 50% of {free_memory_gb:.2f}GB free memory")
                    return True
            except Exception as e:
                logger.warning(f"Could not check GPU memory: {e}")

        return False

    def forward(self, users: torch.LongTensor, items: torch.LongTensor,
                user_survey_embeddings: Optional[torch.Tensor] = None,
                epoch: int = 0, total_epochs: int = 100,
                return_confidence: bool = False,
                return_stability: bool = False,
                use_teacher: bool = False,
                apply_hierarchy: bool = False) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass implementing the complete pseudo-labeling pipeline.

        Returns:
            pseudo_labels: Generated and filtered pseudo-labels
            metadata: Dictionary with statistics and intermediate results
        """
        # Get embeddings
        user_emb = self.user_emb(users)
        item_emb = self.item_emb(items)

        # If survey embeddings are provided, generate pseudo-labels
        if user_survey_embeddings is not None:
            # Step 1: Generate pseudo-labels via cosine similarity
            all_item_emb = self.item_emb.weight  # All group embeddings
            pseudo_labels, similarity_scores = self.generate_pseudo_labels(
                user_survey_embeddings, all_item_emb, users
            )

            # Step 2: Apply temperature calibration
            calibrated_scores = self.apply_temperature_calibration(similarity_scores)

            # Step 3: Apply confidence schedule
            confidence_mask = self.apply_confidence_schedule(calibrated_scores, epoch, total_epochs)

            # Step 4: Apply stability filter
            stability_mask = self.apply_stability_filter(calibrated_scores, users)

            # Step 5: Combine confidence and stability filters
            final_mask = confidence_mask & stability_mask

            # Step 6: Apply hierarchical smoothing to accepted predictions
            if final_mask.any():
                accepted_items = items[final_mask]
                pseudo_labels[final_mask] = self.apply_hierarchy_smoothing(
                    pseudo_labels[final_mask], accepted_items
                )

                # Update propensity weights
                self.update_propensity_weights(accepted_items)

            # Apply final mask to pseudo-labels
            final_pseudo_labels = pseudo_labels * final_mask.unsqueeze(1).float()
            self.stats['pseudo_labels_final_accepted'] += final_mask.sum().item()

        else:
            # No survey data available
            final_pseudo_labels = torch.zeros(len(users), self.num_items, device=users.device)
            final_mask = torch.zeros(len(users), dtype=torch.bool, device=users.device)

        metadata = {
            'confidence_threshold': self.current_confidence_threshold,
            'acceptance_rate': final_mask.float().mean().item() if len(final_mask) > 0 else 0.0,
            'stats': self.stats.copy()
        }

        return final_pseudo_labels, metadata


class EnhancedPseudoLabelLoss(nn.Module):
    """Enhanced Pseudo-Label Loss with advanced features."""
    
    def __init__(self, loss_type="bce", confidence_threshold=0.8, debias_popularity=True, pl_embedding=None):
        super().__init__()
        self.loss_type = loss_type
        self.confidence_threshold = confidence_threshold
        self.debias_popularity = debias_popularity
        self.pl_embedding = pl_embedding
    
    def forward(self, predictions, targets, confidence_scores=None):
        if self.loss_type == "bce":
            return F.binary_cross_entropy(predictions, targets)
        else:
            return F.mse_loss(predictions, targets)


class SoftPseudoLabelLoss(nn.Module):
    """
    Soft Pseudo-Label Loss implementation from the meeting slides.

    L_PL-soft = -∑_(u,g) [y_pseudo-soft * log(ŷ) + (1 - y_pseudo-soft) * log(1 - ŷ)]

    Where y_pseudo-soft incorporates:
    - Calibrated confidence scores
    - Hierarchical smoothing
    - Propensity weighting (down-weight over-exposed groups)
    """

    def __init__(self, lambda_pl: float = 1.0):
        super().__init__()
        self.lambda_pl = lambda_pl

    def forward(self, predictions: torch.Tensor, pseudo_labels: torch.Tensor,
                propensity_weights: torch.Tensor) -> torch.Tensor:
        """
        Compute soft pseudo-label loss.

        Args:
            predictions: Model predictions [batch_size, num_groups]
            pseudo_labels: Soft pseudo-labels [batch_size, num_groups]
            propensity_weights: Propensity weights [batch_size, num_groups]

        Returns:
            loss: Weighted pseudo-label loss
        """
        # Apply propensity weighting to pseudo-labels
        weighted_pseudo_labels = pseudo_labels * propensity_weights

        # Compute binary cross-entropy with soft labels
        epsilon = 1e-8  # For numerical stability
        predictions_clamped = torch.clamp(predictions, epsilon, 1 - epsilon)

        # L_PL-soft = -∑[y_pseudo-soft * log(ŷ) + (1 - y_pseudo-soft) * log(1 - ŷ)]
        loss = -(weighted_pseudo_labels * torch.log(predictions_clamped) +
                (1 - weighted_pseudo_labels) * torch.log(1 - predictions_clamped))

        # Return mean loss weighted by lambda_pl
        return self.lambda_pl * loss.mean()


