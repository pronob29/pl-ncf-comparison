"""
Simplified engine for testing - compatible with new PLEmbedding implementation
"""

from pathlib import Path
from typing import Dict, Tuple, Union, Optional
import numpy as np
import logging

import torch
from torch import nn
from tqdm import tqdm
from tensorboardX import SummaryWriter

from utils import save_checkpoint, use_optimizer
from metrics import MetronAtK
from pl import PLEmbedding, SoftPseudoLabelLoss

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.

    FL(p_t) = -α * (1 - p_t)^γ * log(p_t)

    Reduces relative loss for well-classified examples (p_t > 0.5),
    focusing training on hard negatives. Prevents BCE overfitting.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 1.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Model predictions [batch_size]
            target: Ground truth labels [batch_size]

        Returns:
            Focal loss value
        """
        # Binary cross-entropy
        bce = nn.functional.binary_cross_entropy(pred, target, reduction='none')

        # Probability of correct class
        p_t = torch.where(target == 1, pred, 1 - pred)

        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Focal loss: alpha * focal_weight * bce
        focal_loss = self.alpha * focal_weight * bce

        return focal_loss.mean()


class Engine(object):
    """
    Simplified trainer/evaluator for testing with new PL implementation.
    """

    def __init__(self, config: dict):
        self.config = config
        self.lambda_pl = config.get("lambda_pl", 0.0)
        self.tau = config.get("tau", 0.8)
        self.temperature = config.get("temperature", 1.0)
        self.current_epoch = 0
        self.total_epochs = config.get("num_epoch", 100)

        # Memory management configuration
        self.max_eval_batch_size = config.get("max_eval_batch_size", 1000)
        self.enable_chunking = config.get("enable_chunking", True)

        # 🚀 v11: Curriculum learning parameters
        self.use_curriculum = config.get("use_curriculum_learning", False)
        self.curriculum_warmup_epochs = config.get("curriculum_warmup_epochs", 5)
        self.curriculum_pl_warmup_weight = config.get("curriculum_pl_warmup_weight", 2.0)

        # 🚀 v11: Focal loss option
        self.use_focal_loss = config.get("use_focal_loss", False)
        self.focal_gamma = config.get("focal_loss_gamma", 2.0)

        # 🚀 v11: Gradient clipping
        self.gradient_clip_norm = config.get("gradient_clip_norm", 1.0)

        # Initialize optimizer and losses
        self.opt = use_optimizer(self.model, config)
        if self.use_focal_loss:
            self.bce = FocalLoss(gamma=self.focal_gamma)
            print(f"✨ Using Focal Loss (γ={self.focal_gamma}) instead of BCE")
        else:
            self.bce = nn.BCELoss()

        # Print curriculum settings
        if self.use_curriculum:
            print(f"📚 Curriculum Learning ENABLED: warmup={self.curriculum_warmup_epochs} epochs, PL boost={self.curriculum_pl_warmup_weight}x")

        # Pseudo-label loss if lambda_pl > 0
        if self.lambda_pl > 0:
            self.pl_crit = SoftPseudoLabelLoss(lambda_pl=self.lambda_pl)
        else:
            self.pl_crit = None

        # User text lookup (may be None)
        self.user_text: Optional[Dict[int, torch.Tensor]] = config.get("user_content_embed")

        # Ground truth pseudo-labels (for support groups)
        self.enhanced_pl_data = config.get("enhanced_pl_data", None)
        if self.enhanced_pl_data is not None:
            self.pseudo_label_matrix = self.enhanced_pl_data['pseudo_label_matrix']
            self.confidence_mask = self.enhanced_pl_data['confidence_mask']
            logger.info(f"✅ Loaded ground truth pseudo-labels: "
                       f"shape={self.pseudo_label_matrix.shape}, "
                       f"confident={self.confidence_mask.sum().item()}")
        else:
            self.pseudo_label_matrix = None
            self.confidence_mask = None

        # PL statistics tracking
        self.pl_stats = {
            'total_batches': 0,
            'pl_loss_sum': 0.0,
            'bce_loss_sum': 0.0,
            'pseudo_labels_used': 0,
        }

        # Tensorboard writer with organized directory structure
        self.alias = config.get("alias", "model")
        dataset = config.get("dataset", "unknown")

        # Extract model name from alias for TensorBoard logging
        if "_" in self.alias:
            parts = self.alias.split("_")
            # Find where dataset name appears and reconstruct without it
            model_parts = [p for p in parts if p not in ["support_groups_full_164", "support_groups_full_164_loo"]]
            model_seed = "_".join(model_parts)
        else:
            model_seed = self.alias
        self.writer = SummaryWriter(log_dir=f"runs/{dataset}/{model_seed}")

    def _predict(self, users: torch.Tensor, items: torch.Tensor,
                user_text_embed: Optional[torch.Tensor] = None):
        """Make predictions. Returns predictions and optional cos_pl for PL models."""
        if user_text_embed is not None:
            output = self.model(users, items, user_text_embed=user_text_embed)
        else:
            output = self.model(users, items)

        # Handle both baseline models (single tensor) and PL models (tuple)
        if isinstance(output, tuple):
            return output  # (predictions, cos_pl)
        else:
            return output  # just predictions

    def _lookup_user_text(self, users) -> Optional[torch.Tensor]:
        """Lookup user text embeddings - OPTIMIZED to avoid slow .tolist() on large tensors."""
        if self.user_text is None:
            return None

        # ✨ CRITICAL FIX: Avoid .tolist() on large CUDA tensors (extremely slow!)
        # Move to CPU first if on CUDA, then convert
        if isinstance(users, torch.Tensor):
            users_flat = users.flatten()
            if users_flat.is_cuda:
                users_flat = users_flat.cpu()
            ids = users_flat.numpy().tolist()  # numpy conversion much faster than torch .tolist()
            device = users.device
        else:
            ids = list(users)
            device = torch.device("cpu")

        # Create zero vector if needed
        if not hasattr(self, "_zero_text"):
            dim = self.config.get("user_text_dim", 0)
            self._zero_text = torch.zeros(dim, dtype=torch.float32, device="cpu")  # Keep on CPU initially

        # ✨ OPTIMIZED: Build list on CPU, then move entire batch to GPU at once
        vecs = []
        for u in ids:
            user_vec = self.user_text.get(int(u), self._zero_text)
            if not isinstance(user_vec, torch.Tensor):
                user_vec = torch.tensor(user_vec, dtype=torch.float32, device="cpu")
            elif user_vec.device != torch.device("cpu"):
                user_vec = user_vec.cpu()
            vecs.append(user_vec)

        if not vecs:
            return None

        # Stack on CPU first, then move to target device (much faster than moving each vector)
        stacked = torch.stack(vecs)
        if device != torch.device("cpu"):
            stacked = stacked.to(device)
        return stacked

    def _get_memory_info(self):
        """Get current GPU memory usage."""
        if torch.cuda.is_available():
            return {
                'allocated': torch.cuda.memory_allocated(),
                'reserved': torch.cuda.memory_reserved(),
                'max_allocated': torch.cuda.max_memory_allocated()
            }
        return {'allocated': 0, 'reserved': 0, 'max_allocated': 0}

    def _chunked_predict(self, users: torch.Tensor, items: torch.Tensor,
                        user_text_embed: Optional[torch.Tensor] = None, chunk_size: int = None):
        """
        Memory-efficient chunked prediction to avoid CUDA OOM errors.

        Args:
            users: User indices tensor
            items: Item indices tensor
            user_text_embed: Optional user text embeddings
            chunk_size: Size of chunks to process (default: self.max_eval_batch_size)

        Returns:
            Predictions tensor (concatenated from all chunks)
        """
        if chunk_size is None:
            chunk_size = self.max_eval_batch_size

        batch_size = len(users)

        # If batch is small enough, process normally
        if not self.enable_chunking or batch_size <= chunk_size:
            return self._predict(users, items, user_text_embed)

        # Log memory usage before chunking
        memory_info = self._get_memory_info()
        logger.info(f"Large batch detected ({batch_size} samples). "
                   f"GPU memory: {memory_info['allocated']/1e9:.2f}GB allocated, "
                   f"{memory_info['reserved']/1e9:.2f}GB reserved. "
                   f"Using chunked evaluation with chunk_size={chunk_size}")

        # Process in chunks
        all_predictions = []
        all_cos_pl = []

        for start_idx in range(0, batch_size, chunk_size):
            end_idx = min(start_idx + chunk_size, batch_size)

            # Extract chunk
            user_chunk = users[start_idx:end_idx]
            item_chunk = items[start_idx:end_idx]
            text_chunk = user_text_embed[start_idx:end_idx] if user_text_embed is not None else None

            # Predict for chunk
            chunk_output = self._predict(user_chunk, item_chunk, text_chunk)

            # Handle both baseline and PL model outputs
            if isinstance(chunk_output, tuple):
                predictions, cos_pl = chunk_output
                all_predictions.append(predictions)
                all_cos_pl.append(cos_pl)
            else:
                all_predictions.append(chunk_output)

        # Concatenate results
        final_predictions = torch.cat(all_predictions, dim=0)

        if all_cos_pl:
            final_cos_pl = torch.cat(all_cos_pl, dim=0)
            return final_predictions, final_cos_pl
        else:
            return final_predictions

    def train_single_batch(self, users, items, ratings):
        """
        Train a single batch with PROPER PL loss computation using ground truth pseudo-labels.
        """
        self.opt.zero_grad()

        # Get user text embeddings if available
        user_text_embed = self._lookup_user_text(users)

        # Forward pass
        model_output = self._predict(users, items, user_text_embed)

        # Handle both baseline and PL model outputs
        if isinstance(model_output, tuple):
            predictions, cos_pl = model_output
        else:
            predictions = model_output
            cos_pl = None

        # FIX 3.3: Check for invalid prediction values BEFORE computing loss
        if torch.isnan(predictions).any() or torch.isinf(predictions).any():
            print(f"⚠️  WARNING: Invalid predictions detected (NaN/Inf) - skipping batch")
            return 0.0, 0.0, 0.0

        # FIX 3.3: Clamp predictions to valid range for BCE
        predictions = torch.clamp(predictions, 1e-7, 1.0 - 1e-7)

        # Main BCE loss
        loss_bce = self.bce(predictions, ratings.float())

        # ✅ CRITICAL FIX: Compute ACTUAL PL loss with ground truth pseudo-labels!
        loss_pl = torch.tensor(0.0, device=predictions.device)

        if self.pl_crit is not None and hasattr(self.model, 'pl') and self.model.pl is not None:
            if self.pseudo_label_matrix is not None:
                # Use ground truth pseudo-labels from AlignFeatures
                try:
                    # Move pseudo-label matrix to same device as predictions
                    device = predictions.device
                    if self.pseudo_label_matrix.device != device:
                        self.pseudo_label_matrix = self.pseudo_label_matrix.to(device)
                        self.confidence_mask = self.confidence_mask.to(device)

                    # Extract pseudo-labels for batch items
                    batch_size = len(users)
                    batch_pseudo_labels = torch.zeros(batch_size, device=device)

                    for idx in range(batch_size):
                        user_id = users[idx].item()
                        item_id = items[idx].item()

                        # Check if we have a confident pseudo-label for this user-item pair
                        if user_id < self.pseudo_label_matrix.shape[0] and item_id < self.pseudo_label_matrix.shape[1]:
                            if self.confidence_mask[user_id, item_id]:
                                batch_pseudo_labels[idx] = self.pseudo_label_matrix[user_id, item_id]
                                self.pl_stats['pseudo_labels_used'] += 1

                    # FIX 3.3: Check for invalid pseudo-labels
                    if torch.isnan(batch_pseudo_labels).any() or torch.isinf(batch_pseudo_labels).any():
                        print(f"⚠️  WARNING: Invalid pseudo-labels detected (NaN/Inf) - skipping PL loss")
                        loss_pl = torch.tensor(0.0, device=predictions.device)
                    else:
                        # FIX 3.3: Clamp pseudo-labels to valid range
                        batch_pseudo_labels = torch.clamp(batch_pseudo_labels, 0.0, 1.0)

                        # Only compute PL loss for samples with confident pseudo-labels
                        has_pl = (batch_pseudo_labels > 0).float()
                        if has_pl.sum() > 0:
                            # Compute soft BCE loss with pseudo-labels
                            epsilon = 1e-8
                            predictions_clamped = torch.clamp(predictions, epsilon, 1 - epsilon)

                            # Weighted BCE: prioritize samples with pseudo-labels
                            pl_loss_raw = -(
                                batch_pseudo_labels * torch.log(predictions_clamped) +
                                (1 - batch_pseudo_labels) * torch.log(1 - predictions_clamped)
                            )

                            # Apply confidence mask as weight
                            loss_pl = (pl_loss_raw * has_pl).sum() / (has_pl.sum() + epsilon)
                            loss_pl = self.lambda_pl * loss_pl

                            # FIX 3.3: Check if PL loss is valid
                            if torch.isnan(loss_pl) or torch.isinf(loss_pl):
                                print(f"⚠️  WARNING: Invalid PL loss (NaN/Inf) - using zero")
                                loss_pl = torch.tensor(0.0, device=predictions.device)
                            else:
                                self.pl_stats['pl_loss_sum'] += loss_pl.item()

                except Exception as e:
                    logger.warning(f"Error computing PL loss: {e}. Continuing with BCE only.")
                    loss_pl = torch.tensor(0.0, device=predictions.device)

        # Update statistics
        self.pl_stats['total_batches'] += 1
        self.pl_stats['bce_loss_sum'] += loss_bce.item()

        # 🚀 v11: CURRICULUM LEARNING - Adjust loss weights based on epoch
        if self.use_curriculum and self.current_epoch < self.curriculum_warmup_epochs:
            # Warmup phase: Boost PL signal, reduce BCE weight
            # This prevents early overfitting to negatives
            warmup_progress = self.current_epoch / self.curriculum_warmup_epochs
            bce_weight = 0.3 + 0.7 * warmup_progress  # 0.3 → 1.0
            pl_weight = self.curriculum_pl_warmup_weight  # e.g., 2.0

            total_loss = bce_weight * loss_bce + pl_weight * loss_pl

            # Track curriculum weights (print only at start of epoch)
            if self.pl_stats['total_batches'] == 1:
                print(f"📚 Curriculum [Epoch {self.current_epoch}]: BCE_weight={bce_weight:.2f}, PL_weight={pl_weight:.2f}")
        else:
            # Standard training: equal weights
            total_loss = loss_bce + loss_pl

        # FIX 3.3: Final check for invalid total loss before backward pass
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"⚠️  WARNING: Invalid total loss (NaN/Inf) - skipping backward pass")
            return 0.0, loss_bce.item() if not torch.isnan(loss_bce) else 0.0, 0.0

        # Backward pass
        total_loss.backward()

        # 🚀 v11: GRADIENT CLIPPING - Prevent explosive gradients
        if self.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clip_norm)

        self.opt.step()

        return total_loss.item(), loss_bce.item(), loss_pl.item() if isinstance(loss_pl, torch.Tensor) else loss_pl

    def set_epoch(self, epoch: int):
        """Update current epoch for confidence scheduling."""
        self.current_epoch = epoch

    def print_pl_stats(self, epoch: int):
        """Print pseudo-label statistics."""
        if self.pl_stats['total_batches'] > 0:
            avg_bce = self.pl_stats['bce_loss_sum'] / self.pl_stats['total_batches']
            avg_pl = self.pl_stats['pl_loss_sum'] / self.pl_stats['total_batches']

            print(f"[Epoch {epoch}] Stats - BCE: {avg_bce:.4f}, PL: {avg_pl:.4f}, "
                  f"PL Used: {self.pl_stats['pseudo_labels_used']}", flush=True)

            # Log to tensorboard
            if self.writer is not None:
                try:
                    self.writer.add_scalar('Loss/BCE', avg_bce, epoch)
                    self.writer.add_scalar('Loss/PL', avg_pl, epoch)
                    self.writer.add_scalar('PL/Used', self.pl_stats['pseudo_labels_used'], epoch)
                except Exception:
                    pass  # Ignore tensorboard errors

        # Reset stats
        self.pl_stats = {
            'total_batches': 0,
            'pl_loss_sum': 0.0,
            'bce_loss_sum': 0.0,
            'pseudo_labels_used': 0,
        }

    def evaluate(self, evaluate_data, epoch_id, top_k=5):
        """Evaluate the model with memory-efficient chunking."""
        self.model.eval()

        with torch.no_grad():
            test_users, test_items, negative_users, negative_items = evaluate_data

            # ✨ OPTIMIZATION: Skip user text lookup for baseline models (lambda_pl=0)
            # This avoids expensive lookups for 600K+ samples in MovieLens/Goodbooks
            if self.lambda_pl > 0 and self.user_text is not None:
                test_user_text = self._lookup_user_text(test_users)
                neg_user_text = self._lookup_user_text(negative_users)
            else:
                test_user_text = None
                neg_user_text = None

            # Move tensors to device
            device = next(self.model.parameters()).device
            test_users = test_users.to(device)
            test_items = test_items.to(device)
            negative_users = negative_users.to(device)
            negative_items = negative_items.to(device)

            # Use chunked prediction to avoid CUDA OOM
            test_output = self._chunked_predict(test_users, test_items, test_user_text)
            neg_output = self._chunked_predict(negative_users, negative_items, neg_user_text)

            # Extract predictions (ignore cos_pl for evaluation)
            if isinstance(test_output, tuple):
                test_scores = test_output[0]
            else:
                test_scores = test_output

            if isinstance(neg_output, tuple):
                negative_scores = neg_output[0]
            else:
                negative_scores = neg_output

            # Ensure scores are 1D
            if test_scores.dim() > 1:
                test_scores = test_scores.squeeze()
            if negative_scores.dim() > 1:
                negative_scores = negative_scores.squeeze()

        self.model.train()

        # ✨ CRITICAL FIX: Ensure CUDA operations complete before CPU transfer
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Calculate metrics
        metron = MetronAtK(top_k=top_k)
        metron.subjects = [test_users.view(-1).cpu().numpy(),
                          test_items.view(-1).cpu().numpy(),
                          test_scores.view(-1).cpu().numpy(),
                          negative_users.view(-1).cpu().numpy(),
                          negative_items.view(-1).cpu().numpy(),
                          negative_scores.view(-1).cpu().numpy()]

        hr = metron.cal_hit_ratio()
        ndcg = metron.cal_ndcg()
        auc = metron.cal_auc()

        print(f'[Epoch {epoch_id:3d}] HR@{top_k} = {hr:.4f}, NDCG@{top_k} = {ndcg:.4f}, AUC = {auc:.4f}', flush=True)

        # Log to tensorboard - with error handling and flush
        if self.writer is not None:
            try:
                self.writer.add_scalar(f'HR@{top_k}', hr, epoch_id)
                self.writer.add_scalar(f'NDCG@{top_k}', ndcg, epoch_id)
                self.writer.add_scalar('AUC', auc, epoch_id)
                self.writer.flush()  # Force TensorBoard write to disk
            except Exception as e:
                print(f"Warning: TensorBoard logging failed: {e}", flush=True)

        return {'hr': hr, 'ndcg': ndcg, 'auc': auc, 'epoch': epoch_id}

    def save(self, alias, epoch_id, hit_ratio, ndcg):
        """Save model checkpoint."""
        model_dir = Path(self.config.get("model_dir", f"models/trained/{alias}_best.model")).parent
        model_dir.mkdir(parents=True, exist_ok=True)

        save_checkpoint(self.model, self.config["model_dir"])

        # Save metadata
        metadata = {
            'epoch': epoch_id,
            'hr@5': hit_ratio,
            'ndcg@5': ndcg,
            'alias': alias
        }

        metadata_path = str(self.config["model_dir"]).replace('.model', '.metadata')
        with open(metadata_path, 'w') as f:
            import json
            json.dump(metadata, f, indent=2)