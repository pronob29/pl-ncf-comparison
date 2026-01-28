"""
mlp.py ──────────────────────────────────────────────────────────────────────────
Multi-Layer Perceptron branch for the PL-NCF framework.

* No GMF/MF path:   the interaction signal comes from the MLP tower only.
* Optional warm-start: user/item embeddings can be initialised from a
  PLEmbedding checkpoint trained with the pseudo-label objective.

Author: Pronob, 2025-04-30
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from pl import PLEmbedding, SoftPseudoLabelLoss  # Enhanced pseudo-labeling for support groups
from engine import Engine
from utils import use_cuda, resume_checkpoint


# ╭─────────────────────────────── Model ───────────────────────────────╮
class MLP(nn.Module):
    """
    User-item interaction model identical to the 'MLP' component in NeuMF,
    but without the parallel GMF branch.

    Config dictionary expects:
        num_users          : int
        num_items          : int
        latent_dim         : int       (embedding size for this MLP)
        layers             : List[int] (including input_dim = 2*latent_dim)
        weight_init_gaussian : bool
    """

    def __init__(self, config: dict):
        super().__init__()
        self.cfg = config
        d_lat = config["latent_dim"]

        # ── Embeddings ────────────────────────────────────────────────────
        self.user_emb = nn.Embedding(config["num_users"], d_lat)
        self.item_emb = nn.Embedding(config["num_items"], d_lat)

        # ── Fully-connected tower ────────────────────────────────────────
        fc_sizes: List[int] = config["layers"]
        if fc_sizes[0] != 2 * d_lat:
            raise ValueError(
                f"layers[0] must equal 2*latent_dim ({2*d_lat}), got {fc_sizes[0]}"
            )

        self.fc_layers = nn.ModuleList(
            [nn.Linear(in_f, out_f) for in_f, out_f in zip(fc_sizes[:-1], fc_sizes[1:])]
        )

        self.affine_out = nn.Linear(fc_sizes[-1], 1)

        # ── Optional Gaussian initialisation ─────────────────────────────
        if config.get("weight_init_gaussian", False):
            for m in self.modules():
                if isinstance(m, (nn.Linear, nn.Embedding)):
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)

    # ─────────────────────────────────────────────────────────────────────
    def forward(self, u_idx: torch.LongTensor, i_idx: torch.LongTensor, user_text_embed=None) -> torch.Tensor:
        """Return σ(hᵤᵢ)  ∈ (0,1)."""
        # Note: user_text_embed is not used in base MLP, but accepted for consistency
        # Ensure indices are 1D to avoid 3D embeddings
        if isinstance(u_idx, torch.Tensor) and u_idx.dim() > 1:
            u_idx = u_idx.view(-1)
        if isinstance(i_idx, torch.Tensor) and i_idx.dim() > 1:
            i_idx = i_idx.view(-1)

        u_vec = self.user_emb(u_idx)
        i_vec = self.item_emb(i_idx)
        x = torch.cat([u_vec, i_vec], dim=-1)

        for fc in self.fc_layers:
            x = F.relu(fc(x), inplace=True)

        logit = self.affine_out(x)
        return torch.sigmoid(logit).squeeze(-1)

    # -------------------------------------------------------------------
    def load_pretrain_embeddings(self) -> None:
        """
        Warm-start `self.user_emb` and `self.item_emb` from a checkpoint
        produced by *pl.py::PLEmbedding*.

        Required extra keys in `self.cfg`:
            pretrain_pl : str   (directory or file containing state_dict)
            device_id   : int
        """
        cfg = self.cfg
        if "pretrain_pl" not in cfg:
            raise KeyError("config must supply 'pretrain_pl' for pre-training.")

        pl_model = PLEmbedding(
            num_users=cfg["num_users"],
            num_items=cfg["num_items"],
            dim=cfg["latent_dim"],
            trainable=False,            # weights are copied; no grads needed
        )

        if cfg.get("use_cuda", False):
            pl_model.cuda()

        resume_checkpoint(pl_model, model_dir=cfg["pretrain_pl"], device_id=cfg["device_id"])

        assert (
            pl_model.user_emb.weight.shape == self.user_emb.weight.shape
        ), "Latent dimension mismatch between PL checkpoint and current MLP."

        self.user_emb.weight.data.copy_(pl_model.user_emb.weight.data)
        self.item_emb.weight.data.copy_(pl_model.item_emb.weight.data)


# ╭─────────────────────────────── MLPPL ───────────────────────────────╮
class MLPPL(nn.Module):
    """
    Multi-Layer Perceptron with Pseudo-Label (PL) capability.
    
    Combines the MLP architecture with pseudo-labeling branch similar to MF+PL.
    
    Config dictionary expects:
        num_users          : int
        num_items          : int
        latent_dim         : int       (embedding size for MLP)
        pl_dim             : int       (PL branch dimension, e.g., 32)
        layers             : List[int] (including input_dim = 2*latent_dim)
        weight_init_gaussian : bool
        lambda_pl          : float     (PL loss weight, e.g., 0.1)
        pl_loss            : str       (PL loss type, e.g., "bce")
    """

    def __init__(self, config: dict):
        super().__init__()
        self.cfg = config
        d_lat = config["latent_dim"]
        d_pl = config["pl_dim"]

        # ── Main MLP Embeddings ──────────────────────────────────────────────
        self.user_emb = nn.Embedding(config["num_users"], d_lat)
        self.item_emb = nn.Embedding(config["num_items"], d_lat)

        # ── Enhanced PL Branch with configurable features ─────────────────────
        self.pl = PLEmbedding(
            num_users=config["num_users"],
            num_items=config["num_items"],
            dim=d_pl,
            use_enhanced_features=config.get("pl_use_enhanced_features", False),
            temperature=config.get("pl_temperature", 1.0),
            confidence_threshold=config.get("pl_confidence_threshold", 0.8),
            debias_popularity=config.get("pl_debias_popularity", False),
            use_teacher_student=config.get("pl_use_teacher_student", False),
            teacher_momentum=config.get("pl_teacher_momentum", 0.995),
            stability_window=config.get("pl_stability_window", 3),
            stability_epsilon=config.get("pl_stability_epsilon", 0.05),
            hierarchy_mapping=config.get("pl_hierarchy_mapping", None),
            hierarchy_alpha=config.get("pl_hierarchy_alpha", 0.8),
            hierarchy_delta=config.get("pl_hierarchy_delta", 0.2),
            # Memory management for CUDA OOM prevention
            memory_threshold_gb=config.get("memory_threshold_gb", 2.0),
            max_tensor_elements=config.get("max_tensor_elements", 50_000_000),
        )

        # ── Fully-connected tower ────────────────────────────────────────────
        fc_sizes: List[int] = config["layers"]
        if fc_sizes[0] != 2 * d_lat:
            raise ValueError(
                f"layers[0] must equal 2*latent_dim ({2*d_lat}), got {fc_sizes[0]}"
            )

        self.fc_layers = nn.ModuleList(
            [nn.Linear(in_f, out_f) for in_f, out_f in zip(fc_sizes[:-1], fc_sizes[1:])]
        )

        # ── Output layer: combine MLP + PL ────────────────────────────────────
        self.affine_out = nn.Linear(fc_sizes[-1] + 1, 1)  # +1 for PL score

        # ── Optional Gaussian initialisation ─────────────────────────────────
        if config.get("weight_init_gaussian", False):
            for m in self.modules():
                if isinstance(m, (nn.Linear, nn.Embedding)):
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)

    # ─────────────────────────────────────────────────────────────────────
    def forward(self, u_idx: torch.LongTensor, i_idx: torch.LongTensor, user_text_embed=None) -> torch.Tensor:
        """Return (prediction, cos_pl) where prediction ∈ (0,1) and cos_pl is PL score."""
        # Note: user_text_embed is not used in MLP+PL, but accepted for consistency
        # Ensure indices are 1D to avoid 3D embeddings
        if isinstance(u_idx, torch.Tensor) and u_idx.dim() > 1:
            u_idx = u_idx.view(-1)
        if isinstance(i_idx, torch.Tensor) and i_idx.dim() > 1:
            i_idx = i_idx.view(-1)

        # MLP path
        u_vec = self.user_emb(u_idx)
        i_vec = self.item_emb(i_idx)
        x = torch.cat([u_vec, i_vec], dim=-1)

        for fc in self.fc_layers:
            x = F.relu(fc(x), inplace=True)

        # PL path with unified interface
        use_enhanced = self.pl.use_enhanced_features
        pl_result = self.pl(
            u_idx, i_idx,
            return_confidence=use_enhanced,
            return_stability=use_enhanced,
            use_teacher=False,  # Use student for training
            apply_hierarchy=use_enhanced
        )

        # Handle unified return format
        cos_pl_matrix = pl_result[0]  # Always first element [batch_size, num_items]

        # Extract pseudo-labels for specific user-item pairs
        cos_pl = cos_pl_matrix[torch.arange(len(u_idx)), i_idx]  # [batch_size]

        # Combine MLP output with PL score
        combined = torch.cat([x, cos_pl.unsqueeze(-1)], dim=-1)
        logit = self.affine_out(combined)
        
        return torch.sigmoid(logit).squeeze(-1), cos_pl

    # -------------------------------------------------------------------
    def load_pretrain_embeddings(self) -> None:
        """
        Warm-start `self.user_emb` and `self.item_emb` from a checkpoint
        produced by *pl.py::PLEmbedding*.

        Required extra keys in `self.cfg`:
            pretrain_pl : str   (directory or file containing state_dict)
            device_id   : int
        """
        cfg = self.cfg
        if "pretrain_pl" not in cfg:
            raise KeyError("config must supply 'pretrain_pl' for pre-training.")

        pl_model = PLEmbedding(
            num_users=cfg["num_users"],
            num_items=cfg["num_items"],
            dim=cfg["latent_dim"],
            trainable=False,            # weights are copied; no grads needed
        )

        if cfg.get("use_cuda", False):
            pl_model.cuda()

        resume_checkpoint(pl_model, model_dir=cfg["pretrain_pl"], device_id=cfg["device_id"])

        assert (
            pl_model.user_emb.weight.shape == self.user_emb.weight.shape
        ), "Latent dimension mismatch between PL checkpoint and current MLP."

        self.user_emb.weight.data.copy_(pl_model.user_emb.weight.data)
        self.item_emb.weight.data.copy_(pl_model.item_emb.weight.data)

    def update_pl_teacher(self):
        """Update PL teacher embeddings if using teacher-student framework."""
        if hasattr(self, 'pl') and hasattr(self.pl, 'update_teacher'):
            self.pl.update_teacher()


# ╭────────────────────────────── Engine ───────────────────────────────╮
class MLPEngine(Engine):
    """Thin wrapper adding CUDA placement & optional PL warm-start."""

    def __init__(self, config: dict):
        # Determine which model to use based on pl_dim
        if config.get("pl_dim") is not None:
            self.model = MLPPL(config)
        else:
            self.model = MLP(config)

        # move to GPU if requested
        if config.get("use_cuda", False):
            use_cuda(True, config["device_id"])
            self.model.cuda()

        super().__init__(config)        # sets criterion, optimiser, etc.
        print(self.model)

        # optional warm-start
        if config.get("pretrain", False):
            self.model.load_pretrain_embeddings()
