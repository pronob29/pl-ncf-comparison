#mf.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from pl import PLEmbedding, SoftPseudoLabelLoss

class MF(nn.Module):
    """Matrix Factorization with optional Pseudo-Label (PL) and content fusion."""

    def __init__(self, cfg: dict):
        """
        Args:
            cfg keys:
                num_users      : int
                num_items      : int
                latent_dim     : int         (embedding size)
                pl_dim         : int or None (PL branch dim, optional)
                user_text_dim  : int or None (optional)
                weight_init_gaussian : bool
        """
        super().__init__()
        self.cfg = cfg
        n_u, n_i = cfg["num_users"], cfg["num_items"]
        d_mf     = cfg["latent_dim"]
        d_pl     = cfg.get("pl_dim", None)
        self.has_pl = d_pl is not None and d_pl > 0

        # Main MF embeddings
        self.user_emb = nn.Embedding(n_u, d_mf)
        self.item_emb = nn.Embedding(n_i, d_mf)

        # Enhanced PL branch with configurable features (always enhanced)
        if self.has_pl:
            self.pl = PLEmbedding(
                num_users=n_u,
                num_items=n_i,
                dim=d_pl,
                temperature=cfg.get("pl_temperature", 2.0),
                confidence_threshold=cfg.get("pl_confidence_threshold", 0.8),
                debias_popularity=cfg.get("pl_debias_popularity", True),
                use_teacher_student=cfg.get("pl_use_teacher_student", True),
                teacher_momentum=cfg.get("pl_teacher_momentum", 0.995),
                stability_window=cfg.get("pl_stability_window", 3),
                stability_epsilon=cfg.get("pl_stability_epsilon", 0.05),
                hierarchy_mapping=cfg.get("pl_hierarchy_mapping", None),
                hierarchy_alpha=cfg.get("pl_hierarchy_alpha", 0.8),
                hierarchy_delta=cfg.get("pl_hierarchy_delta", 0.2),
                # Memory management for CUDA OOM prevention
                memory_threshold_gb=cfg.get("memory_threshold_gb", 2.0),
                max_tensor_elements=cfg.get("max_tensor_elements", 50_000_000),
            )
        else:
            self.pl = None

        # Optional content fusion (same as NeuMF)
        self.use_text = "user_text_dim" in cfg and cfg["user_text_dim"] is not None
        if self.has_pl and self.use_text:
            self.text_proj = nn.Linear(cfg["user_text_dim"], d_pl)
        else:
            self.text_proj = None

        # Output: sum main path and (optionally) PL path and content
        extra_feats = 0
        if self.has_pl:
            extra_feats += 1
            if self.use_text:
                extra_feats += 1
        self.affine_out = nn.Linear(1 + extra_feats, 1)

        # Optional Gaussian init
        if cfg.get("weight_init_gaussian", False):
            nn.init.normal_(self.user_emb.weight, mean=0.0, std=0.01)
            nn.init.normal_(self.item_emb.weight, mean=0.0, std=0.01)
            if self.text_proj is not None:
                nn.init.normal_(self.text_proj.weight, mean=0.0, std=0.01)
                nn.init.zeros_(self.text_proj.bias)

    def forward(self, user_idx, item_idx, user_text_embed=None):
        # Small utility to coerce any feature to shape (B,1)
        def _as_column(x: torch.Tensor) -> torch.Tensor:
            if x is None:
                return None
            if x.dim() == 1:
                return x.view(-1, 1)
            if x.dim() == 2 and x.shape[1] == 1:
                return x
            # Fallback: flatten feature dimensions to a single scalar via mean
            return x.view(x.shape[0], -1).mean(dim=1, keepdim=True)

        # Ensure indices are 1D to avoid 3D embeddings
        if isinstance(user_idx, torch.Tensor) and user_idx.dim() > 1:
            user_idx = user_idx.view(-1)
        if isinstance(item_idx, torch.Tensor) and item_idx.dim() > 1:
            item_idx = item_idx.view(-1)

        # Main MF score
        u_vec = self.user_emb(user_idx)
        i_vec = self.item_emb(item_idx)
        mf_score = (u_vec * i_vec).sum(dim=-1, keepdim=True)  # shape (B,1)

        feat_list = [_as_column(mf_score)]
        cos_pl = None

        if self.has_pl:
            # Get PL user and item embeddings and compute cosine similarity
            pl_user_emb = self.pl.user_emb(user_idx)  # (B, d_pl)
            pl_item_emb = self.pl.item_emb(item_idx)  # (B, d_pl)

            # Compute cosine similarity
            pl_user_norm = F.normalize(pl_user_emb, p=2, dim=1)
            pl_item_norm = F.normalize(pl_item_emb, p=2, dim=1)
            cos_pl = (pl_user_norm * pl_item_norm).sum(dim=1, keepdim=True)  # (B, 1)

            # Add cosine similarity to feature list
            feat_list.append(_as_column(cos_pl))
            if self.use_text and user_text_embed is not None:
                # Ensure user_text_embed has the right shape (B, text_dim)
                if user_text_embed.dim() == 1:
                    user_text_embed = user_text_embed.unsqueeze(0)

                # Ensure user_text_embed is on the same device as the model
                device = next(self.parameters()).device
                user_text_embed = user_text_embed.to(device)

                text_vec = self.text_proj(user_text_embed)  # shape (B, d_pl)
                item_vec_pl = self.pl.item_emb(item_idx)    # shape (B, d_pl)

                # Ensure both tensors have the same batch size
                if text_vec.shape[0] != item_vec_pl.shape[0]:
                    # If text_vec has batch size 1, expand it to match item_vec_pl
                    if text_vec.shape[0] == 1 and item_vec_pl.shape[0] > 1:
                        text_vec = text_vec.expand(item_vec_pl.shape[0], -1)
                    elif item_vec_pl.shape[0] == 1 and text_vec.shape[0] > 1:
                        item_vec_pl = item_vec_pl.expand(text_vec.shape[0], -1)

                cos_text_item = F.cosine_similarity(text_vec, item_vec_pl, dim=-1)  # shape (B,)
                feat_list.append(_as_column(cos_text_item))  # shape (B,1)

        fused = torch.cat(feat_list, dim=-1)
        y_hat = torch.sigmoid(self.affine_out(fused)).squeeze(-1)

        # Return format depends on whether this is a PL model or baseline
        if self.has_pl:
            return y_hat, cos_pl  # PL models return tuple for pseudo-label loss
        else:
            return y_hat  # Baseline models return just predictions

    def update_pl_teacher(self):
        """Update PL teacher embeddings if using teacher-student framework."""
        if self.has_pl and hasattr(self.pl, 'update_teacher'):
            self.pl.update_teacher()

# Engine remains unchanged
from engine import Engine
from utils import use_cuda

class MFEngine(Engine):
    def __init__(self, cfg: dict):
        self.model = MF(cfg)
        if cfg.get("use_cuda", False):
            use_cuda(True, cfg["device_id"])
            self.model.cuda()
        super().__init__(cfg)
        print(self.model)