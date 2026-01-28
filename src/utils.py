#utils.py

from __future__ import annotations
from pathlib import Path
from typing import Dict, Iterable, Union

import torch
import torch.nn as nn
import torch.optim as optim

def save_checkpoint(model: nn.Module, path: Union[str, Path]) -> None:
    """
    Serialize only `model.state_dict()` to `path` (creates parent dirs).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def resume_checkpoint(
    path: Union[str, Path],
    model: nn.Module | None = None,
    optimizer: optim.Optimizer | None = None,
    map_location: torch.device | None = None
) -> dict:
    """
    Load a checkpoint, explicitly in weights_only mode to avoid untrusted pickles.

    - If the checkpoint is just a bare state_dict, it will be loaded directly.
    - If it's a dict containing 'state_dict' and/or 'optimizer', those keys will be used.
    """
    if map_location is None:
        map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Loading checkpoint from {path} → {map_location}")
    # explicitly limit to tensors only
    ckpt = torch.load(path, map_location=map_location, weights_only=True)

    # extract the actual state_dict
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt  # assume it *is* the state_dict

    if model is not None:
        model.load_state_dict(state_dict)

    if optimizer is not None and isinstance(ckpt, dict) and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])

    return ckpt


# ──────────────────────────────────────────────────────────────────────
#  CUDA helper
# ──────────────────────────────────────────────────────────────────────
def use_cuda(enabled: bool, device_id: int = 0) -> torch.device:
    """
    Enable CUDA and set the active device.

    Returns
    -------
    torch.device – the device to use for `.to(device)`
    """
    if enabled:
        if not torch.cuda.is_available():
            raise EnvironmentError("CUDA requested but not available.")
        torch.cuda.set_device(device_id)
        return torch.device(f"cuda:{device_id}")
    return torch.device("cpu")


# ──────────────────────────────────────────────────────────────────────
#  Optimizer factory
# ──────────────────────────────────────────────────────────────────────
def use_optimizer(
    network: nn.Module,
    cfg: Dict,
    param_groups: Iterable[dict] | None = None,
) -> optim.Optimizer:
    """
    Build an optimizer from a minimal, unified config:

    cfg = {
        "optimizer"    : "adamw",        # 'sgd' | 'adam' | 'adamw' | 'rmsprop'
        "lr"           : 1e-3,
        "weight_decay" : 1e-5,
        "momentum"     : 0.9,            # for SGD / RMSprop
        "betas"        : (0.9, 0.999),   # for Adam / AdamW
        …
    }
    """
    name = cfg["optimizer"].lower()
    lr   = cfg.get("lr", 1e-3)
    wd   = cfg.get("weight_decay", 0.0)
    params = param_groups if param_groups is not None else network.parameters()

    if name == "sgd":
        return optim.SGD(params, lr=lr,
                         momentum=cfg.get("momentum", 0.9),
                         weight_decay=wd)

    if name == "adam":
        return optim.Adam(params, lr=lr,
                          betas=cfg.get("betas", (0.9, 0.999)),
                          weight_decay=wd)

    if name == "adamw":
        return optim.AdamW(params, lr=lr,
                           betas=cfg.get("betas", (0.9, 0.999)),
                           weight_decay=wd)

    if name == "rmsprop":
        return optim.RMSprop(params, lr=lr,
                              alpha=cfg.get("alpha", 0.99),
                              momentum=cfg.get("momentum", 0.0),
                              weight_decay=wd)

    raise ValueError(f"Unknown optimizer: {cfg['optimizer']}")