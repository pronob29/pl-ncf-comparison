"""
Core source code modules for Neural Collaborative Filtering

This package contains all the core functionality for the NCF implementation
including models, training engines, data handling, and evaluation metrics.
"""

# Core modules
from . import config
from . import data
from . import engine
from . import metrics
from . import utils

# Model modules
from . import mf
from . import mlp
from . import neumf
from . import pl

__all__ = [
    "config",
    "data",
    "engine",
    "metrics",
    "utils",
    "mf",
    "mlp",
    "neumf",
    "pl"
]