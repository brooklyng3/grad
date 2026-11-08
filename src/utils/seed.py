"""Central random-seed helper for reproducible training and evaluation."""

import random
from typing import Optional

import numpy as np
import torch


def seed_everything(seed: Optional[int] = 42) -> None:
    """Seed Python, NumPy, and PyTorch (CPU and CUDA) RNGs.

    Args:
        seed: Seed value.  ``None`` performs no seeding.
    """
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
