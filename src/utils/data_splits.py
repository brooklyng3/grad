"""Deterministic, disjoint dataset splits for training and evaluation.

The Mirai feature matrix is ordered as 121,621 benign packets followed by
642,516 malicious packets.  Training the WGAN on the first N attack packets
(and evaluating on the same window) leaks training samples into the benchmark.
This module defines a single source of truth for the split rules so that code
and documentation cannot drift apart.

Attack packets are partitioned by a modulo rule on the attack-relative index:

* ``split="train"``      -> relative index % 3 == 0
* ``split="eval"``       -> relative index % 3 == 1
* ``split="holdout"``    -> relative index % 3 == 2

The holdout third is used only for checkpoint selection/test, so the published
eval partition is never used for model selection.  Benign packets after the
grace period are partitioned by parity for threshold calibration versus FPR
validation (calibration = even relative index, validation = odd).
"""

from typing import Union
import numpy as np

ATTACK_START: int = 121621
TOTAL_ATTACKS: int = 642516
ATTACK_SPLIT_STRIDE: int = 3
ATTACK_TRAIN_REMAINDER: int = 0
ATTACK_EVAL_REMAINDER: int = 1
ATTACK_HOLDOUT_REMAINDER: int = 2

BENIGN_GRACE: int = 55000
BENIGN_END: int = 121621
BENIGN_SPLIT_STRIDE: int = 2
BENIGN_CALIBRATION_REMAINDER: int = 0
BENIGN_VALIDATION_REMAINDER: int = 1


def attack_indices_for_split(
    attack_indices: Union[np.ndarray, list], split: str
) -> np.ndarray:
    """Return attack indices belonging to the requested split.

    Args:
        attack_indices: Sorted absolute indices of attack packets.
        split: ``"train"``, ``"eval"``, or ``"holdout"``.

    Returns:
        The subset of ``attack_indices`` assigned to the split.
    """
    remainders = {
        "train": ATTACK_TRAIN_REMAINDER,
        "eval": ATTACK_EVAL_REMAINDER,
        "holdout": ATTACK_HOLDOUT_REMAINDER,
    }
    if split not in remainders:
        raise ValueError(f"Unknown attack split: {split!r}")
    arr = np.asarray(attack_indices, dtype=np.int64)
    return arr[(arr - ATTACK_START) % ATTACK_SPLIT_STRIDE == remainders[split]]


def benign_indices_for_split(labels: np.ndarray, split: str) -> np.ndarray:
    """Return post-grace benign indices for calibration or validation.

    Args:
        labels: Full ground-truth label vector.
        split: Either ``"calibration"`` or ``"validation"``.

    Returns:
        Absolute packet indices assigned to the split.
    """
    if split not in {"calibration", "validation"}:
        raise ValueError(f"Unknown benign split: {split!r}")
    arr = np.where((labels == 0) & (np.arange(len(labels)) >= BENIGN_GRACE))[0]
    remainder = (
        BENIGN_CALIBRATION_REMAINDER
        if split == "calibration"
        else BENIGN_VALIDATION_REMAINDER
    )
    return arr[(arr - BENIGN_GRACE) % BENIGN_SPLIT_STRIDE == remainder]
