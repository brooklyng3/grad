"""Smoke tests for the NumPy KitNET defender train/execute lifecycle."""

import sys

import numpy as np

sys.path.insert(0, "Kitsune-py")
from KitNET.KitNET import KitNET  # noqa: E402


def test_kitnet_returns_zero_during_grace():
    kitnet = KitNET(10, 3, 5, 10)
    np.random.seed(0)
    score = kitnet.process(np.random.rand(10))
    assert score == 0.0
    assert kitnet.v is None


def test_kitnet_trains_and_scores_after_grace():
    kitnet = KitNET(10, 3, 5, 10)
    np.random.seed(0)
    for _ in range(25):
        kitnet.process(np.random.rand(10))
    score = kitnet.execute(np.random.rand(10))
    assert np.isfinite(score)
    assert score >= 0.0
