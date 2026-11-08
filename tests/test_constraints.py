"""
Unit tests for domain constraints and perturbation boundary projections.
"""

import numpy as np
import torch

from src.utils.constraints import (
    project_linf_ball,
    project_l2_ball,
    enforce_feature_bounds,
    apply_domain_constraints,
    COV_FEATURE_INDICES,
    PCC_FEATURE_INDICES,
)


def test_linf_projection_numpy():
    x_orig = np.array([[10.0, 20.0, 30.0]])
    x_adv = np.array([[15.0, 18.0, 22.0]])
    eps = 2.0

    projected = project_linf_ball(x_adv, x_orig, eps)
    assert np.allclose(projected, np.array([[12.0, 18.0, 28.0]]))
    assert np.max(np.abs(projected - x_orig)) <= eps + 1e-6


def test_linf_scaled_projection():
    # Test relative scaling with s = x_max - x_min
    x_orig = np.array([[1000.0, 0.01]])
    x_adv = np.array([[1100.0, 0.05]])
    scales = np.array([[2000.0, 0.1]])
    eps = 0.05  # 5% of dynamic range

    projected = project_linf_ball(x_adv, x_orig, eps, feature_scales=scales)
    # Feature 0 budget: 0.05 * 2000 = 100 -> bound [900, 1100]
    # Feature 1 budget: 0.05 * 0.1 = 0.005 -> bound [0.005, 0.015]
    assert np.isclose(projected[0, 0], 1100.0)
    assert np.isclose(projected[0, 1], 0.015)


def test_l2_projection_numpy():
    x_orig = np.array([[0.0, 0.0]])
    x_adv = np.array([[3.0, 4.0]])  # Norm is 5.0
    delta = 2.5

    projected = project_l2_ball(x_adv, x_orig, delta)
    norm = np.linalg.norm(projected - x_orig, axis=-1)
    assert np.isclose(norm[0], delta)
    assert np.allclose(projected, np.array([[1.5, 2.0]]))


def test_l2_projection_torch():
    x_orig = torch.tensor([[1.0, 1.0]])
    x_adv = torch.tensor([[4.0, 5.0]])  # Diff: [3, 4], Norm is 5.0
    delta = 1.0

    projected = project_l2_ball(x_adv, x_orig, delta)
    diff_norm = torch.norm(projected - x_orig, p=2, dim=-1).item()
    assert np.isclose(diff_norm, delta, atol=1e-5)


def test_afterimage_signed_features_preserved():
    # 100-feature vector with negative values in Covariance and PCC
    x = np.ones((1, 100)) * 50.0

    # Set Covariance and PCC to negative values
    for cov_i in COV_FEATURE_INDICES:
        x[0, cov_i] = -125.4
    for pcc_i in PCC_FEATURE_INDICES:
        x[0, pcc_i] = -0.75

    # Set an unsigned feature to negative (should be clamped to 0)
    x[0, 0] = -10.0

    bounded = enforce_feature_bounds(x, enforce_afterimage_domain=True)

    # Unsigned feature clamped
    assert bounded[0, 0] == 0.0
    # Covariance negative value preserved!
    for cov_i in COV_FEATURE_INDICES:
        assert bounded[0, cov_i] == -125.4
    # PCC negative value in [-1, 1] preserved!
    for pcc_i in PCC_FEATURE_INDICES:
        assert bounded[0, pcc_i] == -0.75


def test_mtu_and_iat_bounds_enforced():
    from src.utils.constraints import SIZE_MEAN_INDICES, IAT_STAT_INDICES

    x = np.ones((1, 100)) * 100.0
    x[0, SIZE_MEAN_INDICES] = 4000.0
    x[0, IAT_STAT_INDICES] = -5.0

    bounded = enforce_feature_bounds(x)
    assert np.all(bounded[0, SIZE_MEAN_INDICES] == 1500.0)
    assert np.all(bounded[0, IAT_STAT_INDICES] == 0.0)


def test_apply_combined_constraints_torch():
    x_orig = torch.zeros((1, 100))
    x_adv = torch.ones((1, 100)) * 10.0
    eps = 0.5

    out = apply_domain_constraints(x_adv, x_orig, eps_linf=eps)
    assert torch.max(torch.abs(out - x_orig)).item() <= eps + 1e-6
