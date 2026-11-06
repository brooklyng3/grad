"""
Network domain invariants, statistical feature bounds, and perturbation projection.

Enforces realistic physical network traffic constraints on adversarial
perturbations as mandated by Section 5 of AGENTS.md:
  1. Signed statistical features (Covariance and Pearson Correlation Coefficient).
  2. Relative / feature-range-scaled perturbation budgets (L_inf and L2).
  3. Non-negative bounds strictly on unsigned physical features (counts, weights,
     means, standard deviations, magnitudes, radii).
"""

from typing import Union, Optional, List
import numpy as np
import torch

# Physical network constant bounds
MTU_MAX_BYTES: int = 1500
MIN_PACKET_BYTES: int = 0
MIN_IAT: float = 0.0
TTL_MIN: int = 1
TTL_MAX: int = 255

# In Kitsune's 100-dimensional AfterImage feature set:
# 10 features represent Covariance (range: -inf to +inf)
COV_FEATURE_INDICES: List[int] = [20, 27, 34, 41, 48, 70, 77, 84, 91, 98]
# 10 features represent Pearson Correlation Coefficient (range: -1.0 to +1.0)
PCC_FEATURE_INDICES: List[int] = [21, 28, 35, 42, 49, 71, 78, 85, 92, 99]
SIGNED_FEATURE_INDICES: List[int] = sorted(COV_FEATURE_INDICES + PCC_FEATURE_INDICES)
UNSIGNED_FEATURE_INDICES: List[int] = [
    i for i in range(100) if i not in SIGNED_FEATURE_INDICES
]


def project_linf_ball(
    x_adv: Union[torch.Tensor, np.ndarray],
    x_orig: Union[torch.Tensor, np.ndarray],
    eps: float,
    feature_scales: Optional[Union[torch.Tensor, np.ndarray]] = None,
) -> Union[torch.Tensor, np.ndarray]:
    """Clamps x_adv to lie within the L_inf epsilon ball around x_orig.

    If feature_scales (e.g. s = x_max - x_min) is provided, eps acts as a
    relative perturbation budget: |delta_i| <= eps * s_i.
    Otherwise, eps acts as an absolute scalar bound.

    Args:
        x_adv: Adversarially modified feature tensor or array.
        x_orig: Original feature tensor or array.
        eps: Allowable perturbation fraction or magnitude.
        feature_scales: Optional per-feature dynamic range vector.

    Returns:
        Projected adversarial features satisfying L_inf constraints.
    """
    if eps <= 0.0:
        return x_orig

    if isinstance(x_adv, torch.Tensor):
        if feature_scales is not None:
            scales = (
                feature_scales
                if isinstance(feature_scales, torch.Tensor)
                else torch.tensor(
                    feature_scales, device=x_adv.device, dtype=x_adv.dtype
                )
            )
            bound = eps * torch.clamp(scales, min=1e-8)
            delta = torch.clamp(x_adv - x_orig, min=-bound, max=bound)
        else:
            delta = torch.clamp(x_adv - x_orig, min=-eps, max=eps)
        return x_orig + delta
    elif isinstance(x_adv, np.ndarray):
        if feature_scales is not None:
            scales = np.asarray(feature_scales, dtype=x_adv.dtype)
            bound = eps * np.maximum(scales, 1e-8)
            delta = np.clip(x_adv - x_orig, -bound, bound)
        else:
            delta = np.clip(x_adv - x_orig, -eps, eps)
        return x_orig + delta
    else:
        raise TypeError(f"Unsupported array type: {type(x_adv)}")


def project_l2_ball(
    x_adv: Union[torch.Tensor, np.ndarray],
    x_orig: Union[torch.Tensor, np.ndarray],
    delta_budget: float,
    feature_scales: Optional[Union[torch.Tensor, np.ndarray]] = None,
) -> Union[torch.Tensor, np.ndarray]:
    """Projects x_adv to lie within the L2 ball of radius delta_budget.

    If feature_scales is provided, projection is performed in normalized space.

    Args:
        x_adv: Adversarially modified feature tensor or array.
        x_orig: Original feature tensor or array.
        delta_budget: Maximum allowable Euclidean perturbation norm.
        feature_scales: Optional per-feature scale vector.

    Returns:
        Projected adversarial features satisfying L2 constraint.
    """
    if delta_budget <= 0.0:
        return x_orig

    if isinstance(x_adv, torch.Tensor):
        diff = x_adv - x_orig
        if feature_scales is not None:
            scales = (
                feature_scales
                if isinstance(feature_scales, torch.Tensor)
                else torch.tensor(
                    feature_scales, device=x_adv.device, dtype=x_adv.dtype
                )
            )
            scales = torch.clamp(scales, min=1e-8)
            scaled_diff = diff / scales
            norm = torch.norm(scaled_diff, p=2, dim=-1, keepdim=True)
            scale = torch.clamp(delta_budget / (norm + 1e-12), max=1.0)
            return x_orig + (scaled_diff * scale) * scales
        else:
            norm = torch.norm(diff, p=2, dim=-1, keepdim=True)
            scale = torch.clamp(delta_budget / (norm + 1e-12), max=1.0)
            return x_orig + diff * scale
    elif isinstance(x_adv, np.ndarray):
        diff = x_adv - x_orig
        if feature_scales is not None:
            scales = np.maximum(np.asarray(feature_scales, dtype=x_adv.dtype), 1e-8)
            scaled_diff = diff / scales
            norm = np.linalg.norm(scaled_diff, ord=2, axis=-1, keepdims=True)
            scale = np.minimum(delta_budget / (norm + 1e-12), 1.0)
            return x_orig + (scaled_diff * scale) * scales
        else:
            norm = np.linalg.norm(diff, ord=2, axis=-1, keepdims=True)
            scale = np.minimum(delta_budget / (norm + 1e-12), 1.0)
            return x_orig + diff * scale
    else:
        raise TypeError(f"Unsupported array type: {type(x_adv)}")


def enforce_feature_bounds(
    x: Union[torch.Tensor, np.ndarray],
    min_bounds: Optional[Union[torch.Tensor, np.ndarray]] = None,
    max_bounds: Optional[Union[torch.Tensor, np.ndarray]] = None,
    enforce_afterimage_domain: bool = True,
) -> Union[torch.Tensor, np.ndarray]:
    """Clamps features to valid statistical domain boundaries.

    Crucially preserves signed features:
      - Unsigned features (weights, means, std, radii, jitters) >= 0.0.
      - Covariance features (indices [20, 27, 34, 41, 48, 70, 77, 84, 91, 98])
        remain signed in (-inf, +inf).
      - PCC features (indices [21, 28, 35, 42, 49, 71, 78, 85, 92, 99])
        are bounded strictly in [-1.0, 1.0].

    Args:
        x: Feature tensor or numpy array [..., 100].
        min_bounds: Optional explicit lower bound vector.
        max_bounds: Optional explicit upper bound vector.
        enforce_afterimage_domain: Whether to apply AfterImage specific rules.

    Returns:
        Properly clamped feature tensor or numpy array.
    """
    if isinstance(x, torch.Tensor):
        out = x.clone()
        if enforce_afterimage_domain and x.shape[-1] == 100:
            # Clamp unsigned features to non-negative
            unsigned_idx = torch.tensor(UNSIGNED_FEATURE_INDICES, device=x.device)
            out.index_copy_(
                -1,
                unsigned_idx,
                torch.clamp(out.index_select(-1, unsigned_idx), min=0.0),
            )
            # Clamp PCC features to [-1.0, 1.0]
            pcc_idx = torch.tensor(PCC_FEATURE_INDICES, device=x.device)
            out.index_copy_(
                -1,
                pcc_idx,
                torch.clamp(out.index_select(-1, pcc_idx), min=-1.0, max=1.0),
            )
            # Covariance features are allowed to be negative; unconstrained
        elif min_bounds is None and not enforce_afterimage_domain:
            out = torch.clamp(out, min=0.0)

        if min_bounds is not None:
            min_t = (
                min_bounds
                if isinstance(min_bounds, torch.Tensor)
                else torch.tensor(min_bounds, device=x.device, dtype=x.dtype)
            )
            out = torch.maximum(out, min_t)

        if max_bounds is not None:
            max_t = (
                max_bounds
                if isinstance(max_bounds, torch.Tensor)
                else torch.tensor(max_bounds, device=x.device, dtype=x.dtype)
            )
            out = torch.minimum(out, max_t)

        return out

    elif isinstance(x, np.ndarray):
        out = x.copy()
        if enforce_afterimage_domain and x.shape[-1] == 100:
            out[..., UNSIGNED_FEATURE_INDICES] = np.maximum(
                out[..., UNSIGNED_FEATURE_INDICES], 0.0
            )
            out[..., PCC_FEATURE_INDICES] = np.clip(
                out[..., PCC_FEATURE_INDICES], -1.0, 1.0
            )
            # Covariances remain untouched
        elif min_bounds is None and not enforce_afterimage_domain:
            out = np.clip(out, 0.0, None)

        if min_bounds is not None:
            out = np.maximum(out, min_bounds)

        if max_bounds is not None:
            out = np.minimum(out, max_bounds)

        return out
    else:
        raise TypeError(f"Unsupported array type: {type(x)}")


def apply_domain_constraints(
    x_adv: Union[torch.Tensor, np.ndarray],
    x_orig: Union[torch.Tensor, np.ndarray],
    eps_linf: Optional[float] = None,
    delta_l2: Optional[float] = None,
    feature_scales: Optional[Union[torch.Tensor, np.ndarray]] = None,
    min_bounds: Optional[Union[torch.Tensor, np.ndarray]] = None,
    max_bounds: Optional[Union[torch.Tensor, np.ndarray]] = None,
    enforce_afterimage_domain: bool = True,
) -> Union[torch.Tensor, np.ndarray]:
    """Applies combined projection: scaled L_inf, scaled L2, and domain bounds.

    Args:
        x_adv: Candidate mutated feature batch.
        x_orig: Baseline original feature batch.
        eps_linf: Optional L_infinity perturbation threshold.
        delta_l2: Optional L_2 perturbation threshold.
        feature_scales: Dynamic range vector for relative budgeting.
        min_bounds: Explicit lower bound vector.
        max_bounds: Explicit upper bound vector.
        enforce_afterimage_domain: Whether to apply AfterImage signed feature rules.

    Returns:
        Feasible, perturbation-bounded feature vector.
    """
    out = x_adv
    if eps_linf is not None:
        out = project_linf_ball(out, x_orig, eps_linf, feature_scales=feature_scales)
    if delta_l2 is not None:
        out = project_l2_ball(out, x_orig, delta_l2, feature_scales=feature_scales)
    out = enforce_feature_bounds(
        out,
        min_bounds=min_bounds,
        max_bounds=max_bounds,
        enforce_afterimage_domain=enforce_afterimage_domain,
    )
    return out
