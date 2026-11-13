"""
Gradient-based adversarial perturbation baselines for NIDS evasion.

Implements white-box adversarial evasion algorithms tailored for continuous
statistical feature representations (AfterImage 100D vectors) targeting
autoencoder anomaly detection (PyTorchKitNET):
  1. Envelope projection: unbudgeted clamp into the benign normalisation window.
  2. Best-of-k WGAN retry: sample until a packet evades or the retry cap is hit.
  3. Fast Gradient Sign Method (FGSM): Minimizes reconstruction RMSE.
  4. PGD and DeepFool iterative baselines.

All routines enforce domain constraints:
  - Scaled by dynamic ranges (s_global = norm_max - norm_min).
  - Physical boundary projections (signed Covariance, PCC in [-1, 1],
    unsigned >= 0, MTU on packet-size means, IAT >= 0).
"""

from typing import Any, Optional, Tuple
import torch
import torch.nn as nn

from src.utils.constraints import apply_domain_constraints, enforce_feature_bounds


def envelope_projection(
    x: torch.Tensor,
    norm_min: torch.Tensor,
    norm_max: torch.Tensor,
    enforce_domain: bool = True,
) -> torch.Tensor:
    """Project raw features into the defender's benign normalisation envelope.

    This is the projection that does most of the work in the unbudgeted
    manifold attack.  It is exposed as a standalone baseline so its
    contribution can be measured apart from the generator.

    Args:
        x: Raw attack feature tensor [batch_size, 100].
        norm_min: Global benign lower bounds [100].
        norm_max: Global benign upper bounds [100].
        enforce_domain: Whether to re-apply AfterImage domain bounds.

    Returns:
        Features clamped into ``[norm_min, norm_max]``.
    """
    with torch.no_grad():
        x_adv = torch.clamp(x.detach(), min=norm_min, max=norm_max)
        if enforce_domain:
            x_adv = enforce_feature_bounds(x_adv)
    return x_adv


def wgan_best_of_k(
    generator: Any,
    model: nn.Module,
    x: torch.Tensor,
    theta: float,
    norm_min: torch.Tensor,
    scales: torch.Tensor,
    eps: float,
    mode: str = "additive",
    max_retries: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Iteratively sample the generator until each detected packet evades.

    Each retry draws a fresh latent vector.  The per-sample candidate with the
    lowest reconstruction error is retained, and sampling stops early when the
    whole batch has ``RMSE <= theta``.  This supplies the inference-time
    "mutate until bypass" behaviour that the original single-pass WGAN lacked.

    Args:
        generator: ``PolymorphicGenerator`` instance in eval mode.
        model: Trained ``PyTorchKitNET`` anomaly detector.
        x: Detected attack features [batch_size, 100].
        theta: Alert threshold.
        norm_min: Global lower bound vector.
        scales: Global dynamic range vector.
        eps: Perturbation budget in normalised space.
        mode: Generator mode, ``"additive"`` or ``"manifold"``.
        max_retries: Maximum number of generator draws per packet.

    Returns:
        Tuple of (best adversarial features, their RMSEs, per-sample attempts).
    """
    max_retries = max(1, int(max_retries))
    with torch.no_grad():
        best = generator.mutate_raw(x, norm_min, scales, eps=eps, mode=mode)
        best_rmse = model(best)
        attempts = torch.ones(x.size(0), dtype=torch.long, device=x.device)

        for _ in range(1, max_retries):
            pending = best_rmse > theta
            if not bool(pending.any()):
                break
            candidate = generator.mutate_raw(x, norm_min, scales, eps=eps, mode=mode)
            candidate_rmse = model(candidate)
            better = candidate_rmse < best_rmse
            best[better] = candidate[better]
            best_rmse[better] = candidate_rmse[better]
            attempts[pending] += 1
    return best, best_rmse, attempts


def wgan_refine(
    generator: Any,
    model: nn.Module,
    x: torch.Tensor,
    theta: float,
    norm_min: torch.Tensor,
    scales: torch.Tensor,
    eps: float,
    steps: int = 10,
    mode: str = "additive",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Iteratively feeds each mutated packet back into the generator.

    Tests the reviewer hypothesis that conditional mode collapse can be broken
    by multi-step refinement.  Each step samples from G(z_t, x_t), retaining the
    better result per sample.  Note that per-step epsilon accumulates, so total
    L_inf is up to ``steps * eps``.

    Args:
        generator: ``PolymorphicGenerator`` in eval mode.
        model: Anomaly detector.
        x: Original detected attack features.
        theta: Alert threshold.
        norm_min: Global lower bound vector.
        scales: Global dynamic range vector.
        eps: Per-step budget.
        steps: Number of refinement steps.
        mode: Generator mode.

    Returns:
        Tuple of (best features, their RMSEs, per-sample attempts).
    """
    steps = max(1, int(steps))
    with torch.no_grad():
        best = x.clone()
        best_rmse = model(best)
        attempts = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        for _ in range(steps):
            pending = best_rmse > theta
            if not bool(pending.any()):
                break
            candidate = generator.mutate_raw(best, norm_min, scales, eps=eps, mode=mode)
            candidate_rmse = model(candidate)
            better = candidate_rmse < best_rmse
            best[better] = candidate[better]
            best_rmse[better] = candidate_rmse[better]
            attempts[pending] += 1
    return best, best_rmse, attempts


def fgsm_attack(
    model: nn.Module,
    x: torch.Tensor,
    eps: float,
    scales: torch.Tensor,
    target_threshold: Optional[float] = None,
    enforce_domain: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fast Gradient Sign Method (FGSM) targeting Autoencoder Reconstruction RMSE.

    In standard classification, FGSM adds gradient to maximize loss.
    In NIDS evasion, the objective is to MINIMIZE reconstruction error (RMSE)
    so the anomaly score drops below the detection threshold theta:
        x_adv = x - eps * s_global * sign(grad_x(RMSE(x)))

    Args:
        model: Trained PyTorchKitNET anomaly detector.
        x: Input tensor of detected malicious feature vectors [batch_size, 100].
        eps: Relative L_infinity perturbation budget (fraction of dynamic range).
        scales: Per-feature dynamic range vector [100] (norm_max - norm_min).
        target_threshold: Optional threshold theta to measure evasion success.
        enforce_domain: Whether to project features onto domain constraints.

    Returns:
        Tuple of (x_adv, rmse_adv) where:
            x_adv: Adversarially perturbed feature tensor [batch_size, 100].
            rmse_adv: Reconstruction error after perturbation [batch_size].
    """
    x_input = x.clone().detach().requires_grad_(True)
    rmse = model(x_input)
    loss = rmse.sum()
    loss.backward()

    with torch.no_grad():
        grad = x_input.grad
        if grad is None:
            raise RuntimeError(
                "Model backward pass did not compute gradients for input x."
            )

        # Gradient sign descent to lower RMSE towards benign reconstruction
        direction = torch.sign(grad)
        # Scaled perturbation: delta = -eps * scales * sign(grad)
        step = eps * scales * direction
        x_candidate = x - step

        # Apply domain boundaries, non-negativity and PCC bounds
        if enforce_domain:
            x_adv = apply_domain_constraints(
                x_adv=x_candidate,
                x_orig=x,
                eps_linf=eps,
                feature_scales=scales,
                enforce_afterimage_domain=True,
            )
        else:
            x_adv = x_candidate

        rmse_adv = model(x_adv)

    return x_adv, rmse_adv


def pgd_attack(
    model: nn.Module,
    x: torch.Tensor,
    theta: float,
    scales: torch.Tensor,
    eps: float = 2.0,
    alpha: Optional[float] = None,
    num_steps: int = 50,
    manifold_projection: bool = False,
    norm_min: Optional[torch.Tensor] = None,
    norm_max: Optional[torch.Tensor] = None,
    enforce_domain: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Projected Gradient Descent (PGD) multi-step attack for Autoencoder evasion.

    Iteratively takes small gradient descent steps of size alpha, projecting onto
    either an L_inf epsilon ball or the benign manifold [norm_min, norm_max]:
        x_{t+1} = Proj(x_t - alpha * scales * sign(grad_x(RMSE(x_t))))

    Args:
        model: Trained PyTorchKitNET anomaly detector.
        x: Input tensor of detected malicious feature vectors [batch_size, 100].
        theta: Alert threshold.
        scales: Global dynamic range scale vector [100].
        eps: Maximum L_inf perturbation budget.
        alpha: Step size per iteration (defaults to eps / 10).
        num_steps: Number of PGD iterations.
        manifold_projection: If True, shapes traffic into benign [norm_min, norm_max].
        norm_min: Global lower bound tensor.
        norm_max: Global upper bound tensor.
        enforce_domain: Whether to enforce physical network domain boundaries.

    Returns:
        Tuple of (x_adv, rmse_adv) after num_steps iterations.
    """
    x_adv = x.clone().detach()
    step_size = alpha if alpha is not None else (eps / 10.0)

    for _ in range(num_steps):
        x_adv.requires_grad_(True)
        rmse = model(x_adv)
        loss = rmse.sum()
        loss.backward()

        with torch.no_grad():
            grad = x_adv.grad
            if grad is None:
                break

            # Descent step in direction of lower RMSE
            direction = torch.sign(grad)
            x_adv_new = x_adv - step_size * scales * direction

            if manifold_projection and norm_min is not None and norm_max is not None:
                # Shape features into benign operational window
                x_adv = torch.clamp(x_adv_new, min=norm_min, max=norm_max)
            else:
                # L_inf projection onto epsilon ball around original x
                delta = torch.clamp(x_adv_new - x, min=-eps * scales, max=eps * scales)
                x_adv = x + delta

            if enforce_domain:
                x_adv = apply_domain_constraints(
                    x_adv=x_adv,
                    x_orig=x,
                    eps_linf=None if manifold_projection else eps,
                    feature_scales=scales,
                    enforce_afterimage_domain=True,
                )
            x_adv = x_adv.detach()

    with torch.no_grad():
        final_rmse = model(x_adv)

    return x_adv, final_rmse


def deepfool_attack(
    model: nn.Module,
    x: torch.Tensor,
    theta: float,
    scales: torch.Tensor,
    max_iter: int = 50,
    overshoot: float = 0.02,
    step_size: float = 0.1,
    max_eps: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """DeepFool iterative perturbation algorithm for Autoencoder decision boundaries.

    Iteratively computes the minimal gradient step required to push detected samples
    across the decision boundary into the benign region (RMSE <= theta).

    Args:
        model: Trained PyTorchKitNET anomaly detector.
        x: Input tensor of detected malicious feature vectors [batch_size, 100].
        theta: Anomaly alert threshold.
        scales: Per-feature dynamic range vector [100].
        max_iter: Maximum number of gradient iterations.
        overshoot: Multiplicative factor pushing sample slightly beyond boundary.
        step_size: Learning rate / step scaling per iteration.
        max_eps: Optional maximum L_infinity budget cap.

    Returns:
        Tuple of (x_adv, rmse_adv, iterations_used).
    """
    x_adv = x.clone().detach()

    for iteration in range(max_iter):
        x_adv.requires_grad_(True)
        rmse = model(x_adv)

        # Check if all samples in batch successfully evaded
        with torch.no_grad():
            evaded_mask = rmse <= theta
            if torch.all(evaded_mask):
                return x_adv.detach(), rmse.detach(), iteration

        # Compute gradient of loss with respect to x_adv
        loss = rmse.sum()
        loss.backward()

        with torch.no_grad():
            grad = x_adv.grad
            if grad is None:
                break

            # Distance to decision boundary for samples still detected
            dist = torch.clamp(rmse - theta, min=0.0)  # [batch_size]

            # Gradient norm normalized by feature scales
            scaled_grad = grad / torch.clamp(scales, min=1e-8)
            grad_norm = torch.norm(scaled_grad, p=2, dim=-1, keepdim=True) + 1e-12

            # DeepFool minimal perturbation vector
            step = (dist.unsqueeze(-1) / grad_norm) * (
                grad / torch.clamp(scales, min=1e-8)
            )
            step = step * (1.0 + overshoot) * step_size * scales

            # Update adversarial candidate
            x_adv_new = x_adv - step

            # Constrain to domain rules and optional max_eps budget
            x_adv_clamped = apply_domain_constraints(
                x_adv=x_adv_new,
                x_orig=x,
                eps_linf=max_eps,
                feature_scales=scales,
                enforce_afterimage_domain=True,
            )
            x_adv = x_adv_clamped.detach()

    with torch.no_grad():
        final_rmse = model(x_adv)

    return x_adv, final_rmse, max_iter
