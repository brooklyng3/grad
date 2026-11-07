"""
Gradient-based adversarial perturbation baselines for NIDS evasion.

Implements standard white-box adversarial evasion algorithms tailored for
continuous statistical feature representations (AfterImage 100D vectors)
targeting autoencoder anomaly detection (PyTorchKitNET):
  1. Fast Gradient Sign Method (FGSM): Minimizes reconstruction RMSE.
  2. DeepFool Baseline: Iteratively perturbs features until RMSE <= theta.

Both baselines enforce domain constraints:
  - Scaled by dynamic ranges (s_global = norm_max - norm_min).
  - Physical boundary projections (signed Covariance, PCC in [-1, 1], unsigned >= 0).
"""

from typing import Optional, Tuple, Dict
import torch
import torch.nn as nn

from src.utils.constraints import apply_domain_constraints


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


def evaluate_perturbation_sweep(
    model: nn.Module,
    x_detected: torch.Tensor,
    theta: float,
    scales: torch.Tensor,
    eps_list: list = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
) -> Dict[float, Dict[str, float]]:
    """Evaluates FGSM evasion rates across an epsilon budget sweep.

    Computes Conditional ASR on detected attack packets to observe the
    mathematical transition curve from 0% to ~100% evasion.

    Args:
        model: PyTorchKitNET model.
        x_detected: Batch of detected malicious samples (RMSE > theta).
        theta: Alert threshold.
        scales: Global dynamic range scale vector.
        eps_list: List of epsilon fractions to evaluate.

    Returns:
        Dictionary mapping epsilon -> metrics {conditional_asr, mean_rmse, ...}.
    """
    model.eval()
    results = {}
    num_samples = x_detected.size(0)

    for eps in eps_list:
        x_adv, rmse_adv = fgsm_attack(
            model=model,
            x=x_detected,
            eps=eps,
            scales=scales,
            target_threshold=theta,
            enforce_domain=True,
        )
        evaded = (rmse_adv <= theta).sum().item()
        asr = (evaded / num_samples) * 100.0

        results[eps] = {
            "conditional_asr": asr,
            "mean_rmse": rmse_adv.mean().item(),
            "median_rmse": rmse_adv.median().item(),
            "min_rmse": rmse_adv.min().item(),
            "max_rmse": rmse_adv.max().item(),
        }

    return results
