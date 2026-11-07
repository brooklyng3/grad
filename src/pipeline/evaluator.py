"""
Evaluation and Evasion Benchmarker Engine (Phase 4).

Streams and evaluates network traffic through the defender (PyTorchKitNET) under
clean baseline conditions and polymorphic adversarial mutations (WGAN-GP, PGD, FGSM).

Enforces scientific reporting rigor:
  1. Conditional Adversarial Success Rate (ASR_cond):
         ASR_cond = (Detected Packets with RMSE_adv <= theta) / Num_Detected_Packets
  2. Overall Evasion Rate (ER_total):
         ER_total = (Pre-evading Packets + Successfully Mutated Packets) / Total_Attacks
  3. Feature distortion distances (L0, L1, L2, Linf).
"""

import os
import sys
from typing import Dict, Any, Optional
import numpy as np
import torch

# Ensure project root and Kitsune-py are importable
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("Kitsune-py"))

from src.defender.torch_kitnet import PyTorchKitNET  # noqa: E402


class PipelineEvaluator:
    """Evaluates NIDS evasion rates, detection metrics, and perturbation costs."""

    def __init__(
        self,
        model: PyTorchKitNET,
        theta: float,
        device: torch.device,
        batch_size: int = 512,
    ) -> None:
        self.model = model
        self.theta = theta
        self.device = device
        self.batch_size = batch_size
        self.norm_min, self.norm_max, self.scales = model.get_global_feature_bounds(
            as_torch=True
        )

    @torch.no_grad()
    def score_batches(self, x: torch.Tensor) -> np.ndarray:
        """Scores a feature tensor in mini-batches on device, returning RMSE array."""
        self.model.eval()
        scores = []
        num_samples = x.size(0)

        for i in range(0, num_samples, self.batch_size):
            batch = x[i : i + self.batch_size].to(self.device)
            batch_rmse = self.model(batch)
            scores.append(batch_rmse.cpu().numpy())

        return np.concatenate(scores)

    def evaluate_evasion(
        self,
        x_orig: torch.Tensor,
        x_adv: torch.Tensor,
        num_pre_evaded: int = 0,
        total_attack_population: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Calculates comprehensive evasion and perturbation distance metrics.

        Args:
            x_orig: Original detected malicious features [N, 100].
            x_adv: Adversarially perturbed features [N, 100].
            num_pre_evaded: Number of benign-resembling attack packets.
            total_attack_population: Total attack population (e.g. 642,516).

        Returns:
            Dictionary containing ASR_cond, ER_total, distortion norms, and RMSE.
        """
        assert (
            x_orig.shape == x_adv.shape
        ), "Shape mismatch between original and perturbed tensors"
        num_detected = x_orig.size(0)

        # 1. Compute reconstruction errors
        rmse_orig = self.score_batches(x_orig)
        rmse_adv = self.score_batches(x_adv)

        # 2. Evasion success counts
        evaded_mask = rmse_adv <= self.theta
        num_evaded = int(np.sum(evaded_mask))
        cond_asr = (num_evaded / (num_detected + 1e-12)) * 100.0

        if total_attack_population is not None:
            overall_er = (
                (num_pre_evaded + num_evaded) / float(total_attack_population)
            ) * 100.0
        else:
            overall_er = cond_asr

        # 3. Normalized perturbation distance metrics
        diff_raw = (x_adv - x_orig).cpu()
        scales_cpu = self.scales.cpu()
        diff_norm = diff_raw / torch.clamp(scales_cpu, min=1e-8)

        # L_infinity (maximum per-feature normalized deviation)
        linf_per_sample = torch.max(torch.abs(diff_norm), dim=-1)[0].numpy()
        # L2 norm (normalized Euclidean perturbation energy)
        l2_per_sample = torch.norm(diff_norm, p=2, dim=-1).numpy()
        # L1 norm (average normalized deviation)
        l1_per_sample = torch.norm(diff_norm, p=1, dim=-1).numpy()
        # L0 norm (fraction of modified features with |diff| > 1e-5)
        l0_per_sample = torch.sum(torch.abs(diff_raw) > 1e-5, dim=-1).float().numpy()

        return {
            "num_detected_evaluated": num_detected,
            "num_successfully_evaded": num_evaded,
            "conditional_asr_pct": float(cond_asr),
            "overall_evasion_rate_pct": float(overall_er),
            "rmse_metrics": {
                "initial_mean": float(np.mean(rmse_orig)),
                "initial_median": float(np.median(rmse_orig)),
                "adversarial_mean": float(np.mean(rmse_adv)),
                "adversarial_median": float(np.median(rmse_adv)),
                "adversarial_min": float(np.min(rmse_adv)),
                "adversarial_max": float(np.max(rmse_adv)),
            },
            "perturbation_distances": {
                "mean_linf": float(np.mean(linf_per_sample)),
                "max_linf": float(np.max(linf_per_sample)),
                "mean_l2": float(np.mean(l2_per_sample)),
                "max_l2": float(np.max(l2_per_sample)),
                "mean_l1": float(np.mean(l1_per_sample)),
                "mean_l0_modified_features": float(np.mean(l0_per_sample)),
            },
        }
