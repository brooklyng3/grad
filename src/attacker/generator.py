"""
Wasserstein GAN Generator for Polymorphic Feature Mutation.

Operates in normalized feature space:
    x_norm = (x_raw - norm_min) / scales

Supports two generation modes:
  1. 'additive': Bounded box perturbation delta_norm within [-eps, +eps].
  2. 'manifold': Manifold-shaping / target-envelope mode that dynamically
     attenuates extreme volumetric features down into the benign window [0, 1].
"""

from typing import List, Optional
import torch
import torch.nn as nn

from src.utils.constraints import enforce_feature_bounds


class PolymorphicGenerator(nn.Module):
    """Conditional WGAN Generator producing stealthy adversarial feature mutations."""

    def __init__(
        self,
        feature_dim: int = 100,
        latent_dim: int = 32,
        hidden_dims: List[int] = [128, 256, 128],
        max_eps: float = 0.5,
        mode: str = "additive",
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.max_eps = max_eps
        self.mode = mode

        # Input dimension: latent noise + conditioned malicious features
        input_dim = latent_dim + feature_dim

        layers: List[nn.Module] = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            prev_dim = h_dim

        # Final output layer generates normalized perturbation delta
        layers.append(nn.Linear(prev_dim, feature_dim))
        # Tanh bounds output perturbation delta to [-1.0, +1.0]
        layers.append(nn.Tanh())

        self.network = nn.Sequential(*layers)

    def forward(
        self,
        x_mal_norm: torch.Tensor,
        z: Optional[torch.Tensor] = None,
        eps: Optional[float] = None,
        mode: Optional[str] = None,
    ) -> torch.Tensor:
        """Generates perturbed normalized feature vector.

        Args:
            x_mal_norm: Normalized malicious feature batch [batch_size, 100].
            z: Latent noise vector [batch_size, latent_dim] (default: N(0, 1)).
            eps: Optional budget scaling factor. Defaults to self.max_eps.
            mode: Operational mode ('additive' or 'manifold'). Defaults to self.mode.

        Returns:
            Perturbed normalized feature tensor.
        """
        batch_size = x_mal_norm.size(0)
        budget = eps if eps is not None else self.max_eps
        active_mode = mode if mode is not None else self.mode

        if z is None:
            z = torch.randn(
                (batch_size, self.latent_dim),
                device=x_mal_norm.device,
                dtype=x_mal_norm.dtype,
            )

        # Concatenate latent noise with source attack features
        cond_input = torch.cat([z, x_mal_norm], dim=-1)

        # Raw tanh perturbation delta in [-1, +1]
        delta_raw = self.network(cond_input)

        if active_mode == "manifold":
            # Manifold shaping: dynamically scale perturbation with feature magnitude
            # so extreme volumetric rates (> 1.0) can be attenuated into benign [0, 1]
            dynamic_budget = torch.clamp(x_mal_norm, min=budget)
            delta_norm = delta_raw * dynamic_budget
            x_adv_norm = torch.clamp(x_mal_norm + delta_norm, min=0.0, max=1.0)
        else:
            # Additive box perturbation: [-budget, +budget]
            delta_norm = delta_raw * budget
            x_adv_norm = x_mal_norm + delta_norm

        return x_adv_norm

    def mutate_raw(
        self,
        x_mal_raw: torch.Tensor,
        norm_min: torch.Tensor,
        scales: torch.Tensor,
        eps: Optional[float] = None,
        z: Optional[torch.Tensor] = None,
        mode: Optional[str] = None,
    ) -> torch.Tensor:
        """Convenience method accepting raw unnormalized feature vectors.

        Normalizes inputs -> computes perturbation -> unnormalizes to raw feature space
        while enforcing signed statistical invariants (preserving negative covariances).

        Args:
            x_mal_raw: Raw feature tensor [batch_size, 100].
            norm_min: Global lower bound tensor [100].
            scales: Global dynamic range scale tensor [100].
            eps: Epsilon budget.
            z: Latent vector.
            mode: Optional generation mode ('additive' or 'manifold').

        Returns:
            Adversarially perturbed raw feature tensor [batch_size, 100].
        """
        # 1. Normalize
        x_norm = (x_mal_raw - norm_min) / scales

        # 2. Forward pass in normalized space
        x_adv_norm = self.forward(x_norm, z=z, eps=eps, mode=mode)

        # 3. Unnormalize and enforce domain bounds (preserving covariances)
        x_adv_raw = enforce_feature_bounds(x_adv_norm * scales + norm_min)
        return x_adv_raw
