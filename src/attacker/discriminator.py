"""
Wasserstein GAN Critic (Discriminator) with Gradient Penalty.

Operates strictly in normalized feature space to prevent gradient explosion
across unnormalized network feature scales:
    x_norm = (x - norm_min) / scales

Enforces 1-Lipschitz continuity via the Gradient Penalty:
    L_gp = E[ (||grad_x_hat D(x_hat)||_2 - 1)^2 ]
"""

from typing import List
import torch
import torch.nn as nn


class WGANDiscriminator(nn.Module):
    """Wasserstein Critic scoring realism of normalized feature vectors."""

    def __init__(
        self,
        feature_dim: int = 100,
        hidden_dims: List[int] = [128, 256, 128],
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim

        layers: List[nn.Module] = []
        prev_dim = feature_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            # LayerNorm preserves sample-wise gradient penalty in WGAN-GP
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            prev_dim = h_dim

        # Final scalar critic score (no sigmoid, unconstrained linear output)
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Computes Wasserstein critic score for normalized feature vectors.

        Args:
            x_norm: Normalized feature tensor [batch_size, 100].

        Returns:
            Scalar realism score tensor [batch_size, 1]. Higher is more benign.
        """
        return self.network(x_norm)


def compute_gradient_penalty(
    critic: nn.Module,
    real_samples: torch.Tensor,
    fake_samples: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Calculates 1-Lipschitz Gradient Penalty for WGAN-GP.

    Interpolates between real benign samples and fake generated adversarial samples:
        x_hat = epsilon * real + (1 - epsilon) * fake
    and penalizes squared distance of ||grad_x_hat D(x_hat)||_2 from 1.0.

    Args:
        critic: WGANDiscriminator module.
        real_samples: Real benign normalized feature tensor [batch_size, 100].
        fake_samples: Adversarial normalized feature tensor [batch_size, 100].
        device: CUDA or CPU torch device.

    Returns:
        Scalar gradient penalty loss tensor.
    """
    batch_size = real_samples.size(0)

    # Random uniform weight epsilon for each batch element
    epsilon = torch.rand((batch_size, 1), device=device, dtype=real_samples.dtype)

    # Linear interpolation
    interpolates = (
        epsilon * real_samples + ((1 - epsilon) * fake_samples)
    ).requires_grad_(True)

    # Critic forward pass on interpolated samples
    d_interpolates = critic(interpolates)

    # Gradient of Critic output with respect to interpolated inputs
    fake_grad_outputs = torch.ones(
        d_interpolates.shape, device=device, dtype=real_samples.dtype
    )

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake_grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    # Compute Euclidean norm of gradients across feature dimensions
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)

    # Penalty for deviating from 1-Lipschitz (norm == 1)
    gradient_penalty = torch.mean((gradient_norm - 1.0) ** 2)
    return gradient_penalty
