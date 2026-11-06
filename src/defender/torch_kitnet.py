"""
PyTorch Differentiable Implementation of KitNET (Kitsune Anomaly Detector).

Provides full PyTorch autograd (.backward()) compatibility and CUDA GPU batching
for gradient-based adversarial attack generators (FGSM, DeepFool, WGAN-GP) while
loading directly from the trained NumPy KitNET model weights.
"""

from typing import List, Any, Union, Optional, Tuple
import pickle
import numpy as np
import torch
import torch.nn as nn


class DAE(nn.Module):
    """PyTorch implementation of a single Denoising Autoencoder in KitNET."""

    def __init__(
        self,
        n_visible: int,
        n_hidden: int,
        norm_min: np.ndarray,
        norm_max: np.ndarray,
        W: np.ndarray,
        hbias: np.ndarray,
        vbias: np.ndarray,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        # Normalization parameters (0-1 min-max scaling)
        # Protect against division by zero on zero-variance / constant features
        norm_range = np.maximum(norm_max - norm_min, 1e-12)
        self.register_buffer(
            "norm_min",
            torch.tensor(norm_min, dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "norm_range",
            torch.tensor(norm_range, dtype=torch.float32, device=device),
        )

        # Autoencoder weights: W is [n_visible, n_hidden]
        # In KitNET: encode is sigmoid(x * W + hbias)
        #            decode is sigmoid(h * W.T + vbias)
        self.W = nn.Parameter(torch.tensor(W, dtype=torch.float32, device=device))
        self.hbias = nn.Parameter(
            torch.tensor(hbias, dtype=torch.float32, device=device)
        )
        self.vbias = nn.Parameter(
            torch.tensor(vbias, dtype=torch.float32, device=device)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass computing normalized reconstruction and RMSE.

        Args:
            x: Input tensor [batch_size, n_visible].

        Returns:
            Tuple of (rmse_loss, reconstructed_x).
        """
        # 0-1 Normalization
        x_norm = (x - self.norm_min) / self.norm_range

        # Encode: sigmoid(x * W + hbias)
        hidden = torch.sigmoid(torch.matmul(x_norm, self.W) + self.hbias)

        # Decode: sigmoid(hidden * W.T + vbias)
        reconstructed = torch.sigmoid(torch.matmul(hidden, self.W.t()) + self.vbias)

        # RMSE per sample in the batch
        diff_sq = (x_norm - reconstructed) ** 2
        rmse = torch.sqrt(torch.mean(diff_sq, dim=-1) + 1e-12)
        return rmse, reconstructed


class PyTorchKitNET(nn.Module):
    """PyTorch wrapper for the complete KitNET Ensemble & Master Autoencoder.

    Enables:
      1. High-throughput CUDA mini-batch evaluation (batch_size 64 to 256).
      2. Direct end-to-end gradient backpropagation for FGSM, DeepFool, and WGAN.
      3. Exact equivalence to original Kitsune-py NumPy inference.
    """

    def __init__(
        self,
        v: List[List[int]],
        ensemble_daes: List[DAE],
        output_dae: DAE,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.v = v  # Feature clustering mapping indices
        self.ensemble = nn.ModuleList(ensemble_daes)
        self.output_layer = output_dae
        self.device = device
        self.to(device)

    @classmethod
    def from_kitnet_model(
        cls,
        kitnet_obj: Any,
        device: Optional[Union[str, torch.device]] = None,
    ) -> "PyTorchKitNET":
        """Instantiates PyTorchKitNET from a trained Kitsune-py KitNET object.

        Args:
            kitnet_obj: Trained KitNET instance (e.g. unpickled from model file).
            device: Target torch device ('cuda' or 'cpu').

        Returns:
            Fully initialized PyTorchKitNET model on the specified device.
        """
        if device is None:
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            dev = torch.device(device)
        else:
            dev = device

        v = kitnet_obj.v
        ensemble_daes: List[DAE] = []

        for dae_np in kitnet_obj.ensembleLayer:
            dae_pt = DAE(
                n_visible=dae_np.params.n_visible,
                n_hidden=dae_np.params.n_hidden,
                norm_min=dae_np.norm_min,
                norm_max=dae_np.norm_max,
                W=dae_np.W,
                hbias=dae_np.hbias,
                vbias=dae_np.vbias,
                device=dev,
            )
            ensemble_daes.append(dae_pt)

        output_dae_np = kitnet_obj.outputLayer
        output_dae_pt = DAE(
            n_visible=output_dae_np.params.n_visible,
            n_hidden=output_dae_np.params.n_hidden,
            norm_min=output_dae_np.norm_min,
            norm_max=output_dae_np.norm_max,
            W=output_dae_np.W,
            hbias=output_dae_np.hbias,
            vbias=output_dae_np.vbias,
            device=dev,
        )

        return cls(
            v=v,
            ensemble_daes=ensemble_daes,
            output_dae=output_dae_pt,
            device=dev,
        )

    @classmethod
    def from_file(
        cls,
        model_path: str,
        device: Optional[Union[str, torch.device]] = None,
    ) -> "PyTorchKitNET":
        """Loads a trained KitNET from pickled file and converts to PyTorch."""
        with open(model_path, "rb") as f:
            kitnet_obj = pickle.load(f)
        return cls.from_kitnet_model(kitnet_obj, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes KitNET anomaly reconstruction error across a batch.

        Args:
            x: Batch of continuous feature vectors [batch_size, feature_dim].

        Returns:
            RMSE reconstruction error tensor [batch_size].
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        batch_size = x.size(0)
        num_ensembles = len(self.ensemble)

        # Allocate ensemble RMSE output layer inputs
        s_l1 = torch.zeros(
            (batch_size, num_ensembles), dtype=torch.float32, device=x.device
        )

        # 1. Ensemble Layer execution
        for a in range(num_ensembles):
            cluster_indices = self.v[a]
            xi = x[:, cluster_indices]
            dae_rmse, _ = self.ensemble[a](xi)
            s_l1[:, a] = dae_rmse

        # 2. Output Layer execution
        final_rmse, _ = self.output_layer(s_l1)
        return final_rmse

    def get_global_feature_bounds(self, as_torch: bool = True) -> Tuple[
        Union[torch.Tensor, np.ndarray],
        Union[torch.Tensor, np.ndarray],
        Union[torch.Tensor, np.ndarray],
    ]:
        """Reconstructs global 100-dimensional norm_min, norm_max, and scales.

        Reassembles the individual micro-autoencoder normalization bounds back into
        full 100-dimensional vectors mapped by self.v. Used by Phase 3 perturbation
        engines (FGSM, DeepFool, WGAN-GP) to enforce feature ranges and scale budgets.

        Args:
            as_torch: If True, returns torch.Tensors on self.device; else np.ndarrays.

        Returns:
            Tuple of (norm_min, norm_max, scales) where scales = norm_max - norm_min.
        """
        # Determine total feature dimension from self.v
        n_features = sum(len(sub) for sub in self.v)
        global_min = torch.zeros(n_features, dtype=torch.float32, device=self.device)
        global_max = torch.zeros(n_features, dtype=torch.float32, device=self.device)

        for a_idx, dae in enumerate(self.ensemble):
            cluster_indices = torch.tensor(
                self.v[a_idx], dtype=torch.long, device=self.device
            )
            # Reconstruct dae norm_max from norm_min + norm_range
            dae_min = dae.norm_min
            dae_max = dae.norm_min + dae.norm_range
            global_min.index_copy_(0, cluster_indices, dae_min)
            global_max.index_copy_(0, cluster_indices, dae_max)

        scales = torch.clamp(global_max - global_min, min=1e-12)

        if not as_torch:
            return (
                global_min.cpu().numpy(),
                global_max.cpu().numpy(),
                scales.cpu().numpy(),
            )
        return global_min, global_max, scales
