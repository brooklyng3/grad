"""
Unit test verifying exact mathematical equivalence between Kitsune-py KitNET
and our differentiable PyTorchKitNET implementation, including autograd gradient flow.
"""

import sys
import numpy as np
import torch

sys.path.insert(0, "Kitsune-py")
from KitNET.KitNET import KitNET  # noqa: E402
from src.defender.torch_kitnet import PyTorchKitNET  # noqa: E402


def test_pytorch_kitnet_numerical_parity():
    # 1. Instantiate and train a small KitNET model in NumPy
    n_features = 10
    max_ae = 3
    fm_grace = 5
    ad_grace = 10
    kitnet_np = KitNET(n_features, max_ae, fm_grace, ad_grace)

    np.random.seed(42)
    for _ in range(25):
        sample = np.random.rand(n_features)
        kitnet_np.process(sample)

    # 2. Convert to PyTorchKitNET
    device = torch.device("cpu")
    kitnet_pt = PyTorchKitNET.from_kitnet_model(kitnet_np, device=device)
    kitnet_pt.eval()

    # 3. Test on 10 random samples and verify parity within 1e-4 RMSE
    for _ in range(10):
        test_x_np = np.random.rand(n_features)
        rmse_np = kitnet_np.execute(test_x_np)

        test_x_pt = torch.tensor(
            test_x_np, dtype=torch.float32, device=device
        ).unsqueeze(0)
        with torch.no_grad():
            rmse_pt = kitnet_pt(test_x_pt).item()

        assert np.isclose(rmse_np, rmse_pt, atol=1e-4)


def test_pytorch_kitnet_autograd_gradients():
    # Verify backward() produces non-zero gradients on input features
    n_features = 10
    kitnet_np = KitNET(n_features, 3, 5, 10)
    for _ in range(25):
        kitnet_np.process(np.random.rand(n_features))

    device = torch.device("cpu")
    kitnet_pt = PyTorchKitNET.from_kitnet_model(kitnet_np, device=device)

    # Input tensor with requires_grad=True (similar to FGSM)
    x = torch.tensor(
        np.random.rand(2, n_features),
        dtype=torch.float32,
        requires_grad=True,
    )
    rmse = kitnet_pt(x)
    loss = rmse.sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == (2, n_features)
    assert not torch.all(x.grad == 0.0)
