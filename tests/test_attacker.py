"""
Unit tests for Phase 3 Attacker Engine:
  - FGSM attack on PyTorchKitNET
  - DeepFool attack on PyTorchKitNET
  - PolymorphicGenerator forward and mutate_raw passes in normalized space
  - WGANDiscriminator forward and 1-Lipschitz Gradient Penalty calculation
"""

import numpy as np
import pytest
import torch

from src.defender.torch_kitnet import PyTorchKitNET
from src.attacker.perturbations import (
    deepfool_attack,
    envelope_projection,
    fgsm_attack,
    wgan_best_of_k,
    wgan_refine,
)
from src.attacker.generator import PolymorphicGenerator
from src.attacker.discriminator import (
    WGANDiscriminator,
    compute_gradient_penalty,
)


@pytest.fixture(scope="module")
def defender_setup():
    model_path = "data/processed/kitsune_mirai_model.pkl"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PyTorchKitNET.from_file(model_path, device=device)
    model.eval()

    norm_min, norm_max, scales = model.get_global_feature_bounds(as_torch=True)
    with open("data/processed/threshold_config.yaml", "r") as f:
        import yaml

        theta = float(yaml.safe_load(f)["threshold"])

    # Load detected malicious samples for test fixtures
    features = np.load("data/processed/mirai_features.npy", mmap_mode="r")
    labels = []
    with open("data/raw/mirai_labels.csv", "r") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                labels.append(int(line_str))
    labels = np.array(labels, dtype=np.int32)
    attack_indices = np.where(labels == 1)[0]

    # Find samples with RMSE > theta
    sample_slice = features[attack_indices[:500]]
    candidates = torch.tensor(sample_slice, dtype=torch.float32, device=device)
    with torch.no_grad():
        scores = model(candidates)
    detected_mask = scores > theta
    x_detected = candidates[detected_mask][:32]

    return {
        "model": model,
        "device": device,
        "norm_min": norm_min,
        "norm_max": norm_max,
        "scales": scales,
        "theta": theta,
        "x_detected": x_detected,
    }


def test_fgsm_attack_reduces_rmse(defender_setup):
    model = defender_setup["model"]
    x = defender_setup["x_detected"]
    scales = defender_setup["scales"]

    with torch.no_grad():
        initial_rmse = model(x)

    # Perform FGSM with budget eps=0.25
    x_adv, adv_rmse = fgsm_attack(
        model=model,
        x=x,
        eps=0.25,
        scales=scales,
        enforce_domain=True,
    )

    # Assert shape and that FGSM strictly reduces average reconstruction error
    assert x_adv.shape == x.shape
    assert adv_rmse.mean().item() < initial_rmse.mean().item()


def test_deepfool_attack_step(defender_setup):
    model = defender_setup["model"]
    x = defender_setup["x_detected"][:10]
    scales = defender_setup["scales"]
    theta = defender_setup["theta"]

    with torch.no_grad():
        initial_rmse = model(x)

    x_adv, adv_rmse, iters = deepfool_attack(
        model=model,
        x=x,
        theta=theta,
        scales=scales,
        max_iter=15,
        step_size=0.2,
        max_eps=2.0,
    )

    assert x_adv.shape == x.shape
    assert adv_rmse.mean().item() < initial_rmse.mean().item()


def test_polymorphic_generator(defender_setup):
    device = defender_setup["device"]
    x_raw = defender_setup["x_detected"]
    norm_min = defender_setup["norm_min"]
    scales = defender_setup["scales"]

    gen = PolymorphicGenerator(feature_dim=100, latent_dim=32, max_eps=0.5).to(device)
    gen.eval()

    # Test raw feature mutation wrapper
    with torch.no_grad():
        x_adv_raw = gen.mutate_raw(x_raw, norm_min, scales, eps=0.1)

    assert x_adv_raw.shape == x_raw.shape
    # Check that perturbation in normalized space satisfies eps bound
    norm_diff = torch.abs((x_adv_raw - x_raw) / scales)
    assert torch.max(norm_diff).item() <= 0.1 + 1e-5


def test_envelope_projection_evades_detected_samples(defender_setup):
    model = defender_setup["model"]
    x = defender_setup["x_detected"]
    theta = defender_setup["theta"]

    x_adv = envelope_projection(
        x, defender_setup["norm_min"], defender_setup["norm_max"]
    )
    with torch.no_grad():
        rmse = model(x_adv)

    assert x_adv.shape == x.shape
    assert float((rmse <= theta).float().mean()) >= 0.9


def test_wgan_best_of_k_never_worse_than_single_draw(defender_setup):
    device = defender_setup["device"]
    x = defender_setup["x_detected"][:16]
    model = defender_setup["model"]
    theta = defender_setup["theta"]
    generator = PolymorphicGenerator(
        feature_dim=100, latent_dim=32, max_eps=2.0, mode="additive"
    ).to(device)
    generator.eval()

    # Use the same initial RNG state so the first best-of-k draw is identical
    # to the single draw; the invariant is that best-of-k can only improve it.
    torch.manual_seed(123)
    with torch.no_grad():
        single = generator.mutate_raw(
            x,
            defender_setup["norm_min"],
            defender_setup["scales"],
            eps=2.0,
            mode="additive",
        )
        single_rmse = model(single)

    torch.manual_seed(123)
    _, best_rmse, attempts = wgan_best_of_k(
        generator,
        model,
        x,
        theta,
        defender_setup["norm_min"],
        defender_setup["scales"],
        eps=2.0,
        mode="additive",
        max_retries=3,
    )

    assert torch.all(best_rmse <= single_rmse + 1e-6)
    assert int(attempts.min()) >= 1
    assert int(attempts.max()) <= 3


def test_wgan_refine_never_worse_than_input(defender_setup):
    model = defender_setup["model"]
    x = defender_setup["x_detected"][:16]
    generator = PolymorphicGenerator(
        feature_dim=100, latent_dim=32, max_eps=2.0, mode="additive"
    ).to(defender_setup["device"])
    generator.eval()
    with torch.no_grad():
        initial = model(x)

    _, best_rmse, attempts = wgan_refine(
        generator,
        model,
        x,
        defender_setup["theta"],
        defender_setup["norm_min"],
        defender_setup["scales"],
        eps=2.0,
        steps=3,
    )
    assert torch.all(best_rmse <= initial + 1e-6)
    assert int(attempts.max()) <= 3


def test_wgan_discriminator_and_gradient_penalty(defender_setup):
    device = defender_setup["device"]
    critic = WGANDiscriminator(feature_dim=100).to(device)

    batch_size = 16
    real_norm = torch.rand((batch_size, 100), device=device)
    fake_norm = torch.rand((batch_size, 100), device=device)

    # Test forward pass
    score = critic(real_norm)
    assert score.shape == (batch_size, 1)

    # Test Gradient Penalty computation
    gp = compute_gradient_penalty(critic, real_norm, fake_norm, device=device)
    assert gp.dim() == 0  # Scalar loss
    assert gp.item() >= 0.0
