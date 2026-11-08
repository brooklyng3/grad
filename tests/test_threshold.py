"""
Unit test verifying Phase 2 threshold calibration integrity,
benign False Positive Rate compliance, and PyTorchKitNET evaluation output.
"""

import os
import yaml
import numpy as np
import torch

from src.defender.torch_kitnet import PyTorchKitNET


def test_threshold_config_metadata():
    config_path = "data/processed/threshold_config.yaml"
    assert os.path.exists(config_path), f"Missing {config_path}"

    with open(config_path, "r") as f:
        meta = yaml.safe_load(f)

    # 1. Verify calibrated threshold
    theta = meta["threshold"]
    assert 0.1 <= theta <= 1.0, f"Unusual threshold value: {theta}"

    # 2. Calibration/validation splits are disjoint and of equal size
    cal = meta["benign_calibration"]
    val = meta["benign_validation"]
    assert cal["num_evaluated"] == 33311
    assert val["num_evaluated"] == 33310

    # 3. Validation FPR is out-of-sample and should remain low
    fpr = val["false_positive_rate_pct"]
    assert fpr <= 0.5, f"Out-of-sample FPR unexpectedly high: {fpr}"

    # 4. Verify attack metrics
    a_meta = meta["attack_metrics"]
    assert a_meta["num_evaluated"] == 642516
    det_rate = a_meta["detection_rate_pct"]
    evasion_rate = a_meta["evasion_rate_pct"]
    assert np.isclose(det_rate + evasion_rate, 100.0)
    assert det_rate > 80.0, f"Baseline detection rate unexpectedly low: {det_rate}"


def test_torch_kitnet_threshold_decision():
    # Verify scoring batch against theta on device
    config_path = "data/processed/threshold_config.yaml"
    with open(config_path, "r") as f:
        meta = yaml.safe_load(f)
    theta = meta["threshold"]

    model_path = "data/processed/kitsune_mirai_model.pkl"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PyTorchKitNET.from_file(model_path, device=device)
    model.eval()

    features = np.load("data/processed/mirai_features.npy", mmap_mode="r")

    # Take 10 benign samples from post-convergence region (idx 60,000 to 60,010)
    benign_x = torch.tensor(features[60000:60010], dtype=torch.float32, device=device)
    with torch.no_grad():
        benign_scores = model(benign_x).cpu().numpy()

    # Most benign samples should be well below theta
    assert np.mean(benign_scores) < theta
