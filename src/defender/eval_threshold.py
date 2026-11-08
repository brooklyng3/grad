"""
Phase 2: Defender Evaluation & Alert Threshold Calibration.

Evaluates the trained KitNET model (PyTorchKitNET) on the full Mirai dataset:
  1. Splits post-convergence benign packets (i >= 55,000) by parity into disjoint
     calibration and validation halves.
  2. Calibrates the anomaly alert threshold theta on the calibration half only at
     the 99.9th percentile.
  3. Measures the false positive rate out-of-sample on the validation half.
  4. Evaluates detection metrics across the 642,516 Mirai attack packets:
         - Detection Rate (Recall / True Positive Rate)
         - Baseline Evasion Rate (attacks <= theta)
  5. Saves threshold calibration metadata to data/processed/threshold_config.yaml.
"""

import os
import sys
import time
from typing import Dict, Any, Tuple
import numpy as np
import torch
import yaml

# Ensure workspace and Kitsune-py are importable
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("Kitsune-py"))
from src.defender.torch_kitnet import PyTorchKitNET  # noqa: E402
from src.utils.data_splits import benign_indices_for_split  # noqa: E402


def load_dataset(
    features_path: str = "data/processed/mirai_features.npy",
    labels_path: str = "data/raw/mirai_labels.csv",
) -> Tuple[np.ndarray, np.ndarray]:
    """Loads memory-mapped features and ground-truth label array."""
    print(f"Loading features from {features_path} (mmap mode)...")
    features = np.load(features_path, mmap_mode="r")

    print(f"Loading labels from {labels_path}...")
    labels = []
    with open(labels_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                labels.append(int(line))
    labels = np.array(labels, dtype=np.int32)

    assert len(features) == len(
        labels
    ), f"Length mismatch: {len(features)} features vs {len(labels)} labels"
    return features, labels


def evaluate_and_calibrate(
    config_path: str = "configs/kitsune_config.yaml",
    batch_size: int = 512,
) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    d_cfg = config["data"]
    k_cfg = config["kitsune"]

    model_path = d_cfg["model_weights"]
    features_path = d_cfg["processed_features"]
    labels_path = d_cfg["raw_labels"]
    percentile_target = k_cfg.get("threshold_percentile", 99.9)
    grace_period = k_cfg.get("fm_grace", 5000) + k_cfg.get("ad_grace", 50000)

    print("=" * 70)
    print("PHASE 2: DEFENDER EVALUATION & THRESHOLD CALIBRATION")
    print(f"Model Path:         {model_path}")
    print(f"Features Path:      {features_path}")
    print(f"Grace Period Cut:   {grace_period} packets")
    print(f"Target Percentile:  {percentile_target}%")
    print("=" * 70)

    # 1. Load Model onto CUDA GPU (or CPU fallback)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Instantiating PyTorchKitNET on device: {device}")
    model = PyTorchKitNET.from_file(model_path, device=device)
    model.eval()

    # 2. Load Features & Labels
    features, labels = load_dataset(features_path, labels_path)
    total_packets = len(labels)
    first_attack_idx = int(np.where(labels == 1)[0][0])
    print(f"Total dataset packets:       {total_packets:,}")
    print(f"First attack packet index:   {first_attack_idx:,}")

    # 3. Disjoint benign calibration and validation splits
    # Calibration = even relative index, validation = odd relative index.
    calibration_indices = benign_indices_for_split(labels, "calibration")
    validation_indices = benign_indices_for_split(labels, "validation")
    num_cal_benign = len(calibration_indices)
    num_val_benign = len(validation_indices)

    def score_indices(indices: np.ndarray) -> np.ndarray:
        out = []
        with torch.no_grad():
            for start_idx in range(0, len(indices), batch_size):
                batch_sub_idx = indices[start_idx : start_idx + batch_size]
                batch_x = torch.tensor(
                    features[batch_sub_idx], dtype=torch.float32, device=device
                )
                out.append(model(batch_x).cpu().numpy())
        return np.concatenate(out)

    print(
        f"\nScoring {num_cal_benign:,} calibration and "
        f"{num_val_benign:,} validation benign packets on GPU..."
    )
    start_time = time.time()
    benign_rmses = score_indices(calibration_indices)
    cal_eval_time = time.time() - start_time
    print(
        f"Benign calibration scoring complete in {cal_eval_time:.2f}s "
        f"({num_cal_benign / cal_eval_time:.1f} pkts/s)"
    )

    # 4. Calibrate Alert Threshold Theta on the calibration split only
    theta = float(np.percentile(benign_rmses, percentile_target))
    mean_benign_rmse = float(np.mean(benign_rmses))
    std_benign_rmse = float(np.std(benign_rmses))
    max_benign_rmse = float(np.max(benign_rmses))
    min_benign_rmse = float(np.min(benign_rmses))

    # 5. Measure FPR out-of-sample on the disjoint validation split
    validation_rmses = score_indices(validation_indices)
    fp_count = int(np.sum(validation_rmses > theta))
    fpr = (fp_count / num_val_benign) * 100.0
    val_mean = float(np.mean(validation_rmses))
    val_std = float(np.std(validation_rmses))

    print("-" * 70)
    print("THRESHOLD CALIBRATION RESULTS:")
    print(f"  Calculated Theta (θ):       {theta:.6f}")
    print(
        f"  Calibration RMSE (mean ± std): "
        f"{mean_benign_rmse:.6f} ± {std_benign_rmse:.6f}"
    )
    print(
        f"  Calibration RMSE (min - max):  "
        f"{min_benign_rmse:.6f} - {max_benign_rmse:.6f}"
    )
    print(
        f"  Validation FPR (out-of-sample): {fpr:.3f}% "
        f"({fp_count}/{num_val_benign}), mean RMSE {val_mean:.6f}"
    )
    print("-" * 70)

    # 6. Batch Evaluation Across Malicious Traffic (Attack Packets)
    malicious_indices = np.where(labels == 1)[0]
    num_attacks = len(malicious_indices)
    print(f"\nEvaluating {num_attacks:,} malicious Mirai packets on GPU...")

    malicious_rmses = []
    start_attack_eval = time.time()

    with torch.no_grad():
        for start_idx in range(0, num_attacks, batch_size):
            batch_sub_idx = malicious_indices[start_idx : start_idx + batch_size]
            batch_x = torch.tensor(
                features[batch_sub_idx], dtype=torch.float32, device=device
            )
            batch_rmse = model(batch_x)
            malicious_rmses.append(batch_rmse.cpu().numpy())

    malicious_rmses = np.concatenate(malicious_rmses)
    attack_eval_time = time.time() - start_attack_eval
    print(
        f"Attack evaluation complete in {attack_eval_time:.2f}s "
        f"({num_attacks / attack_eval_time:.1f} pkts/s)"
    )

    # 7. Baseline Defensive & Evasion Metrics
    detected_count = int(np.sum(malicious_rmses > theta))
    evaded_count = int(np.sum(malicious_rmses <= theta))
    detection_rate = (detected_count / num_attacks) * 100.0
    evasion_rate = (evaded_count / num_attacks) * 100.0
    mean_attack_rmse = float(np.mean(malicious_rmses))
    std_attack_rmse = float(np.std(malicious_rmses))
    max_attack_rmse = float(np.max(malicious_rmses))
    min_attack_rmse = float(np.min(malicious_rmses))

    print("-" * 70)
    print("BASELINE DEFENSIVE METRICS (CLEAN MIRAI ATTACKS):")
    print(f"  Total Attacks Evaluated:    {num_attacks:,}")
    print(f"  Attacks Detected:           {detected_count:,} ({detection_rate:.2f}%)")
    print(f"  Attacks Evaded (Baseline):  {evaded_count:,} ({evasion_rate:.2f}%)")
    print(
        f"  Attack RMSE (mean ± std):   {mean_attack_rmse:.6f} ± {std_attack_rmse:.6f}"
    )
    print(
        f"  Attack RMSE (min - max):    {min_attack_rmse:.6f} - {max_attack_rmse:.6f}"
    )
    print("=" * 70)

    # 8. Persist Threshold Configuration & Evaluation Metrics
    output_meta_path = "data/processed/threshold_config.yaml"
    results = {
        "threshold": theta,
        "percentile": percentile_target,
        "grace_period_excluded": grace_period,
        "benign_calibration": {
            "num_evaluated": num_cal_benign,
            "mean_rmse": mean_benign_rmse,
            "std_rmse": std_benign_rmse,
            "min_rmse": min_benign_rmse,
            "max_rmse": max_benign_rmse,
        },
        "benign_validation": {
            "num_evaluated": num_val_benign,
            "mean_rmse": val_mean,
            "std_rmse": val_std,
            "false_positive_count": fp_count,
            "false_positive_rate_pct": fpr,
        },
        "attack_metrics": {
            "num_evaluated": num_attacks,
            "detected_count": detected_count,
            "evaded_count": evaded_count,
            "detection_rate_pct": detection_rate,
            "evasion_rate_pct": evasion_rate,
            "mean_rmse": mean_attack_rmse,
            "std_rmse": std_attack_rmse,
            "min_rmse": min_attack_rmse,
            "max_rmse": max_attack_rmse,
        },
    }

    with open(output_meta_path, "w") as f:
        yaml.safe_dump(results, f, sort_keys=False)
    print(f"Calibration metadata saved to {output_meta_path}")

    # Free CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


if __name__ == "__main__":
    evaluate_and_calibrate()
