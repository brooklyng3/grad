"""
Feature Extraction and KitNET Training Pipeline (Phase 1.5).

Streams packets from data/raw/mirai.pcap.tsv through AfterImage (FE.get_next_vector()),
saves the 100-dimensional continuous statistical feature vectors directly to disk via
numpy.memmap (data/processed/mirai_features.npy), and trains KitNET online.

Only the neural network ensemble (K.AnomDetector) is persisted to
data/processed/kitsune_mirai_model.pkl, avoiding unpicklable parser state.
"""

import os
import sys
import time
import pickle
from typing import Optional
import numpy as np
import yaml

# Add Kitsune-py to Python module path
sys.path.insert(0, os.path.abspath("Kitsune-py"))
from FeatureExtractor import FE  # noqa: E402
from KitNET.KitNET import KitNET  # noqa: E402


def run_extraction_and_training(
    config_path: str = "configs/kitsune_config.yaml",
    packet_limit: Optional[int] = None,
) -> None:
    # 1. Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    k_cfg = config["kitsune"]
    d_cfg = config["data"]

    tsv_path = d_cfg["raw_tsv"]
    npy_path = d_cfg["processed_features"]
    model_path = d_cfg["model_weights"]

    feature_dim = k_cfg["feature_dim"]
    max_ae = k_cfg["max_ae"]
    fm_grace = k_cfg["fm_grace"]
    ad_grace = k_cfg["ad_grace"]
    lr = k_cfg.get("learning_rate", 0.1)
    hidden_ratio = k_cfg.get("hidden_ratio", 0.75)

    limit = packet_limit or d_cfg.get("packet_limit") or np.inf

    print("=" * 70)
    print("PHASE 1.5: FEATURE EXTRACTION & DEFENDER TRAINING RUN")
    print(f"TSV Source:       {tsv_path}")
    print(f"Feature Target:   {npy_path}")
    print(f"Model Target:     {model_path}")
    print(f"Hyperparameters:  maxAE={max_ae}, FMgrace={fm_grace}, ADgrace={ad_grace}")
    print("=" * 70)

    # 2. Initialize Feature Extractor
    start_time = time.time()
    fe = FE(tsv_path, limit=limit)
    total_packets = fe.limit
    print(f"Total packets to process: {total_packets}")

    # 3. Create or open memory-mapped array for 100D float32 features
    os.makedirs(os.path.dirname(npy_path), exist_ok=True)
    features_mmap = np.lib.format.open_memmap(
        npy_path,
        mode="w+",
        dtype=np.float32,
        shape=(total_packets, feature_dim),
    )

    # 4. Initialize KitNET Anomaly Detector
    kitnet = KitNET(
        n=feature_dim,
        max_autoencoder_size=max_ae,
        FM_grace_period=fm_grace,
        AD_grace_period=ad_grace,
        learning_rate=lr,
        hidden_ratio=hidden_ratio,
    )

    # 5. Streaming loop
    print("Beginning extraction and online training loop...")
    count = 0
    checkpoint_step = 25000
    loop_start = time.time()

    while count < total_packets:
        v = fe.get_next_vector()
        if len(v) == 0:
            print(f"Reached EOF at packet index {count}")
            break

        # Save feature vector to memory-mapped array
        features_mmap[count] = v

        # Online update / training KitNET
        kitnet.process(v)

        count += 1
        if count % checkpoint_step == 0:
            elapsed = time.time() - loop_start
            rate = count / (elapsed + 1e-6)
            eta = (total_packets - count) / (rate + 1e-6)
            print(
                f"[{count:,}/{total_packets:,}] "
                f"Speed: {rate:.1f} pkts/s | "
                f"Elapsed: {elapsed:.1f}s | "
                f"ETA: {eta:.1f}s"
            )
            # Flush memmap periodically to ensure disk persistence
            features_mmap.flush()

    # Flush final memmap
    features_mmap.flush()
    total_time = time.time() - start_time
    print("-" * 70)
    print(f"Processing complete! Processed {count:,} feature vectors.")
    rate = count / (total_time + 1e-6)
    print(f"Total time: {total_time:.2f}s ({rate:.1f} pkts/s)")

    # 6. Save only the KitNET neural network ensemble
    print(f"Serializing trained KitNET model to {model_path}...")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(kitnet, f)

    file_size_mb = os.path.getsize(npy_path) / (1024 * 1024)
    model_size_kb = os.path.getsize(model_path) / 1024
    print(f"Saved {npy_path} ({file_size_mb:.2f} MB)")
    print(f"Saved {model_path} ({model_size_kb:.2f} KB)")
    print("=" * 70)


if __name__ == "__main__":
    run_extraction_and_training()
