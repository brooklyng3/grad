# Phase 1.5 Comprehensive Technical Log: Feature Extraction, Memmapped Storage & KitNET Training

This document provides an exhaustive, granular engineering record of the streaming feature extraction pass, memory-mapped array serialization, KitNET state transitions, CorClust ensemble clustering, and PyTorch CUDA validation executed during Phase 1.5.

---

## 1. System Environment & Execution Baseline
- **Execution Script:** `scripts/extract_and_train.py`
- **Config Profile:** `configs/kitsune_config.yaml`
- **Source Data:** `data/raw/mirai.pcap.tsv` (764,138 lines; 1 header + 764,137 data lines)
- **Feature Target:** `data/processed/mirai_features.npy`
- **Model Target:** `data/processed/kitsune_mirai_model.pkl`
- **Host Resource Limits:** Intel Core i5-9400F, 16 GB DDR4, NVIDIA GTX 1660 SUPER.

---

## 2. Pipeline Architecture & Implementation

### 2.1. Architectural Distinction Enforced
To avoid passing raw text fields (e.g. `eth.src`, `ip.src`, MAC address hex strings) directly into numerical models, `scripts/extract_and_train.py` establishes a clean, decoupled two-tier architecture:
1. `FeatureExtractor.py` parses each packet header line from `mirai.pcap.tsv` and feeds packet metadata into `AfterImage` (`incStatDB`). The pure-Python module is used; the compiled `AfterImage_extrapolate` extension is an opt-in alternative via `KITSUNE_USE_CYTHON=1`.
2. `AfterImage` computes 100 continuous decaying statistical metrics across 5 damped windows ($\lambda \in \{5, 3, 1, 0.1, 0.01\}$).
3. The resulting numeric float vector $\mathbf{v} \in \mathbb{R}^{100}$ is written directly to an open memory-mapped disk buffer (`np.lib.format.open_memmap`) and simultaneously passed to `kitnet.process(v)`.

### 2.2. Script Execution Command
```bash
source .venv/bin/activate
python3 scripts/extract_and_train.py
```

---

## 3. Training Dynamics & KitNET State Transitions

The unsupervised KitNET learner progressed through three distinct operational phases:

### 3.1. Phase A: Feature Mapping (`FMgrace = 5000`)
- **Packets 1 to 5,000:** Anomaly detector disabled (`off-mode`).
- `CorClust` tracked incremental Pearson correlation coefficients across all 100 features.
- At packet 5,000, hierarchical clustering grouped the 100 features into **16 micro-autoencoders** based on maximum ensemble size `maxAE = 10`.
- Console log:
  ```text
  The Feature-Mapper found a mapping: 100 features to 16 autoencoders.
  Feature-Mapper: execute-mode, Anomaly-Detector: train-mode
  ```

### 3.2. Phase B: Anomaly Detector Training (`ADgrace = 50000`)
- **Packets 5,001 to 55,000:** The 16 micro-autoencoders and the master output autoencoder were trained online via stochastic gradient descent ($\text{lr} = 0.1$, $\text{hidden\_ratio} = 0.75$).
- Normalization parameters (`norm_min`, `norm_max`) were continuously calibrated over observed benign network flow dynamics.
- At packet 55,000, training grace period completed:
  ```text
  Feature-Mapper: execute-mode, Anomaly-Detector: execute-mode
  ```

### 3.3. Phase C: Pure Inference Mode
- **Packets 55,001 to 764,137:** KitNET operated in execute mode, calculating reconstruction RMSE without updating weights.

---

## 4. Extraction Telemetry & Progress Milestones

| Packet Milestone | Processing Speed | Elapsed Time | Estimated Time Remaining (ETA) |
| :--- | :--- | :--- | :--- |
| **25,000** | 974.9 pkts/s | 25.6s | 758.1s |
| **50,000** | 740.1 pkts/s | 67.6s | 964.9s |
| **100,000** | 592.3 pkts/s | 168.8s | 1121.2s |
| **200,000** | 494.2 pkts/s | 404.7s | 1141.5s |
| **300,000** | 467.1 pkts/s | 642.2s | 993.6s |
| **400,000** | 451.4 pkts/s | 886.1s | 806.7s |
| **500,000** | 441.0 pkts/s | 1133.7s | 598.9s |
| **600,000** | 432.7 pkts/s | 1386.5s | 379.3s |
| **700,000** | 425.8 pkts/s | 1643.9s | 150.6s |
| **764,137 (EOF)** | **422.5 pkts/s** | **1,808.47s (30.14 min)** | **0.0s** |

---

## 5. Persistence & Verification

### 5.1. Model Serialization (Detector State Only)
The trained `KitNET` object (the same object stored as `K.AnomDetector` in the
`Kitsune` wrapper) was serialized. It includes the feature map and feature-mapper
state but not the outer wrapper or its file handles:
- **Output:** `data/processed/kitsune_mirai_model.pkl` (187.36 KB)
- Contained variables:
  - Feature mapping index list `kitnet.v` (16 sub-lists mapping 100 features)
  - 16 trained `dA` micro-autoencoders (`ensembleLayer`)
  - 1 trained `dA` master autoencoder (`outputLayer`)
  - Total trained packet count: `55,001`

### 5.2. Memory-Mapped Feature Matrix (`mirai_features.npy`)
- **Dimensions:** $(764137, 100)$
- **Dtype:** `float32` (IEEE 754 single precision)
- **Disk Footprint:** $291.50\text{ MB}$ ($305,654,800\text{ bytes}$)
- **RAM Efficiency:** Slices of this array can be indexed on-demand via `np.load(..., mmap_mode='r')` with negligible memory overhead.

### 5.3. PyTorch CUDA Integration Test
Verified that `PyTorchKitNET` successfully loads the pickled model weights onto the local NVIDIA GeForce GTX 1660 SUPER GPU and executes batch evaluation:
```python
import numpy as np, torch
from src.defender.torch_kitnet import PyTorchKitNET

features = np.load("data/processed/mirai_features.npy", mmap_mode="r")
pt_kitnet = PyTorchKitNET.from_file("data/processed/kitsune_mirai_model.pkl")
pt_kitnet.eval()

# Batch of 128 samples on CUDA
x_batch = torch.tensor(
    features[100000:100128], dtype=torch.float32, device=pt_kitnet.device
)
with torch.no_grad():
  scores = pt_kitnet(x_batch)

print("Device:", pt_kitnet.device)
print("Score at index 100,000:", scores[0].item())
```

#### Diagnostic Output:
```text
PyTorchKitNET batch execution successful!
Scores shape: torch.Size([128])
First score: 0.009949348866939545
Device: cuda
```
- Parity Check: NumPy `kitnet.execute(features[100000])` yielded `0.009949344235095484`.
- Difference between PyTorch GPU float32 and NumPy CPU float64: $< 5 \times 10^{-9}$ (exact numerical equivalence).

---

## 6. Mathematical Invariants & Phase 2/3 Preparatory Architecture

### 6.1. Grace Period Exclusion for Threshold Calibration ($\theta$)
An empirical audit of `data/raw/mirai_labels.csv` revealed:
- **Total Packets:** 764,137
- **Total Benign Packets:** 121,621 (indices `0` to `121,620`)
- **First Attack Packet Index:** `121,621` (Mirai attack commences)
- **Total Malicious Packets:** 642,516 (indices `121,621` to `764,136`)

**Critical Calibration Rule:**
During the initial 55,000 packets (`FMgrace=5000` + `ADgrace=50000`), the model is actively learning:
- Packets 0 to 4,999 return $\text{RMSE} = 0.0$ (clustering mode).
- Packets 5,000 to 54,999 experience continuous stochastic weight updates where reconstruction error is non-stationary.
Including these 55,000 packets would severely skew the empirical cumulative distribution and artificially suppress the 99.9th percentile threshold $\theta$.
Therefore, Phase 2 threshold calibration is strictly defined over the **post-convergence benign subset**. Phase 6 split that range by parity into disjoint calibration and validation halves:
$$\mathcal{D}_{\text{cal}} = \{ \mathbf{x}_i \mid \text{label}_i = 0 \land (i-55{,}000) \bmod 2 = 0 \} \quad (|\mathcal{D}_{\text{cal}}| = 33,311)$$
$$\mathcal{D}_{\text{val}} = \{ \mathbf{x}_i \mid \text{label}_i = 0 \land (i-55{,}000) \bmod 2 = 1 \} \quad (|\mathcal{D}_{\text{val}}| = 33,310)$$
$$\theta = \text{percentile}(\{ \text{RMSE}(\mathbf{x}) \mid \mathbf{x} \in \mathcal{D}_{\text{cal}} \}, 99.9)$$

### 6.2. Global Normalization Feature Bounds (`get_global_feature_bounds`)
Each micro-autoencoder $a \in \{0, \dots, 15\}$ maintains localized normalization vectors:
$$\mathbf{norm\_min}^{(a)}, \mathbf{norm\_max}^{(a)} \in \mathbb{R}^{d_a} \quad \text{where } \sum_{a=0}^{15} d_a = 100$$
To enable global feature budgeting for Phase 3 attack generators without data leakage, `PyTorchKitNET` provides `get_global_feature_bounds()`:
$$\mathbf{norm\_min}_{\text{global}}[v_a] = \mathbf{norm\_min}^{(a)}$$
$$\mathbf{norm\_max}_{\text{global}}[v_a] = \mathbf{norm\_max}^{(a)}$$
$$\mathbf{s}_{\text{global}} = \max(\mathbf{norm\_max}_{\text{global}} - \mathbf{norm\_min}_{\text{global}}, 10^{-12})$$
Validation confirmed:
- Zero-scale count: $0$ (every feature has positive variance).
- Min scale: $1.5743$.
- Max scale: $977,758.44$.
This allows Phase 3 perturbation bounds ($\epsilon \cdot \mathbf{s}$) to directly match the normalized input space of KitNET's autoencoders.
