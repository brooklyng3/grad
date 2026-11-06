# Phase 2 Comprehensive Technical Log: Defender Evaluation, Threshold Calibration & Baseline Robustness

This document provides an exhaustive, granular engineering record of the mathematical threshold calibration, empirical distribution audits, GPU batch throughput, False Positive Rate verification, and clean attack evasion benchmarking executed during Phase 2.

---

## 1. System Environment & Execution Baseline
- **Execution Script:** `src/defender/eval_threshold.py`
- **Trained Model:** `data/processed/kitsune_mirai_model.pkl` loaded via `PyTorchKitNET`
- **Feature Source:** `data/processed/mirai_features.npy` (764,137 packets, 100 features, `float32`)
- **Label Source:** `data/raw/mirai_labels.csv` (764,137 binary ground-truth labels)
- **Target Percentile:** $99.9\text{th}$ percentile of post-convergence benign traffic
- **Compute Device:** NVIDIA GeForce GTX 1660 SUPER (CUDA)
- **Batch Size:** 512 samples/batch

---

## 2. Dataset Distribution & Grace Period Partitioning

An empirical audit of the ground truth labels established the exact temporal boundaries:
- **Total Instances:** 764,137 packets
- **Total Benign Traffic:** 121,621 packets (indices `0` to `121,620`)
- **First Attack Packet:** Index `121,621` (Mirai Telnet infection and propagation starts)
- **Total Attack Traffic:** 642,516 packets (indices `121,621` to `764,136`)

### 2.1. Grace Period Exclusion Justification
As established in Phase 1.5, the first 55,000 packets consist of:
- `FMgrace = 5,000` packets: KitNET is in clustering mode (`CorClust`), returning $\text{RMSE} = 0.0$.
- `ADgrace = 50,000` packets: The 16 micro-autoencoders are actively updating their weights on incoming traffic via SGD; the loss landscape is dynamic and non-stationary.

Including these 55,000 packets would depress the empirical percentile calculation. Therefore, the calibration set was strictly defined as:
$$\mathcal{D}_{\text{cal}} = \{ \mathbf{x}_i \mid \text{label}_i = 0 \land i \in [55,000, 121,620] \}$$
$$|\mathcal{D}_{\text{cal}}| = 66,621 \text{ packets}$$

---

## 3. GPU Batch Evaluation & Anomaly Threshold Calibration

### 3.1. Benign Calibration Performance
- Evaluated $66,621$ samples in batches of 512 using `with torch.no_grad():` on CUDA.
- **Execution Runtime:** $1.72$ seconds.
- **Evaluation Throughput:** $38,770.7$ packets/second.

### 3.2. Calibrated Value & Distribution Statistics
$$\theta = \text{percentile}(\{ \text{RMSE}(\mathbf{x}) \mid \mathbf{x} \in \mathcal{D}_{\text{cal}} \}, 99.9) = \mathbf{0.453419}$$

- **Mean Benign RMSE ($\mu$):** $0.015085$
- **Standard Deviation ($\sigma$):** $0.044458$
- **Minimum Benign RMSE:** $0.002349$
- **Maximum Benign RMSE:** $6.930492$
- **False Positive Count:** $67$ packets scored $> \theta$ out of $66,621$.
- **Empirical False Positive Rate (FPR):**
  $$\text{FPR} = \frac{67}{66,621} \times 100\% = \mathbf{0.10057\%}$$
  *(Matches the theoretical $0.1\%$ expectation for a $99.9\text{th}$ percentile threshold).*

---

## 4. Baseline Attack Evaluation (Unperturbed Mirai Traffic)

All $642,516$ clean Mirai attack packets ($i \ge 121,621$) were passed through `PyTorchKitNET` on CUDA to benchmark baseline defender effectiveness before adversarial polymorphic mutations.

### 4.1. Attack Evaluation Performance
- Evaluated $642,516$ samples in batches of 512 on CUDA.
- **Execution Runtime:** $14.02$ seconds.
- **Evaluation Throughput:** $45,814.1$ packets/second.

### 4.2. Baseline Defensive Metrics
- **Attacks Detected ($\text{RMSE} > \theta$):** $566,393$ packets
- **Baseline Detection Rate (TPR / Recall):** **$88.1524\%$**
- **Attacks Evaded ($\text{RMSE} \le \theta$):** $76,123$ packets
- **Baseline Evasion Rate:** **$11.8476\%$**
- **Mean Attack RMSE:** $17.028671 \pm 423.012390$
- **Minimum Attack RMSE:** $0.001764$
- **Maximum Attack RMSE:** $99,328.125000$

### 4.3. Analysis of Baseline Evasion (11.85%)
Even without adversarial perturbation, ~11.85% of packets in the Mirai capture have low reconstruction error ($\text{RMSE} \le \theta$). In real botnet telemetry, a fraction of attack packets consist of normal connection acknowledgments, TCP handshakes, or standard keep-alives that naturally resemble benign traffic. The remaining $88.15\%$ (566,393 packets) represent the high-volume scanning, brute-force payloads, and telnet command flooding that strongly trigger Kitsune's micro-autoencoders ($\text{RMSE}$ up to $99,328$).

**Objective for Phase 3 (Offensive Engine):**
Mutate the $566,393$ detected attack packets via WGAN-GP and gradient baselines (FGSM/DeepFool) under domain constraints so their RMSE drops below $\theta = 0.453419$, driving the evasion rate toward $100\%$.

---

## 5. Output Metadata Profile (`data/processed/threshold_config.yaml`)
```yaml
threshold: 0.4534189701080322
percentile: 99.9
grace_period_excluded: 55000
benign_metrics:
  num_evaluated: 66621
  mean_rmse: 0.015084798447787762
  std_rmse: 0.04445762187242508
  min_rmse: 0.002348531037569046
  max_rmse: 6.930492401123047
  false_positive_count: 67
  false_positive_rate_pct: 0.10056888968943727
attack_metrics:
  num_evaluated: 642516
  detected_count: 566393
  evaded_count: 76123
  detection_rate_pct: 88.15235729538253
  evasion_rate_pct: 11.847642704617472
  mean_rmse: 17.028671264648438
  std_rmse: 423.01239013671875
  min_rmse: 0.001764054992236197
  max_rmse: 99328.125
```

---

## 6. Verification Test Suite (`tests/test_threshold.py`)
- `test_threshold_config_metadata`: Confirms calibrated $\theta \in [0.1, 1.0]$, FPR within expected statistical tolerance ($\approx 0.1\%$), and detection rate $> 80\%$.
- `test_torch_kitnet_threshold_decision`: Confirms post-convergence benign sample scores reliably fall below $\theta$.
- **Full Suite Status:** 10 / 10 tests passing across all project test modules.
