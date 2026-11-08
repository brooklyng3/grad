# Phase 2 Summary: Defender Evaluation & Alert Threshold Calibration

This document provides a high-level executive summary of the evaluation methodology, calibrated anomaly alert threshold $\theta$, and baseline defensive metrics established during Phase 2.

---

## 1. Objectives Achieved
1. **GPU-Accelerated Evaluation Pipeline (`src/defender/eval_threshold.py`):** Leveraged `PyTorchKitNET` on the NVIDIA GeForce GTX 1660 SUPER, scoring packets in batches of 512 (21,956 pkts/s benign calibration, 40,200 pkts/s attack).
2. **Grace-Period Exclusion Safeguard Enforced:** Filtered out the first 55,000 packets (`FMgrace + ADgrace`) to eliminate unrepresentative $0.0$ scores and early SGD fluctuations.
3. **Anomaly Threshold Calibration:** Calibrated the alert threshold $\theta$ on $33,311$ post-convergence benign packets (even parity of the range $[55000, 121620]$) at the 99.9th percentile:
   $$\theta = \mathbf{0.455986}$$
   and measured FPR out-of-sample on the disjoint $33,310$-packet odd-parity split: **$0.096\%$** ($32 / 33,310$).
4. **Baseline Evasion Benchmarking:** Evaluated all $642,516$ clean Mirai attack packets against $\theta$:
   - **Baseline Detection Rate (TPR):** **$88.15\%$** ($566,393 / 642,516$)
   - **Baseline Evasion Rate:** **$11.85\%$** ($76,123 / 642,516$)
   - **Mean Attack RMSE:** $17.03$ (well above $\theta = 0.453$)
5. **Persistent Metadata Configuration:** Saved complete statistical distribution metrics and threshold values to `data/processed/threshold_config.yaml`.
6. **Automated Verification:** Added `tests/test_threshold.py` verifying split sizes, out-of-sample FPR tolerance, and runtime decision sanity (10 / 10 at the time; 28 / 28 after Phase 6).

---

## 2. Key Metrics Summary

| Metric | Calibration / Validation Benign | Clean Mirai Attack Set |
| :--- | :--- | :--- |
| **Packet Count** | 33,311 / 33,310 packets | 642,516 packets |
| **Evaluation Time** | 1.52 seconds | 15.98 seconds |
| **Throughput** | 21,956 pkts/s | 40,200 pkts/s |
| **Mean RMSE** | $0.015203 \pm 0.052498$ (cal) | $17.028671 \pm 423.012390$ |
| **Min - Max RMSE** | $0.002396 - 6.930492$ (cal) | $0.001764 - 99,328.125$ |
| **Calibrated $\theta$ (99.9%)** | **0.455986** | — |
| **Out-of-sample FPR / Detection Rate** | $\text{FPR} = 0.096\%$ | $\text{TPR} = 88.15\%$ |
| **Baseline Evasion Rate** | — | $\mathbf{11.85\%}$ |

---

## 3. Artifacts Produced
- `src/defender/eval_threshold.py`: Standalone GPU evaluation and calibration module.
- `data/processed/threshold_config.yaml`: Persistent YAML file containing $\theta$ and metrics.
- `tests/test_threshold.py`: Pytest suite verifying threshold integrity and test coverage.
