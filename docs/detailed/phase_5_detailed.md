# Phase 5 Comprehensive Technical Log: Unified Metrics, Reporting & End-to-End Pipeline CLI

This document provides an exhaustive, granular engineering record of the unified evaluation orchestrator, reporting suite, metric mathematical formulations, and experimental telemetry produced during Phase 5.

---

## 1. Mathematical Formulations (`src/utils/metrics.py`)

### 1.1. Binary Anomaly Detection Metrics
Let $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^N$ be the evaluation dataset where $y_i \in \{0, 1\}$ ($0 = \text{Benign}, 1 = \text{Malicious}$).
Given the anomaly reconstruction score $s_i = \text{RMSE}(\mathbf{x}_i)$ and calibrated threshold $\theta$:

$$\hat{y}_i = \mathbb{I}(s_i > \theta)$$

The contingency counts are defined as:
- **True Positives ($\text{TP}$):** $\sum_{i=1}^N \mathbb{I}(y_i = 1 \land \hat{y}_i = 1)$
- **False Positives ($\text{FP}$):** $\sum_{i=1}^N \mathbb{I}(y_i = 0 \land \hat{y}_i = 1)$
- **True Negatives ($\text{TN}$):** $\sum_{i=1}^N \mathbb{I}(y_i = 0 \land \hat{y}_i = 0)$
- **False Negatives ($\text{FN}$):** $\sum_{i=1}^N \mathbb{I}(y_i = 1 \land \hat{y}_i = 0)$

Derived classification metrics:
$$\text{TPR} = \frac{\text{TP}}{\text{TP} + \text{FN}}, \quad \text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}$$
$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}, \quad \text{F}_1 = \frac{2 \cdot \text{Precision} \cdot \text{TPR}}{\text{Precision} + \text{TPR}}$$
$$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{N}, \quad \text{Balanced Accuracy} = \frac{\text{TPR} + (1 - \text{FPR})}{2}$$

### 1.2. ROC Curve and Area Under the Curve (ROC-AUC)
Parametric curve $(\text{FPR}(\tau), \text{TPR}(\tau))$ evaluated across varying decision thresholds $\tau \in [0, \infty)$:

$$\text{AUC} = \int_{0}^{1} \text{TPR}(\text{FPR}^{-1}(u)) \, du \approx \sum_{k=1}^{K-1} \frac{\text{TPR}_k + \text{TPR}_{k+1}}{2} \cdot (\text{FPR}_{k+1} - \text{FPR}_k)$$

### 1.3. Decoupled Evasion Metrics
To prevent metric distortion caused by malicious packets that naturally resemble benign traffic:

$$\text{ASR}_{\text{cond}} = \frac{\sum_{i \in \mathcal{M}_{\text{det}}} \mathbb{I}(\text{RMSE}(\mathbf{x}_{\text{adv}}^{(i)}) \le \theta)}{|\mathcal{M}_{\text{det}}|} \times 100\%$$
$$\text{ER}_{\text{total}} = \frac{N_{\text{pre-evaded}} + \sum_{i \in \mathcal{M}_{\text{det}}} \mathbb{I}(\text{RMSE}(\mathbf{x}_{\text{adv}}^{(i)}) \le \theta)}{N_{\text{total\_attacks}}} \times 100\%$$

Where:
- $|\mathcal{M}_{\text{det}}| = 566,393$ detected Mirai packets.
- $N_{\text{pre-evaded}} = 76,123$ naturally evading baseline attack packets.
- $N_{\text{total\_attacks}} = 642,516$ total attack population.

---

## 2. CLI Architecture & Execution Modes (`run_pipeline.py`)

The root script `run_pipeline.py` integrates all modular defensive and offensive subsystems into a unified interface:

```text
run_pipeline.py
├── Core Components Loader
│   ├── PyTorchKitNET.from_file(model_path)
│   ├── threshold_config.yaml -> theta = 0.453419
│   ├── mirai_features.npy (np.load, mmap_mode='r')
│   └── PipelineEvaluator (CUDA batch_size=256)
├── Mode: evaluate_clean
├── Mode: fgsm_sweep
├── Mode: pgd_attack
├── Mode: train_wgan
└── Mode: evaluate_all
```

### CLI Command Options
```bash
python run_pipeline.py \
  --mode {evaluate_clean, fgsm_sweep, pgd_attack, train_wgan, evaluate_all} \
  [--config configs/kitsune_config.yaml] \
  [--wgan_config configs/wgan_config.yaml] \
  [--threshold_config data/processed/threshold_config.yaml] \
  [--features_path data/processed/mirai_features.npy] \
  [--model_path data/processed/kitsune_mirai_model.pkl] \
  [--output_dir reports] \
  [--samples 1000] \
  [--batch_size 256] \
  [--device cuda]
```

---

## 3. Experimental Telemetry & Verification Data

### 3.1. Clean Baseline Detection Telemetry (`evaluate_clean`)
- **Benign Validation Subset:** $66,621$ packets (post-grace period: indices $55,000$ to $121,621$).
- **Malicious Evaluation Subset:** $53,543$ packets (sampled across all $642,516$ attack packets).
- **Execution Time:** $4.88$ seconds ($24,604.4$ pkts/sec).
- **Confusion Matrix:**
  - $\text{TN} = 66,554$
  - $\text{FP} = 67$ ($\text{FPR} = 0.1006\%$, verifying the calibrated 99.9th percentile threshold)
  - $\text{FN} = 6,358$ (Naturally evading baseline attack packets)
  - $\text{TP} = 47,185$ ($\text{TPR} = 88.13\%$)
- **Precision:** $99.86\%$
- **$\text{F}_1$-Score:** $0.9363$
- **$\text{ROC-AUC}$:** $0.9418$
- **Overall Accuracy:** $94.65\%$

### 3.2. Single-Step FGSM Parameter Sweep (`fgsm_sweep`)
Evaluated across $1,000$ confirmed detected attack packets ($\text{RMSE}_{\text{orig}} = 13.3351$):

| $\epsilon$ Budget | $\text{ASR}_{\text{cond}}$ | $\text{ER}_{\text{total}}$ | Mean RMSE | Mean $L_\infty$ | Mean $L_2$ | Mean $L_1$ | Active Features ($L_0$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $0.01$ | $0.00\%$ | $11.85\%$ | $13.3290$ | $0.0100$ | $0.0974$ | $0.4612$ | $52.68$ |
| $0.05$ | $0.00\%$ | $11.85\%$ | $13.3047$ | $0.0500$ | $0.4851$ | $2.3060$ | $52.68$ |
| $0.10$ | $0.00\%$ | $11.85\%$ | $13.2744$ | $0.1000$ | $0.9683$ | $4.6120$ | $52.68$ |
| $0.25$ | $0.00\%$ | $11.85\%$ | $13.1846$ | $0.2500$ | $2.4088$ | $11.5300$ | $52.68$ |
| $0.50$ | $0.00\%$ | $11.85\%$ | $13.0347$ | $0.5000$ | $4.7968$ | $23.0600$ | $52.68$ |
| $1.00$ | $0.00\%$ | $11.85\%$ | $12.7497$ | $1.0000$ | $9.4291$ | $46.1200$ | $52.68$ |
| $2.00$ | $0.00\%$ | $11.85\%$ | $12.3114$ | $2.0000$ | $18.7428$ | $92.2400$ | $52.68$ |

*Observation:* FGSM perturbations modify exclusively the unconstrained features ($52.68$ features out of $100$), leaving immutable IP headers, protocol identifiers, and ports untouched. However, single-step perturbations cannot overcome the volumetric magnitude of Mirai flooding.

### 3.3. Multi-Step PGD & Manifold Traffic-Shaping Telemetry (`pgd_attack`)
Evaluated across $1,000$ confirmed detected attack packets:

| Configuration | Steps | $\text{ASR}_{\text{cond}}$ | $\text{ER}_{\text{total}}$ | Mean RMSE | Min RMSE | Max RMSE | Mean $L_\infty$ | Mean $L_2$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PGD (Box, $\epsilon=1.0$)** | 20 | $0.00\%$ | $11.85\%$ | $12.7293$ | $1.1271$ | $14.7568$ | $1.0000$ | $3.8720$ |
| **PGD (Box, $\epsilon=2.0$)** | 50 | $0.00\%$ | $11.85\%$ | $12.1354$ | $0.6012$ | $14.1086$ | $2.0000$ | $7.6623$ |
| **Manifold Shaping** | 20 | **$100.00\%$** | **$12.00\%$** | **$0.0906$** | $0.0472$ | $0.1928$ | $34.6493$ | $72.8434$ |
| **Manifold Shaping** | 50 | **$100.00\%$** | **$12.00\%$** | **$0.0903$** | $0.0472$ | $0.1928$ | $34.6495$ | $72.8434$ |

*Observation:* Under Manifold Traffic-Shaping, every single detected packet ($1,000 / 1,000$) falls below the anomaly threshold ($\max \text{RMSE} = 0.1928 < \theta = 0.4534$).

---

## 4. Hardware Verification & Resource Compliance

- **GPU:** NVIDIA GeForce GTX 1660 SUPER (6 GB VRAM).
  - Peak VRAM during evaluation sweeps: **$1.64$ GB** (strict compliance with $< 4$ GB rule).
  - Memory freed via explicit `torch.cuda.empty_cache()` calls between cycles.
- **CPU:** Intel Core i5-9400F (6 physical cores).
  - Multiprocessing workers strictly budgeted $\le 4$.
  - Load average remained $< 2.2$.
- **System Memory:** 16 GB DDR4.
  - Memory-mapped arrays (`mmap_mode='r'`) prevented multi-gigabyte heap allocations.
  - Resident Set Size (RSS) remained $< 1.2$ GB.

---

## 5. Artifact Manifest

| Path | Description |
| :--- | :--- |
| `src/utils/metrics.py` | Anomaly detection metrics, ROC-AUC, distortion aggregation, and report exporters. |
| `run_pipeline.py` | Unified CLI orchestrator supporting 5 execution modes. |
| `tests/test_metrics.py` | Pytest suite covering detection metrics, ROC-AUC, trade-offs, and formatting. |
| `reports/clean_detection_metrics.json` | Baseline detection telemetry (TPR, FPR, F1, ROC-AUC). |
| `reports/evaluation_table.md` | Final comparative benchmark table across all paradigms. |
| `reports/fgsm_tradeoff.csv` | Granular FGSM perturbation budget sweep data. |
| `reports/pgd_benchmark.json` | Granular multi-step PGD and manifold traffic-shaping telemetry. |
| `docs/overview/phase_5_overview.md` | Executive overview of Phase 5. |
| `docs/detailed/phase_5_detailed.md` | Exhaustive technical documentation of Phase 5. |
