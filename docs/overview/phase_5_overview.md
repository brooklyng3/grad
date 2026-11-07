# Phase 5 Summary: Unified Metrics, Reporting & End-to-End Pipeline CLI

This document provides an executive summary of the unified end-to-end evaluation pipeline, reporting utilities, and comprehensive experimental findings across both early scanning and volumetric flood attack paradigms against the Kitsune Network Intrusion Detection System (NIDS).

---

## 1. Objectives Achieved

1. **Evaluation Metrics & Reporting Engine (`src/utils/metrics.py`):**
   - Standardized anomaly detection evaluation: Confusion Matrix ($\text{TP}, \text{FP}, \text{TN}, \text{FN}$), True Positive Rate ($\text{TPR}$ / Recall), False Positive Rate ($\text{FPR}$), Precision, $\text{F}_1$-Score, and Balanced Accuracy.
   - Exact ROC curve computation and Area Under Curve ($\text{ROC-AUC}$).
   - Distortion vs. Evasion trade-off curve aggregation ($L_\infty, L_2, L_0$ vs. $\text{ASR}_{\text{cond}}$).
   - Corrected population extrapolation for subsample evaluation:
     $$\text{ER}_{\text{total}} = \frac{N_{\text{pre-evaded}} + |\mathcal{M}_{\text{det}}| \cdot \left(\frac{\text{ASR}_{\text{cond}}}{100}\right)}{N_{\text{total\_attacks}}} \times 100\%$$
   - Structured export utilities for JSON, CSV, and GitHub-flavored markdown tables for thesis documentation.

2. **Unified CLI Orchestrator (`run_pipeline.py`):**
   - Single command-line interface controlling all defensive evaluation and offensive adversarial benchmarking routines.
   - Five operational modes:
     - `--mode evaluate_clean`: High-throughput GPU scoring of clean traffic; validates baseline FPR and ROC-AUC.
     - `--mode fgsm_sweep`: Parameterized perturbation sweeps across $\epsilon \in [0.01, 2.0]$.
     - `--mode pgd_attack`: Comparative evaluation of multi-step iterative PGD and Manifold Traffic-Shaping.
     - `--mode train_wgan`: Closed-loop adversarial feedback training of the WGAN-GP generator (supports `--manifold_shaping`).
     - `--mode evaluate_all`: End-to-end multi-paradigm benchmark evaluating both scanning and volumetric flood regimes.

3. **Rigorous Test Suite:**
   - Added `tests/test_metrics.py` (5 tests) and verified subsample extrapolation in `tests/test_pipeline.py`.
   - **All 22 unit tests passing** across the complete project test suite.
   - 100% compliant with `black` formatting and `flake8` linting standards.

---

## 2. Definitive Comparative Benchmark Summary

Evaluated across two distinct attack regimes on $1,000$ confirmed detected Mirai attack samples ($\text{RMSE}_{\text{orig}} > \theta = 0.453419$) alongside $66,621$ post-grace benign packets:

### Phase A: Reconnaissance / Scanning Traffic (Early Attack, Baseline $\text{RMSE} \approx 3.80$)

| Paradigm / Attack | Perturbation Constraint | Mean Attack RMSE | Conditional ASR ($\text{ASR}_{\text{cond}}$) | Overall Evasion Rate ($\text{ER}_{\text{total}}$) | Mean $L_\infty$ Distortion |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Clean Baseline** | None (Unperturbed) | $3.7961$ | $0.00\%$ | $11.85\%$ | $0.0000$ |
| **Single-Step FGSM** | $\epsilon = 0.05 \cdot \mathbf{s}$ | $3.7664$ | $0.00\%$ | $11.85\%$ | $0.0500$ |
| **Single-Step FGSM** | $\epsilon = 0.25 \cdot \mathbf{s}$ | $3.6568$ | $0.10\%$ | $11.94\%$ | $0.2500$ |
| **Multi-Step PGD (Box)** | 50 steps, $\epsilon = 2.0 \cdot \mathbf{s}$ | $2.6743$ | $7.20\%$ | $18.19\%$ | $1.9968$ |
| **Multi-Step PGD (Manifold)** | 50 steps, $[\mathbf{min}, \mathbf{max}]$ | **$0.0845$** | **$100.00\%$** | **$100.00\%$** | $8.8060$ |
| **Polymorphic WGAN-GP (Box)** | $\epsilon = 2.0 \cdot \mathbf{s}$, closed loop | $2.9505$ | $0.00\%$ | $11.85\%$ | $1.9999$ |
| **Polymorphic WGAN-GP (Manifold)** | Envelope Shaping, single pass | **$0.2276$** | **$100.00\%$** | **$100.00\%$** | $9.5686$ |

---

### Phase B: Volumetric Flooding Traffic (Heavy Attack, Baseline $\text{RMSE} \approx 13.51$)

| Paradigm / Attack | Perturbation Constraint | Mean Attack RMSE | Conditional ASR ($\text{ASR}_{\text{cond}}$) | Overall Evasion Rate ($\text{ER}_{\text{total}}$) | Mean $L_\infty$ Distortion |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Clean Baseline** | None (Unperturbed) | $13.5061$ | $0.00\%$ | $11.85\%$ | $0.0000$ |
| **Single-Step FGSM** | $\epsilon = 0.05 \cdot \mathbf{s}$ | $13.4756$ | $0.00\%$ | $11.85\%$ | $0.0500$ |
| **Single-Step FGSM** | $\epsilon = 0.25 \cdot \mathbf{s}$ | $13.3556$ | $0.00\%$ | $11.85\%$ | $0.2500$ |
| **Multi-Step PGD (Box)** | 50 steps, $\epsilon = 2.0 \cdot \mathbf{s}$ | $12.3060$ | $0.00\%$ | $11.85\%$ | $2.0000$ |
| **Multi-Step PGD (Manifold)** | 50 steps, $[\mathbf{min}, \mathbf{max}]$ | **$0.0899$** | **$100.00\%$** | **$100.00\%$** | $35.0449$ |
| **Polymorphic WGAN-GP (Box)** | $\epsilon = 2.0 \cdot \mathbf{s}$, closed loop | $12.4103$ | $0.00\%$ | $11.85\%$ | $1.9998$ |
| **Polymorphic WGAN-GP (Manifold)** | Envelope Shaping, single pass | **$0.2285$** | **$100.00\%$** | **$100.00\%$** | $35.0841$ |

### Baseline Clean Detection Metrics (120,164 Packets Scored)
- **Calibrated Threshold ($\theta$):** $0.453419$ (99.9th percentile post-grace benign RMSE)
- **$\text{ROC-AUC}$:** $0.9418$
- **True Positive Rate ($\text{TPR}$ / Recall):** $88.13\%$
- **False Positive Rate ($\text{FPR}$):** $0.1006\%$
- **Precision:** $99.86\%$ | **$\text{F}_1$-Score:** $0.9363$ | **Overall Accuracy:** $94.65\%$
- **Inference Throughput:** $24,000+$ pkts/sec on NVIDIA GTX 1660 SUPER

---

## 3. Core Insights & Thesis Findings

1. **Why Standard Gradient Attacks Stall on Volumetric NIDS Features:**
   In volumetric DoS attacks like Mirai, packet rates and byte counts reach $15\times$ to $35\times$ the maximum benign baseline. Because autoencoders unnormalize inputs with $\mathbf{s} = \mathbf{norm\_max} - \mathbf{norm\_min}$ and process them through bounded $\text{sigmoid}$ activation functions ($\sigma(z) \in (0, 1)$), extreme inputs push neurons into deep saturation. In this regime, gradients vanish ($\sigma'(z) \approx 0$). Bounded additive perturbations ($\epsilon \le 2.0$) can slightly attenuate moderate scanning attacks ($7.20\%$ ASR) but cannot bridge the volumetric flooding gap ($0.00\%$ ASR).

2. **The Power of Manifold Traffic-Shaping in WGAN-GP:**
   By equipping `PolymorphicGenerator` with a Manifold-Shaping Envelope mode that dynamically attenuates volumetric statistics down into the empirical benign feature manifold ($[0, 1]$ in normalized space), Kitsune's ensemble reconstruction error collapses to $0.2285 \ll \theta = 0.4534$. This achieves **$100.00\%$ Conditional ASR** on both scanning and volumetric flooding packets, matching iterative PGD while operating orders of magnitude faster via **single-pass neural inference** ($>100,000$ pkts/sec).

3. **Mathematical Origin of Inactive Features ($L_0 \approx 53$):**
   All 100 features in AfterImage are continuous damped statistical summaries (weights, means, variances, radiuses, covariances, correlations). There are no raw IP or port headers. The ~47 features that remain unmodified during FGSM/PGD do so because:
   - Saturated autoencoder sigmoids produce vanishing gradients ($\frac{\partial \text{RMSE}}{\partial x_j} \approx 0$).
   - Inactive 2D host-pair channels remain at zero activity ($x_j = 0.0$), and any gradient descent step attempting to step negative is immediately clamped back to $0.0$ by `enforce_feature_bounds`.
