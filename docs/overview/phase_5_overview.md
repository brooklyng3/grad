# Phase 5 Summary: Unified Metrics, Reporting & End-to-End Pipeline CLI

This document provides an executive summary of the unified end-to-end evaluation pipeline, reporting utilities, and comprehensive experimental findings across all attack paradigms against the Kitsune Network Intrusion Detection System (NIDS).

---

## 1. Objectives Achieved

1. **Evaluation Metrics & Reporting Engine (`src/utils/metrics.py`):**
   - Standardized anomaly detection evaluation: Confusion Matrix ($\text{TP}, \text{FP}, \text{TN}, \text{FN}$), True Positive Rate ($\text{TPR}$ / Recall), False Positive Rate ($\text{FPR}$), Precision, $\text{F}_1$-Score, and Balanced Accuracy.
   - Exact ROC curve computation and Area Under Curve ($\text{ROC-AUC}$).
   - Distortion vs. Evasion trade-off curve aggregation ($L_\infty, L_2, L_0$ vs. $\text{ASR}_{\text{cond}}$).
   - Structured export utilities for JSON, CSV, and GitHub-flavored markdown tables for thesis documentation.

2. **Unified CLI Orchestrator (`run_pipeline.py`):**
   - Single command-line interface controlling all defensive evaluation and offensive adversarial benchmarking routines.
   - Five operational modes:
     - `--mode evaluate_clean`: High-throughput GPU scoring of clean traffic; validates baseline FPR and ROC-AUC.
     - `--mode fgsm_sweep`: Parameterized perturbation sweeps across $\epsilon \in [0.01, 2.0]$.
     - `--mode pgd_attack`: Comparative evaluation of multi-step iterative PGD and Manifold Traffic-Shaping.
     - `--mode train_wgan`: Closed-loop adversarial feedback training of the WGAN-GP generator against PyTorchKitNET.
     - `--mode evaluate_all`: End-to-end multi-paradigm benchmark generating comprehensive thesis report tables.

3. **Rigorous Test Suite:**
   - Added `tests/test_metrics.py` (5 tests).
   - **All 22 unit tests passing** across the complete project test suite.
   - 100% compliant with `black` formatting and `flake8` linting standards.

---

## 2. Definitive Comparative Benchmark Summary

Evaluated on $1,000$ confirmed detected Mirai attack samples ($\text{RMSE}_{\text{orig}} > \theta = 0.453419$) alongside $66,621$ post-grace benign packets:

| Paradigm / Attack | Perturbation Constraint | Mean Attack RMSE | Conditional ASR ($\text{ASR}_{\text{cond}}$) | Overall Evasion Rate ($\text{ER}_{\text{total}}$) | Mean $L_\infty$ Distortion |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Clean Baseline** | None (Unperturbed) | $13.3351$ | $0.00\%$ | $11.85\%$ | $0.0000$ |
| **Single-Step FGSM** | $\epsilon = 0.05 \cdot \mathbf{s}$ | $13.3047$ | $0.00\%$ | $11.85\%$ | $0.0500$ |
| **Single-Step FGSM** | $\epsilon = 0.25 \cdot \mathbf{s}$ | $13.1846$ | $0.00\%$ | $11.85\%$ | $0.2500$ |
| **Multi-Step PGD** | 50 steps, $\epsilon = 2.0 \cdot \mathbf{s}$ | $12.1354$ | $0.00\%$ | $11.85\%$ | $2.0000$ |
| **Polymorphic WGAN-GP** | $\epsilon = 2.0 \cdot \mathbf{s}$, closed loop | $12.1385$ | $0.00\%$ | $11.85\%$ | $2.0000$ |
| **Manifold Traffic-Shaping** | 50 steps, $[\mathbf{min}, \mathbf{max}]$ | **$0.0903$** | **$100.00\%$** | **$12.00\%$** | $34.6495$ |

### Baseline Clean Detection Metrics (120,164 Packets Scored)
- **Calibrated Threshold ($\theta$):** $0.453419$ (99.9th percentile post-grace benign RMSE)
- **$\text{ROC-AUC}$:** $0.9418$
- **True Positive Rate ($\text{TPR}$ / Recall):** $88.13\%$
- **False Positive Rate ($\text{FPR}$):** $0.1006\%$
- **Precision:** $99.86\%$
- **$\text{F}_1$-Score:** $0.9363$
- **Overall Accuracy:** $94.65\%$
- **Inference Throughput:** $24,600+$ pkts/sec on NVIDIA GTX 1660 SUPER

---

## 3. Core Insights & Thesis Findings

1. **Why Standard Gradient Attacks Stall on Volumetric NIDS Features:**
   In volumetric DoS attacks like Mirai, packet rates and byte counts reach $15\times$ to $35\times$ the maximum benign baseline. Because autoencoders unnormalize inputs with $\mathbf{s} = \mathbf{norm\_max} - \mathbf{norm\_min}$ and process them through bounded $\text{sigmoid}$ activation functions ($\sigma(z) \in (0, 1)$), extreme inputs push neurons into deep saturation. In this regime, gradients vanish ($\sigma'(z) \approx 0$). Local $L_\infty$ perturbations ($\epsilon \le 2.0$) reduce reconstruction RMSE slightly but cannot bridge the volumetric gap.

2. **The Power of Manifold Traffic-Shaping:**
   Projecting adversarial perturbations onto the empirical benign feature manifold ($[\mathbf{norm\_min}, \mathbf{norm\_max}]$) eliminates the volumetric signature without corrupting packet syntax. Under Manifold Traffic-Shaping, Kitsune's ensemble reconstruction error collapses to $0.0903 \ll \theta = 0.4534$, achieving **$100.0\%$ Conditional ASR** on previously detected packets.

3. **Scientific Decoupling of Evasion Metrics:**
   Reporting only "Evasion Rate" conflates natural benign-resemblance ($11.85\%$ of Mirai packets resemble idle benign traffic) with active evasion. Separating **$\text{ASR}_{\text{cond}}$** from **$\text{ER}_{\text{total}}$** provides true visibility into attacker effectiveness.
