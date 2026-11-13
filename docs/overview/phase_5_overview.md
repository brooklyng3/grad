# Phase 5 Summary: Unified Metrics, Reporting & End-to-End Pipeline CLI

This document provides an executive summary of the unified end-to-end evaluation pipeline, reporting utilities, and comprehensive experimental findings across both early scanning and volumetric flood attack paradigms against the Kitsune Network Intrusion Detection System (NIDS).

> [!WARNING]
> **Results corrected in Phase 6.** The tables and insights below are the
> post-fix benchmark (seeded, disjoint holdout, shared ε = 2.0). The original
> Phase 5 numbers were in-sample for the WGAN and are superseded. Most
> importantly, the suite contains a **clamp-only** baseline and an
> **untrained-generator** control: the trained generator beats the clamp on
> RMSE and benign proximity while the untrained generator scores 0%. See
> `phase_6_detailed.md`. Latencies are single-run measurements and vary by
> roughly ±20%. The canonical table, including the benign-kNN proximity column,
> is `reports/evaluation_table.md`.

---

## 1. Objectives Achieved

1. **Evaluation Metrics & Reporting Engine (`src/utils/metrics.py`):**
   - Standardized anomaly detection evaluation: Confusion Matrix ($\text{TP}, \text{FP}, \text{TN}, \text{FN}$), True Positive Rate ($\text{TPR}$ / Recall), False Positive Rate ($\text{FPR}$), Precision, $\text{F}_1$-Score, and Balanced Accuracy.
   - Exact ROC curve computation and Area Under Curve ($\text{ROC-AUC}$).
   - Distortion vs. Evasion trade-off curve aggregation ($L_\infty, L_2, L_0$ vs. $\text{ASR}_{\text{cond}}$).
   - Attack-attribution ablations: envelope clamp only and untrained manifold generator (Phase 6).
   - Best-of-k retry generation and bounded DeepFool benchmark (Phase 6).
   - Corrected population extrapolation for subsample evaluation:
     $$\text{ER}_{\text{total}} = \frac{N_{\text{pre-evaded}} + |\mathcal{M}_{\text{det}}| \cdot \left(\frac{\text{ASR}_{\text{cond}}}{100}\right)}{N_{\text{total\_attacks}}} \times 100\%$$
   - Generation latency tracking (ms / 1,000 pkts) on GPU hardware.

2. **Unified CLI Orchestrator (`run_pipeline.py`):**
   - Single command-line interface controlling all defensive evaluation and offensive adversarial benchmarking routines.
   - Five operational modes:
     - `--mode evaluate_clean`: High-throughput GPU scoring of clean traffic; validates baseline FPR and ROC-AUC.
     - `--mode fgsm_sweep`: Parameterized perturbation sweeps across $\epsilon \in [0.01, 2.0]$.
     - `--mode pgd_attack`: Comparative evaluation of multi-step iterative PGD and Manifold Traffic-Shaping.
     - `--mode train_wgan`: Closed-loop adversarial feedback training of the WGAN-GP generator (supports `--manifold_shaping`).
     - `--mode evaluate_all`: End-to-end multi-paradigm benchmark evaluating both scanning and volumetric flood regimes, including clamp-only and untrained-generator ablations, DeepFool, and best-of-k WGAN.
   - Shared `--eps`, `--seed`, `--retries`, `--train_epochs`, and `--deepfool_max_eps` flags (Phase 6).

3. **Rigorous Test Suite:**
   - Added `tests/test_metrics.py` (6 tests) and verified subsample extrapolation in `tests/test_pipeline.py`.
   - **All 37 unit tests passing** across the complete project test suite (22 at the time of Phase 5; the Phase 6 additions assert evasion behaviour).
   - 100% compliant with `black` formatting and `flake8` linting standards.

---

## 2. Definitive Comparative Benchmark Summary

Evaluated across two disjoint attack holdout regimes on $1,000$ confirmed detected Mirai attack samples each ($\text{RMSE}_{\text{orig}} > \theta = 0.455986$) alongside $33,310$ out-of-sample benign validation packets:

### Phase A: Reconnaissance / Scanning Traffic (Early Attack, Baseline $\text{RMSE} = 5.3381$)

| Paradigm / Attack | Perturbation Constraint | Draws/pkt | Mean Attack RMSE | $\text{ASR}_{\text{cond}}$ | $\text{ER}_{\text{total}}$ | Mean $L_\infty$ | Latency (ms / 1k pkts) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Clean Baseline** | None | 0.00 | $5.3381$ | $0.00\%$ | $11.85\%$ | $0.0000$ | $0.0\text{ ms}$ |
| **Envelope Clamp Only** | Unbudgeted $[\min,\max]$ | 1.00 | **$0.1136$** | **$100.00\%$** | **$100.00\%$** | $12.9735$ | $2.4\text{ ms}$ |
| **Single-Step FGSM** | $\epsilon = 0.05 \cdot \mathbf{s}$ | 1.00 | $5.3076$ | $0.00\%$ | $11.85\%$ | $0.0500$ | $50.7\text{ ms}$ |
| **Single-Step FGSM** | $\epsilon = 0.25 \cdot \mathbf{s}$ | 1.00 | $5.1910$ | $0.00\%$ | $11.85\%$ | $0.2500$ | $53.2\text{ ms}$ |
| **Multi-Step PGD (Box)** | 50 steps, $\epsilon = 2.0 \cdot \mathbf{s}$ | 1.00 | $4.1601$ | $2.30\%$ | $13.88\%$ | $1.9983$ | $2321.2\text{ ms}$ |
| **DeepFool (bounded)** | 50 iters, $\epsilon \le 2.0 \cdot \mathbf{s}$ | 1.00 | $4.5108$ | $0.00\%$ | $11.85\%$ | $1.9683$ | $2387.1\text{ ms}$ |
| **Multi-Step PGD (Manifold)** | Unbudgeted $[\min,\max]$, 50 steps | 1.00 | **$0.0909$** | **$100.00\%$** | **$100.00\%$** | $13.0325$ | $2170.6\text{ ms}$ |
| **WGAN-GP (Box, 1 draw)** | $\epsilon = 2.0 \cdot \mathbf{s}$ | 1.00 | $4.1925$ | $1.60\%$ | $13.43\%$ | $1.9981$ | $19.2\text{ ms}$ |
| **WGAN-GP (Box, best-of-10)** | $\epsilon = 2.0 \cdot \mathbf{s}$, retry | 9.85 | $4.1883$ | $1.60\%$ | $13.43\%$ | $1.9983$ | $153.5\text{ ms}$ |
| **WGAN-GP (Box, 10-step refine)** | $\epsilon$ per step, aggregate | 9.68 | $1.5275$ | $3.80\%$ | $15.20\%$ | $10.4344$ | $148.7\text{ ms}$ |
| **WGAN-GP (Manifold, trained)** | Unbudgeted learned target | 1.00 | **$0.0779$** | **$100.00\%$** | **$100.00\%$** | $13.8967$ | $7.6\text{ ms}$ |
| **WGAN-GP (Manifold, untrained)** | Same forward, random weights | 1.00 | $5.6284$ | $0.00\%$ | $11.85\%$ | $6.1065$ | $3.0\text{ ms}$ |

---

### Phase B: Volumetric Flooding Traffic (Heavy Attack, Baseline $\text{RMSE} = 13.4764$)

| Paradigm / Attack | Perturbation Constraint | Draws/pkt | Mean Attack RMSE | $\text{ASR}_{\text{cond}}$ | $\text{ER}_{\text{total}}$ | Mean $L_\infty$ | Latency (ms / 1k pkts) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Clean Baseline** | None | 0.00 | $13.4764$ | $0.00\%$ | $11.85\%$ | $0.0000$ | $0.0\text{ ms}$ |
| **Envelope Clamp Only** | Unbudgeted $[\min,\max]$ | 1.00 | **$0.1092$** | **$100.00\%$** | **$100.00\%$** | $34.9663$ | $2.1\text{ ms}$ |
| **Single-Step FGSM** | $\epsilon = 0.05 \cdot \mathbf{s}$ | 1.00 | $13.4459$ | $0.00\%$ | $11.85\%$ | $0.0500$ | $50.4\text{ ms}$ |
| **Single-Step FGSM** | $\epsilon = 0.25 \cdot \mathbf{s}$ | 1.00 | $13.3258$ | $0.00\%$ | $11.85\%$ | $0.2500$ | $45.5\text{ ms}$ |
| **Multi-Step PGD (Box)** | 50 steps, $\epsilon = 2.0 \cdot \mathbf{s}$ | 1.00 | $12.2761$ | $0.00\%$ | $11.85\%$ | $2.0000$ | $2274.2\text{ ms}$ |
| **DeepFool (bounded)** | 50 iters, $\epsilon \le 2.0 \cdot \mathbf{s}$ | 1.00 | $12.9098$ | $0.00\%$ | $11.85\%$ | $2.0000$ | $2135.0\text{ ms}$ |
| **Multi-Step PGD (Manifold)** | Unbudgeted $[\min,\max]$, 50 steps | 1.00 | **$0.0896$** | **$100.00\%$** | **$100.00\%$** | $34.9754$ | $1668.1\text{ ms}$ |
| **WGAN-GP (Box, 1 draw)** | $\epsilon = 2.0 \cdot \mathbf{s}$ | 1.00 | $12.2900$ | $0.00\%$ | $11.85\%$ | $1.9998$ | $1.9\text{ ms}$ |
| **WGAN-GP (Box, best-of-10)** | $\epsilon = 2.0 \cdot \mathbf{s}$, retry | 10.00 | $12.2893$ | $0.00\%$ | $11.85\%$ | $1.9998$ | $149.7\text{ ms}$ |
| **WGAN-GP (Box, 10-step refine)** | $\epsilon$ per step, aggregate | 10.00 | $5.1945$ | $0.00\%$ | $11.85\%$ | $19.7303$ | $195.4\text{ ms}$ |
| **WGAN-GP (Manifold, trained)** | Unbudgeted learned target | 1.00 | **$0.0475$** | **$100.00\%$** | **$100.00\%$** | $35.3521$ | $1.8\text{ ms}$ |
| **WGAN-GP (Manifold, untrained)** | Same forward, random weights | 1.00 | $15.0501$ | $0.00\%$ | $11.85\%$ | $15.8726$ | $5.0\text{ ms}$ |

### Baseline Clean Detection Metrics (86,853 Packets Scored)
- **Calibrated Threshold ($\theta$):** $0.455986$ (99.9th percentile benign calibration split)
- **$\text{ROC-AUC}$:** $0.9418$
- **True Positive Rate ($\text{TPR}$ / Recall):** $88.13\%$
- **False Positive Rate ($\text{FPR}$):** $0.0961\%$ (out-of-sample benign validation split)
- **Precision:** $99.93\%$ | **$\text{F}_1$-Score:** $0.9366$ | **Overall Accuracy:** $92.64\%$
- **Benign kNN baseline (normalised $L_2$):** $0.1383$
- **Inference Throughput:** $28,000+$ pkts/sec on NVIDIA GTX 1660 SUPER

---

## 3. Core Insights & Thesis Findings

1. **Why Standard Gradient Attacks Stall on Volumetric NIDS Features:**
   In volumetric DoS attacks like Mirai, packet rates and byte counts reach $15\times$ to $35\times$ the maximum benign baseline. Because autoencoders unnormalize inputs with $\mathbf{s} = \mathbf{norm\_max} - \mathbf{norm\_min}$ and process them through bounded $\text{sigmoid}$ activation functions ($\sigma(z) \in (0, 1)$), extreme inputs push neurons into deep saturation. In this regime, gradients vanish ($\sigma'(z) \approx 0$). Bounded additive perturbations ($\epsilon \le 2.0$) can slightly attenuate moderate scanning attacks ($7.20\%$ ASR) but cannot bridge the volumetric flooding gap ($0.00\%$ ASR).

2. **Single-Pass Generation Beats Iteration; the Trained Generator Now Earns It:**
   While 50-step PGD (Manifold) achieves 100.00% evasion, it requires $1,668$–$2,199\text{ ms}$ per 1,000 packets because it performs 50 forward and backward autograd passes. A single generator pass takes $7.6\text{ ms}$. More importantly, removing the post-hoc clamp makes the learning meaningful: the untrained manifold generator scores 0% (RMSE 5.63–15.05), while the trained generator reaches 99.9–100% at **lower** RMSE than the naive clamp (0.0779 vs 0.114 scanning; 0.0475 vs 0.109 flood) and closer benign-neighbour distance (0.67/0.62 vs 0.91/0.87; benign baseline 0.14). The GAN is not essential for the result, but it now improves on the trivial projection.

3. **Budgeted Attacks Remain Weak:** At the genuine $\epsilon = 2.0$ box, PGD reaches 2.30% on scanning and 0.00% on flood; the WGAN reaches 1.60% and 0.00%. A best-of-10 retry loop does not change ASR because output variance is near zero. Ten feedback-refinement steps raise scanning to 3.80% and cut flood RMSE to 5.19, but aggregate L∞ reaches 10.4/19.7. The 100% figures require the unbudgeted manifold attack and should not be read as an $\epsilon = 2$ result.

4. **Mathematical Origin of Inactive Features ($L_0 \approx 53$):**
   All 100 features in AfterImage are continuous damped statistical summaries (weights, means, variances, radiuses, covariances, correlations). The ~47 features that remain unmodified during FGSM/PGD do so due to vanishing gradients from sigmoid saturation in the micro-autoencoders, combined with zero-activity channels clamped at $0.0$ by domain bounds.
