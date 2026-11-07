# Phase 5 Comprehensive Technical Log: Unified Metrics, Reporting & End-to-End Pipeline CLI

This document provides an exhaustive, granular engineering record of the unified evaluation orchestrator, reporting suite, metric mathematical formulations, and experimental telemetry produced during Phase 5, incorporating the reviewer's corrections on subsample extrapolation, manifold envelope shaping, signed covariance preservation, and gradient saturation dynamics.

---

## 1. Mathematical Formulations (`src/utils/metrics.py` & `src/pipeline/evaluator.py`)

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

### 1.3. Decoupled Evasion Metrics & Subsample Extrapolation
To prevent metric distortion caused by malicious packets that naturally resemble benign traffic, evasion metrics are decoupled:

1. **Conditional Adversarial Success Rate ($\text{ASR}_{\text{cond}}$):**
   Evaluated strictly over the subset of $N_{\text{eval}}$ packets that Kitsune actively detected ($\text{RMSE}_{\text{orig}} > \theta$):
   $$\text{ASR}_{\text{cond}} = \frac{\sum_{i=1}^{N_{\text{eval}}} \mathbb{I}(\text{RMSE}(\mathbf{x}_{\text{adv}}^{(i)}) \le \theta)}{N_{\text{eval}}} \times 100\%$$

2. **Overall Evasion Rate ($\text{ER}_{\text{total}}$):**
   When evaluating on a representative holdout subsample of detected traffic ($N_{\text{eval}} \le |\mathcal{M}_{\text{det}}|$), $\text{ER}_{\text{total}}$ extrapolates the sample success proportion across the full detected population $|\mathcal{M}_{\text{det}}| = 566,393$:
   $$\text{ER}_{\text{total}} = \frac{N_{\text{pre-evaded}} + |\mathcal{M}_{\text{det}}| \cdot \left(\frac{\text{ASR}_{\text{cond}}}{100}\right)}{N_{\text{total\_attacks}}} \times 100\%$$

Where:
- $|\mathcal{M}_{\text{det}}| = 566,393$ detected Mirai packets.
- $N_{\text{pre-evaded}} = 76,123$ naturally evading baseline attack packets.
- $N_{\text{total\_attacks}} = 642,516$ total attack population.

*Properties of the Extrapolation Formula:*
- When $\text{ASR}_{\text{cond}} = 100\%$, $\text{ER}_{\text{total}} = \frac{76,123 + 566,393}{642,516} \times 100\% = \mathbf{100.00\%}$.
- When $\text{ASR}_{\text{cond}} = 0\%$, $\text{ER}_{\text{total}} = \frac{76,123 + 0}{642,516} \times 100\% = \mathbf{11.85\%}$.
- When evaluating the complete dataset ($N_{\text{eval}} = |\mathcal{M}_{\text{det}}|$), the formula reduces identically to direct enumeration.

---

## 2. CLI Architecture & Execution Modes (`run_pipeline.py`)

The root script `run_pipeline.py` integrates all defensive and offensive subsystems:

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
├── Mode: train_wgan (--manifold_shaping)
└── Mode: evaluate_all (--attack_phase {scanning, flood, both})
```

---

## 3. Experimental Telemetry & Verification Data

### 3.1. Clean Baseline Detection Telemetry (`evaluate_clean`)
- **Benign Validation Subset:** $66,621$ packets (post-grace period: indices $55,000$ to $121,621$).
- **Malicious Evaluation Subset:** $53,543$ packets (sampled across all $642,516$ attack packets).
- **Execution Time:** $4.88$ seconds ($24,000+$ pkts/sec).
- **Confusion Matrix:**
  - $\text{TN} = 66,554$
  - $\text{FP} = 67$ ($\text{FPR} = 0.1006\%$, verifying the calibrated 99.9th percentile threshold)
  - $\text{FN} = 6,358$ (Naturally evading baseline attack packets)
  - $\text{TP} = 47,185$ ($\text{TPR} = 88.13\%$)
- **Precision:** $99.86\%$ | **$\text{F}_1$-Score:** $0.9363$ | **$\text{ROC-AUC}$:** $0.9418$ | **Overall Accuracy:** $94.65\%$

### 3.2. Single-Step FGSM Parameter Sweep (`fgsm_sweep`)
Evaluated across $1,000$ confirmed detected attack packets ($\text{RMSE}_{\text{orig}} = 13.5061$):

| $\epsilon$ Budget | $\text{ASR}_{\text{cond}}$ | $\text{ER}_{\text{total}}$ | Mean RMSE | Mean $L_\infty$ | Mean $L_2$ | Mean $L_1$ | Active Features ($L_0$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $0.01$ | $0.00\%$ | $11.85\%$ | $13.5000$ | $0.0100$ | $0.0974$ | $0.4612$ | $52.68$ |
| $0.05$ | $0.00\%$ | $11.85\%$ | $13.4756$ | $0.0500$ | $0.4851$ | $2.3060$ | $52.68$ |
| $0.10$ | $0.00\%$ | $11.85\%$ | $13.4454$ | $0.1000$ | $0.9683$ | $4.6120$ | $52.68$ |
| $0.25$ | $0.00\%$ | $11.85\%$ | $13.3556$ | $0.2500$ | $2.4088$ | $11.5300$ | $52.68$ |
| $0.50$ | $0.00\%$ | $11.85\%$ | $13.2057$ | $0.5000$ | $4.7968$ | $23.0600$ | $52.68$ |
| $1.00$ | $0.00\%$ | $11.85\%$ | $12.9207$ | $1.0000$ | $9.4291$ | $46.1200$ | $52.68$ |
| $2.00$ | $0.00\%$ | $11.85\%$ | $12.4824$ | $2.0000$ | $18.7428$ | $92.2400$ | $52.68$ |

*Scientific Clarification on Inactive Features ($L_0 \approx 52.68$):*
All 100 features in AfterImage are continuous statistical aggregations (1D/2D damped window weights, means, variances, radiuses, covariances, correlations). There are no static packet header or port fields in the 100D vector. The ~47 features that remain unmodified during FGSM sweeps do so because:
1. **Sigmoid Saturation & Vanishing Gradients:** Under volumetric flooding, normalized features reach up to $35\times$ baseline. The micro-autoencoder sigmoid neurons saturate ($\sigma(z) \to 1.0$), resulting in vanishing gradients $\frac{\partial \text{RMSE}}{\partial x_j} \approx 0$ ($|\nabla_{x_j}| < 10^{-4}$).
2. **Zero-Activity Boundary Clamping:** Inactive 2D host-pair cross-statistics sit at zero ($x_j = 0.0$). Any gradient step attempting to step into negative territory is instantly clamped back to $0.0$ by `enforce_feature_bounds`.

### 3.3. Multi-Paradigm Comparative Benchmark (`evaluate_all`)
Evaluated across both the Reconnaissance / Scanning and Volumetric Flooding regimes on $1,000$ confirmed detected packets each:

#### Phase A: Reconnaissance / Scanning Phase (Baseline $\text{RMSE} = 3.7961$)
| Paradigm / Attack | Perturbation Constraint | Mean Attack RMSE | Conditional ASR ($\text{ASR}_{\text{cond}}$) | Overall Evasion Rate ($\text{ER}_{\text{total}}$) | Mean $L_\infty$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Clean Baseline** | None (Unperturbed) | $3.7961$ | $0.00\%$ | $11.85\%$ | $0.0000$ |
| **Single-Step FGSM** | $\epsilon = 0.05 \cdot \mathbf{s}$ | $3.7664$ | $0.00\%$ | $11.85\%$ | $0.0500$ |
| **Single-Step FGSM** | $\epsilon = 0.25 \cdot \mathbf{s}$ | $3.6568$ | $0.10\%$ | $11.94\%$ | $0.2500$ |
| **Multi-Step PGD (Box)** | 50 steps, $\epsilon = 2.0 \cdot \mathbf{s}$ | $2.6743$ | $7.20\%$ | $18.19\%$ | $1.9968$ |
| **Multi-Step PGD (Manifold)** | 50 steps, $[\mathbf{min}, \mathbf{max}]$ | **$0.0845$** | **$100.00\%$** | **$100.00\%$** | $8.8060$ |
| **Polymorphic WGAN-GP (Box)** | $\epsilon = 2.0 \cdot \mathbf{s}$, closed loop | $2.9505$ | $0.00\%$ | $11.85\%$ | $1.9999$ |
| **Polymorphic WGAN-GP (Manifold)** | Envelope Shaping, single pass | **$0.2276$** | **$100.00\%$** | **$100.00\%$** | $9.5686$ |

#### Phase B: Volumetric Flooding Phase (Baseline $\text{RMSE} = 13.5061$)
| Paradigm / Attack | Perturbation Constraint | Mean Attack RMSE | Conditional ASR ($\text{ASR}_{\text{cond}}$) | Overall Evasion Rate ($\text{ER}_{\text{total}}$) | Mean $L_\infty$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Clean Baseline** | None (Unperturbed) | $13.5061$ | $0.00\%$ | $11.85\%$ | $0.0000$ |
| **Single-Step FGSM** | $\epsilon = 0.05 \cdot \mathbf{s}$ | $13.4756$ | $0.00\%$ | $11.85\%$ | $0.0500$ |
| **Single-Step FGSM** | $\epsilon = 0.25 \cdot \mathbf{s}$ | $13.3556$ | $0.00\%$ | $11.85\%$ | $0.2500$ |
| **Multi-Step PGD (Box)** | 50 steps, $\epsilon = 2.0 \cdot \mathbf{s}$ | $12.3060$ | $0.00\%$ | $11.85\%$ | $2.0000$ |
| **Multi-Step PGD (Manifold)** | 50 steps, $[\mathbf{min}, \mathbf{max}]$ | **$0.0899$** | **$100.00\%$** | **$100.00\%$** | $35.0449$ |
| **Polymorphic WGAN-GP (Box)** | $\epsilon = 2.0 \cdot \mathbf{s}$, closed loop | $12.4103$ | $0.00\%$ | $11.85\%$ | $1.9998$ |
| **Polymorphic WGAN-GP (Manifold)** | Envelope Shaping, single pass | **$0.2285$** | **$100.00\%$** | **$100.00\%$** | $35.0841$ |

---

## 4. Preservation of Signed Statistical Invariants

In accordance with Section 5 of `AGENTS.md`, all adversarial generator and feedback loop unnormalization operations strictly employ `enforce_feature_bounds` from `src/utils/constraints.py` rather than naive zero-clamping:

```python
x_adv_raw = enforce_feature_bounds(x_adv_norm * scales + norm_min)
```

This guarantees:
1. **Unsigned Features ($\ge 0.0$):** Packet rate weights, means, variances, radiuses, and magnitudes cannot step negative.
2. **Covariance Features (Signed in $(-\infty, \infty)$):** The 10 Covariance features (indices 20, 27, 34, 41, 48, 70, 77, 84, 91, 98) preserve valid negative cross-covariance statistics between source-destination IP and MAC pairs.
3. **Correlation Features ($[-1.0, 1.0]$):** The 10 Pearson Correlation Coefficient (PCC) features (indices 21, 28, 35, 42, 49, 71, 78, 85, 92, 99) are strictly bounded within valid mathematical limits.

---

## 5. Artifact Manifest

| Path | Description |
| :--- | :--- |
| `src/utils/metrics.py` | Anomaly detection metrics, ROC-AUC, distortion aggregation, and report exporters. |
| `src/attacker/generator.py` | `PolymorphicGenerator` supporting both additive box and manifold envelope shaping modes. |
| `src/pipeline/feedback_loop.py` | Closed feedback loop with signed covariance preservation and manifold training mode. |
| `src/pipeline/evaluator.py` | Evaluator engine with corrected population extrapolation for subsample evaluation. |
| `run_pipeline.py` | Unified CLI orchestrator supporting scanning and flood attack regimes. |
| `data/attack_samples/wgan_manifold_generator.pt` | Trained Manifold-Shaping WGAN-GP generator checkpoint (100% ASR). |
| `data/attack_samples/wgan_generator.pt` | Trained Additive Box WGAN-GP generator checkpoint. |
| `reports/evaluation_table.md` | Dual-regime comparative benchmark tables (Scanning & Volumetric Flood). |
| `reports/clean_detection_metrics.json` | Baseline detection telemetry (TPR, FPR, F1, ROC-AUC). |
| `reports/pgd_benchmark.json` | Multi-step PGD and manifold traffic-shaping telemetry. |
| `docs/overview/phase_5_overview.md` | Executive overview of Phase 5. |
| `docs/detailed/phase_5_detailed.md` | Exhaustive technical documentation of Phase 5. |
