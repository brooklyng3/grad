# Phase 5 Comprehensive Technical Log: Unified Metrics, Reporting & End-to-End Pipeline CLI

This document provides an exhaustive, granular engineering record of the unified evaluation orchestrator, reporting suite, metric mathematical formulations, and experimental telemetry produced during Phase 5, incorporating the reviewer's corrections on subsample extrapolation, manifold envelope shaping, signed covariance preservation, and gradient saturation dynamics.

> [!WARNING]
> **Superseded results.** The original Phase 5 tables were measured on pre-fix
> artifacts: the WGAN had been trained at ε = 0.05 and evaluated at ε = 2.0, and
> its final "holdout" was in-sample. Phase 6 regenerated every measurement on a
> disjoint seeded split and added clamp-only, untrained-generator, DeepFool, and
> best-of-k rows. The tables in section 3 below are the post-fix values; the
> remediation log is `phase_6_detailed.md`. Latencies are single-run
> measurements and vary by roughly ±20%; the relative ordering is stable. The
> canonical table with the benign-kNN column is `reports/evaluation_table.md`.

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
│   ├── threshold_config.yaml -> theta = 0.455986
│   ├── mirai_features.npy (np.load, mmap_mode='r')
│   └── PipelineEvaluator (CUDA batch_size=256)
├── Mode: evaluate_clean
├── Mode: fgsm_sweep
├── Mode: pgd_attack
├── Mode: train_wgan (--manifold_shaping, --eps, --train_epochs)
└── Mode: evaluate_all (--attack_phase {scanning, flood, both}, --retries)
```

Phase 6 adds `--seed`, `--eps`, `--retries`, `--train_epochs`, and
`--deepfool_max_eps`; every evaluation mode is seeded and the training epsilon
is persisted in checkpoint metadata.

---

## 3. Experimental Telemetry & Verification Data

### 3.1. Clean Baseline Detection Telemetry (`evaluate_clean`)
- **Benign Validation Subset:** $33,310$ packets (odd-parity post-grace split, disjoint from calibration).
- **Malicious Evaluation Subset:** $53,543$ packets (sampled across all $642,516$ attack packets).
- **Execution Time:** $3.06$ seconds ($28,000+$ pkts/sec).
- **Confusion Matrix:**
  - $\text{TN} = 33,278$
  - $\text{FP} = 32$ ($\text{FPR} = 0.0961\%$, out-of-sample)
  - $\text{FN} = 6,358$ (Naturally evading baseline attack packets)
  - $\text{TP} = 47,185$ ($\text{TPR} = 88.13\%$)
- **Precision:** $99.93\%$ | **$\text{F}_1$-Score:** $0.9366$ | **$\text{ROC-AUC}$:** $0.9418$ | **Overall Accuracy:** $92.64\%$
- **Benign kNN baseline (normalised $L_2$):** $0.1383$

### 3.2. Single-Step FGSM Parameter Sweep (`fgsm_sweep`)
Evaluated across $1,000$ confirmed detected attack packets ($\text{RMSE}_{\text{orig}} = 13.4764$):

| $\epsilon$ Budget | $\text{ASR}_{\text{cond}}$ | $\text{ER}_{\text{total}}$ | Mean RMSE | Mean $L_\infty$ | Mean $L_2$ | Mean $L_1$ | Active Features ($L_0$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $0.01$ | $0.00\%$ | $11.85\%$ | $13.4703$ | $0.0100$ | $0.0974$ | $0.4612$ | $52.47$ |
| $0.05$ | $0.00\%$ | $11.85\%$ | $13.4459$ | $0.0500$ | $0.4853$ | $2.3060$ | $52.47$ |
| $0.10$ | $0.00\%$ | $11.85\%$ | $13.4157$ | $0.1000$ | $0.9689$ | $4.6120$ | $52.47$ |
| $0.25$ | $0.00\%$ | $11.85\%$ | $13.3258$ | $0.2500$ | $2.4102$ | $11.5300$ | $52.47$ |
| $0.50$ | $0.00\%$ | $11.85\%$ | $13.1758$ | $0.5000$ | $4.7998$ | $23.0600$ | $52.47$ |
| $1.00$ | $0.00\%$ | $11.85\%$ | $12.8904$ | $1.0000$ | $9.4348$ | $46.1200$ | $52.47$ |
| $2.00$ | $0.00\%$ | $11.85\%$ | $12.4483$ | $2.0000$ | $18.7531$ | $92.2400$ | $52.47$ |

*Scientific Clarification on Inactive Features ($L_0 \approx 52.68$):*
All 100 features in AfterImage are continuous statistical aggregations (1D/2D damped window weights, means, variances, radiuses, covariances, correlations). There are no static packet header or port fields in the 100D vector. The ~47 features that remain unmodified during FGSM sweeps do so because:
1. **Sigmoid Saturation & Vanishing Gradients:** Under volumetric flooding, normalized features reach up to $35\times$ baseline. The micro-autoencoder sigmoid neurons saturate ($\sigma(z) \to 1.0$), resulting in vanishing gradients $\frac{\partial \text{RMSE}}{\partial x_j} \approx 0$ ($|\nabla_{x_j}| < 10^{-4}$).
2. **Zero-Activity Boundary Clamping:** Inactive 2D host-pair cross-statistics sit at zero ($x_j = 0.0$). Any gradient step attempting to step into negative territory is instantly clamped back to $0.0$ by `enforce_feature_bounds`.

### 3.3. Multi-Paradigm Comparative Benchmark (`evaluate_all`)
Evaluated across disjoint holdouts in both the Reconnaissance / Scanning and Volumetric Flooding regimes, $1,000$ confirmed detected packets each, seeded (42), with a shared $\epsilon = 2.0$ budget and a 10-draw cap:

> [!WARNING]
> **Corrected by Phase 6.** The tables below are the post-fix benchmark with
> disjoint holdout splits, seeded runs, and the shared ε = 2.0 budget. The
> original Phase 5 tables were measured on the pre-fix artifacts and are
> superseded. Rows labelled **unbudgeted** are not comparable to ε = 2.0 rows
> without reading the $L_\infty$ column.

#### Phase A: Reconnaissance / Scanning Phase (Baseline $\text{RMSE} = 5.3381$)
| Paradigm / Attack | Perturbation Constraint | Draws/pkt | Mean Attack RMSE | $\text{ASR}_{\text{cond}}$ | $\text{ER}_{\text{total}}$ | Mean $L_\infty$ | Latency (ms / 1k pkts) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Clean Baseline** | None | 0.00 | $5.3381$ | $0.00\%$ | $11.85\%$ | $0.0000$ | $0.0\text{ ms}$ |
| **Envelope Clamp Only** | Unbudgeted $[\min,\max]$ | 1.00 | **$0.1136$** | **$100.00\%$** | **$100.00\%$** | $12.9735$ | $2.4\text{ ms}$ |
| **Single-Step FGSM** | $\epsilon = 0.05 \cdot \mathbf{s}$ | 1.00 | $5.3076$ | $0.00\%$ | $11.85\%$ | $0.0500$ | $50.7\text{ ms}$ |
| **Single-Step FGSM** | $\epsilon = 0.25 \cdot \mathbf{s}$ | 1.00 | $5.1910$ | $0.00\%$ | $11.85\%$ | $0.2500$ | $53.2\text{ ms}$ |
| **Multi-Step PGD (Box)** | 50 steps, $\epsilon = 2.0 \cdot \mathbf{s}$ | 1.00 | $4.1601$ | $2.30\%$ | $13.88\%$ | $1.9991$ | $2321.2\text{ ms}$ |
| **DeepFool (bounded)** | 50 iters, $\epsilon \le 2.0 \cdot \mathbf{s}$ | 1.00 | $4.5108$ | $0.00\%$ | $11.85\%$ | $1.9683$ | $2387.1\text{ ms}$ |
| **Multi-Step PGD (Manifold)** | Unbudgeted $[\min,\max]$, 50 steps | 1.00 | **$0.0909$** | **$100.00\%$** | **$100.00\%$** | $13.0325$ | $2170.6\text{ ms}$ |
| **WGAN-GP (Box, 1 draw)** | $\epsilon = 2.0 \cdot \mathbf{s}$ | 1.00 | $4.1836$ | $1.90\%$ | $13.52\%$ | $1.9990$ | $19.2\text{ ms}$ |
| **WGAN-GP (Box, best-of-10)** | $\epsilon = 2.0 \cdot \mathbf{s}$, retry | 9.83 | $4.1807$ | $1.90\%$ | $13.52\%$ | $1.9991$ | $153.5\text{ ms}$ |
| **WGAN-GP (Box, 10-step refine)** | $\epsilon$ per step, aggregate | 9.67 | $1.1778$ | $3.90\%$ | $15.29\%$ | $11.4744$ | $148.7\text{ ms}$ |
| **WGAN-GP (Manifold, trained)** | Unbudgeted learned target | 1.00 | **$0.0782$** | **$99.90\%$** | **$99.91\%$** | $13.8111$ | $7.6\text{ ms}$ |
| **WGAN-GP (Manifold, untrained)** | Same forward, random weights | 1.00 | $5.6284$ | $0.00\%$ | $11.85\%$ | $6.1065$ | $3.0\text{ ms}$ |

#### Phase B: Volumetric Flooding Phase (Baseline $\text{RMSE} = 13.4764$)
| Paradigm / Attack | Perturbation Constraint | Draws/pkt | Mean Attack RMSE | $\text{ASR}_{\text{cond}}$ | $\text{ER}_{\text{total}}$ | Mean $L_\infty$ | Latency (ms / 1k pkts) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Clean Baseline** | None | 0.00 | $13.4764$ | $0.00\%$ | $11.85\%$ | $0.0000$ | $0.0\text{ ms}$ |
| **Envelope Clamp Only** | Unbudgeted $[\min,\max]$ | 1.00 | **$0.1092$** | **$100.00\%$** | **$100.00\%$** | $34.9663$ | $2.1\text{ ms}$ |
| **Single-Step FGSM** | $\epsilon = 0.05 \cdot \mathbf{s}$ | 1.00 | $13.4459$ | $0.00\%$ | $11.85\%$ | $0.0500$ | $50.4\text{ ms}$ |
| **Single-Step FGSM** | $\epsilon = 0.25 \cdot \mathbf{s}$ | 1.00 | $13.3258$ | $0.00\%$ | $11.85\%$ | $0.2500$ | $45.5\text{ ms}$ |
| **Multi-Step PGD (Box)** | 50 steps, $\epsilon = 2.0 \cdot \mathbf{s}$ | 1.00 | $12.2761$ | $0.00\%$ | $11.85\%$ | $2.0000$ | $2274.2\text{ ms}$ |
| **DeepFool (bounded)** | 50 iters, $\epsilon \le 2.0 \cdot \mathbf{s}$ | 1.00 | $12.9098$ | $0.00\%$ | $11.85\%$ | $2.0000$ | $2135.0\text{ ms}$ |
| **Multi-Step PGD (Manifold)** | Unbudgeted $[\min,\max]$, 50 steps | 1.00 | **$0.0896$** | **$100.00\%$** | **$100.00\%$** | $34.9754$ | $1668.1\text{ ms}$ |
| **WGAN-GP (Box, 1 draw)** | $\epsilon = 2.0 \cdot \mathbf{s}$ | 1.00 | $12.2905$ | $0.00\%$ | $11.85\%$ | $1.9999$ | $1.9\text{ ms}$ |
| **WGAN-GP (Box, best-of-10)** | $\epsilon = 2.0 \cdot \mathbf{s}$, retry | 10.00 | $12.2890$ | $0.00\%$ | $11.85\%$ | $1.9999$ | $149.7\text{ ms}$ |
| **WGAN-GP (Box, 10-step refine)** | $\epsilon$ per step, aggregate | 10.00 | $5.3781$ | $0.00\%$ | $11.85\%$ | $19.8139$ | $195.4\text{ ms}$ |
| **WGAN-GP (Manifold, trained)** | Unbudgeted learned target | 1.00 | **$0.0367$** | **$100.00\%$** | **$100.00\%$** | $35.6243$ | $1.8\text{ ms}$ |
| **WGAN-GP (Manifold, untrained)** | Same forward, random weights | 1.00 | $15.0501$ | $0.00\%$ | $11.85\%$ | $15.8726$ | $5.0\text{ ms}$ |

*Interpretation (post-fix):*
The untrained manifold generator scores 0% (RMSE 5.63–15.05), while the trained
generator reaches 99.9–100% at **lower** RMSE than the naive clamp (0.078 vs
0.114 scanning; 0.037 vs 0.109 flood) and closer benign-neighbour distance
(0.86/0.62 vs 0.91/0.87; benign baseline 0.14). The learned network is
therefore doing real work. The clamp remains faster and is a useful lower
bound.

Best-of-k leaves both ASR rows unchanged (conditional mode collapse). Ten
feedback-refinement steps raise scanning to 3.90% and cut flood RMSE to 5.38,
but total L∞ grows to 11.5/19.8, so it is an aggregate-budget result. The full
table with kNN distances is in `reports/evaluation_table.md`.

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
4. **MTU and IAT bounds (Phase 6):** Packet-size mean channels (indices 1, 4, 7, 10, 13, 16, 23, 30, 37, 44, 66, 73, 80, 87, 94) are clamped to $[0, 1500]$ bytes; jitter mean/std channels are clamped to $\ge 0$. TTL is not present in AfterImage and is therefore not enforceable at this layer.

---

## 5. Artifact Manifest

| Path | Description |
| :--- | :--- |
| `src/utils/metrics.py` | Anomaly detection metrics, ROC-AUC, distortion aggregation, and report exporters. |
| `src/utils/data_splits.py` | Deterministic disjoint attack and benign splits (Phase 6). |
| `src/utils/seed.py` | Central RNG seeding for reproducible runs (Phase 6). |
| `src/attacker/generator.py` | `PolymorphicGenerator` supporting both additive box and manifold envelope shaping modes. |
| `src/pipeline/feedback_loop.py` | Closed feedback loop with signed covariance preservation and manifold training mode. |
| `src/pipeline/evaluator.py` | Evaluator engine with corrected population extrapolation for subsample evaluation. |
| `run_pipeline.py` | Unified CLI orchestrator supporting scanning and flood attack regimes. |
| `data/attack_samples/wgan_manifold_generator.pt` | Trained manifold generator checkpoint (99.9–100% ASR; unclamped learned target). |
| `data/attack_samples/wgan_generator.pt` | Trained additive-box WGAN-GP generator checkpoint. |
| `data/attack_samples/*.meta.json` | Persisted training epsilon, seed, mode, and best-epoch metadata (Phase 6). |
| `reports/wgan_training_summary.json` | Additive WGAN training history and holdout summary. |
| `reports/wgan_manifold_summary.json` | Manifold WGAN training history and holdout summary. |
| `README.md` | Setup, protocol, and reproduction instructions. |
| `docs/overview/phase_6_overview.md` | Remediation summary and post-fix headline results. |
| `docs/detailed/phase_6_detailed.md` | Remediation log and post-fix benchmark. |
| `reports/evaluation_table.md` | Dual-regime comparative benchmark tables (Scanning & Volumetric Flood). |
| `reports/clean_detection_metrics.json` | Baseline detection telemetry (TPR, FPR, F1, ROC-AUC). |
| `reports/pgd_benchmark.json` | Multi-step PGD and manifold traffic-shaping telemetry. |
| `docs/overview/phase_5_overview.md` | Executive overview of Phase 5. |
| `docs/detailed/phase_5_detailed.md` | Exhaustive technical documentation of Phase 5. |
