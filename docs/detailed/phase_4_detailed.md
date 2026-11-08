# Phase 4 Comprehensive Technical Log: Automated Evaluation Engine & Closed Feedback Optimization Loop

This document provides an exhaustive, granular engineering record of the closed-loop adversarial feedback training, the composite hinge evasion loss formulation, evaluation metric decoupling, and the comparative experimental benchmarks executed during Phase 4.

---

## 1. System Environment & Execution Baseline
- **Execution Modules:**
  - `src/pipeline/evaluator.py`: `PipelineEvaluator` class.
  - `src/pipeline/feedback_loop.py`: `AdversarialFeedbackLoop` class.
- **Pre-trained Defender:** `data/processed/kitsune_mirai_model.pkl` loaded via `PyTorchKitNET`.
- **Pre-computed Threshold:** $\theta = 0.453419$ in the original run (0.455986 after Phase 6 recalibration) from `data/processed/threshold_config.yaml`.
- **Training Pools (as originally run):**
  - Benign Pool: $10,000$ post-convergence benign samples ($i \ge 55,000$).
  - Malicious Pool: $10,000$ strictly detected attack samples ($\text{RMSE}_{\text{orig}} > \theta$).
- **Compute Hardware:** Intel Core i5-9400F, 16 GB DDR4, NVIDIA GeForce GTX 1660 SUPER (CUDA).

> [!WARNING]
> **Historical caveats corrected in Phase 6.** The original run trained at
> `eps_linf = 0.05` but evaluated at $\epsilon = 2.0$. The final evaluation
> sliced `x_mal_raw[:1000]`, which are training samples, despite being labelled
> a holdout. The malicious pool was the first 20,000 attacks (3% of the
> population, all scanning). Phase 6 fixed all three: shared persisted ε,
> disjoint modulo split, stratified sampling, holdout model selection, and
> `n_critic` execution. See `phase_6_detailed.md` and section 6 below.

---

## 2. Evaluation Engine Architecture (`src/pipeline/evaluator.py`)

To ensure scientific reproducibility and guard against artificial metric inflation, `PipelineEvaluator` implements three core functions:

### 2.1. Metric Decoupling Formulae
1. **Conditional Adversarial Success Rate ($\text{ASR}_{\text{cond}}$):**
   Measures the attacker's ability to mutate packets that were actively flagged by Kitsune:
   $$\text{ASR}_{\text{cond}} = \frac{\sum_{i=1}^{N_{\text{det}}} \mathbb{I}(\text{RMSE}(\mathbf{x}_{\text{adv}}^{(i)}) \le \theta)}{N_{\text{det}}} \times 100\%$$
2. **Overall Evasion Rate ($\text{ER}_{\text{total}}$):**
   Measures total evasion across the entire attack population (including the $76,123$ pre-evading packets), properly extrapolating sample success proportions when evaluated on subsamples $N_{\text{eval}} \le |\mathcal{M}_{\text{det}}|$:
   $$\text{ER}_{\text{total}} = \frac{N_{\text{pre-evaded}} + |\mathcal{M}_{\text{det}}| \cdot \left(\frac{\text{ASR}_{\text{cond}}}{100}\right)}{N_{\text{total\_attacks}}} \times 100\%$$
   Where $N_{\text{pre-evaded}} = 76,123$, $|\mathcal{M}_{\text{det}}| = 566,393$, and $N_{\text{total\_attacks}} = 642,516$. When evaluating the complete attack population ($N_{\text{eval}} = |\mathcal{M}_{\text{det}}|$), this reduces to the exact count sum.

### 2.2. Normalized Perturbation Distances
To quantify the distortion introduced by the attacker relative to physical feature scales:
$$\tilde{\delta}_j = \frac{\mathbf{x}_{\text{adv}, j} - \mathbf{x}_{\text{orig}, j}}{\mathbf{s}_{\text{global}, j}}$$
- **$L_\infty$ Distance:** $\max_j |\tilde{\delta}_j|$ (worst-case individual feature drift).
- **$L_2$ Distance:** $\|\tilde{\delta}\|_2$ (total perturbation energy).
- **$L_1$ Distance:** $\|\tilde{\delta}\|_1$ (cumulative feature distortion).
- **$L_0$ Distance:** $\sum_j \mathbb{I}(|\mathbf{x}_{\text{adv}, j} - \mathbf{x}_{\text{orig}, j}| > 10^{-5})$ (number of modified features).

---

## 3. Closed Feedback Optimization Mechanics (`src/pipeline/feedback_loop.py`)

### 3.1. Critic Optimization Step
At each step, the Critic is trained on normalized benign and generated samples:
$$\mathcal{L}_C = \mathbb{E}\left[D(\tilde{\mathbf{x}}_{\text{fake}})\right] - \mathbb{E}\left[D(\tilde{\mathbf{x}}_{\text{real}})\right] + \lambda_{GP} \cdot \mathbb{E}_{\hat{\mathbf{x}}}\left[ (\|\nabla_{\hat{\mathbf{x}}} D(\hat{\mathbf{x}})\|_2 - 1)^2 \right]$$

### 3.2. Generator Optimization Step (Composite Objective)
The Generator is trained using an end-to-end differentiable loss combining realism, evasion, and perturbation budget:
$$\mathcal{L}_G = -\mathbb{E}\left[D(\tilde{\mathbf{x}}_{\text{adv}})\right] + \lambda_{\text{evade}} \cdot \mathbb{E}\left[\max(\text{RMSE}(\mathbf{x}_{\text{adv}}) - \theta, 0.0)\right] + \lambda_{\text{dist}} \cdot \|\tilde{\mathbf{x}}_{\text{adv}} - \tilde{\mathbf{x}}_{\text{mal}}\|_2$$

Where:
- $\tilde{\mathbf{x}}_{\text{adv}} = \tilde{\mathbf{x}}_{\text{mal}} + \tanh(\text{MLP}(\mathbf{z}, \tilde{\mathbf{x}}_{\text{mal}})) \cdot \epsilon$
- $\mathbf{x}_{\text{adv}} = \text{enforce\_feature\_bounds}(\tilde{\mathbf{x}}_{\text{adv}} \odot \mathbf{s}_{\text{global}} + \mathbf{norm\_min}_{\text{global}})$ (preserving signed Covariances and $[-1, 1]$ PCCs)
- Autograd flows gradients from `PyTorchKitNET`'s 16 micro-autoencoders directly into the Generator weights via $\frac{\partial \text{RMSE}}{\partial \mathbf{x}_{\text{adv}}} \cdot \frac{\partial \mathbf{x}_{\text{adv}}}{\partial \theta_G}$.

---

## 4. Experimental Benchmarks & Convergence Telemetry

### 4.1. Training Telemetry
- **Hardware Profile:** CUDA batch size = 128, VRAM $< 1.8$ GB.
- **Runtime:** Completed 15 training epochs over 10,000 samples in **63.47 seconds**.
- **Critic Convergence:** $\mathcal{L}_C$ converged from $-2.41$ to $-23.27$.
- **Generator Evasion (in-sample batch ASR):** Conditional ASR on the batches being trained climbed from $0.20\%$ to $6.70\%$ under $\epsilon = 2.0$ (15 epochs at the time; the CLI now defaults to 10).

### 4.2. Comparative Robustness Matrix (Original run: 1,000 **in-sample** detected attacks)

> [!NOTE]
> **Historical sampling scope (pre-fix):** the 1,000 detected attack samples were drawn from the initial attack onset, and — as Phase 6 discovered — those packets were also inside the WGAN training pool. The corrected protocol uses a disjoint modulo split; see section 6 below and `phase_6_detailed.md`.

| Method | Perturbation Paradigm | Mean Attack RMSE | $\text{ASR}_{\text{cond}}$ | $\text{ER}_{\text{total}}$ | Mean $L_\infty$ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Unperturbed Clean** | None | $1.6798$ | $0.00\%$ | $11.85\%$ | $0.000$ |
| **FGSM ($\epsilon=0.05$)** | Single-Step | $2.9710$ | $0.00\%$ | $11.85\%$ | $0.050$ |
| **FGSM ($\epsilon=0.25$)** | Single-Step | $2.8681$ | $0.20\%$ | $12.02\%$ | $0.250$ |
| **WGAN-GP ($\epsilon=2.0$, in-sample)**| Generative Hybrid | $2.6991$ | **$6.70\%$** | **$17.75\%$** | $1.998$ |
| **PGD ($\epsilon=2.0$)** | 50-Step Iterative | $0.8243$ | **$37.00\%$** | **$44.46\%$** | $2.000$ |
| **Manifold Traffic-Shaping**| Rate-Paced Projection | **$0.1085$** | **$100.00\%$** | **$100.00\%$** | $8.945$ |

---

## 5. Persistence & Verification

### 5.1. Model Checkpoints Generated
- `data/attack_samples/wgan_generator.pt` (~480 KB PyTorch state dict).
- `data/attack_samples/wgan_critic.pt` (~450 KB PyTorch state dict).

### 5.2. Automated Test Suite (`tests/test_pipeline.py`)
- `test_evaluator_score_batches`: Validates GPU batching and non-negative RMSE outputs.
- `test_evaluator_evasion_metrics`: Confirms distance norm calculations ($L_0, L_1, L_2, L_\infty$) and metric isolation.
- `test_feedback_loop_initialization`: Validates parameter parsing and CUDA device binding.
- **Full Suite Status:** 17 / 17 tests passing at the time; 36 / 36 after Phase 6.

---

## 6. Phase 6 Corrected Results (disjoint holdout, ε = 2.0)

The additive generator was retrained with the corrected protocol. Best epoch was
selected by held-out ASR; training and evaluation both used ε = 2.0.

| Test | Conditional ASR |
|---|---:|
| Original saved checkpoint, any ε including training window | 0.00% |
| Corrected training, final disjoint holdout (spread over all attacks) | 0.10% |
| Corrected generator, Phase A scanning holdout | **1.60%** |
| Corrected generator, Phase B flood holdout | **0.00%** |
| Corrected generator, scanning holdout, best-of-10 draws | **1.60%** (9.85 draws/pkt) |
| Corrected generator, scanning holdout, 10-step refine | **3.80%** (aggregate L∞ 10.43) |

The budgeted WGAN is therefore no longer non-functional, but it remains far
weaker than the unbudgeted manifold attack (100%). After removing the post-hoc
clamp (second review round), the **trained** manifold generator beats the naive
clamp on RMSE (0.078 vs 0.114 scanning; 0.037 vs 0.109 flood), while the
untrained generator scores 0%. See `phase_6_detailed.md` section 15.
