# Phase 3 Comprehensive Technical Log: Offensive Attacker Engine & Mathematical Perturbation Mechanics

This document provides an exhaustive, granular engineering record of the mathematical formulation, normalized latent generation, Lipschitz gradient penalties, autograd descent mechanisms, and baseline evasion sweeps developed during Phase 3.

---

## 1. System Environment & Module Architecture
- **Directory:** `src/attacker/`
- **Files Created:**
  - `src/attacker/perturbations.py`: FGSM, DeepFool, and epsilon sweep routines.
  - `src/attacker/generator.py`: `PolymorphicGenerator(nn.Module)`.
  - `src/attacker/discriminator.py`: `WGANDiscriminator(nn.Module)` and `compute_gradient_penalty()`.
  - `tests/test_attacker.py`: Pytest verification fixtures and test cases.
- **Reference Models & Data:**
  - `PyTorchKitNET` loaded from `data/processed/kitsune_mirai_model.pkl`.
  - Threshold $\theta = 0.455986$ from `data/processed/threshold_config.yaml` (after Phase 6 recalibration).
  - Dynamic scale vector $\mathbf{s}_{\text{global}} \in \mathbb{R}^{100}$ from `get_global_feature_bounds()`.

---

## 2. Gradient-Based Perturbation Mechanics (`src/attacker/perturbations.py`)

### 2.1. Inverted FGSM for Autoencoders
In standard adversarial classification (e.g. Goodfellow et al.), an attacker *adds* the sign of the loss gradient to maximize classification error:
$$\mathbf{x}_{\text{adv}} = \mathbf{x} + \epsilon \cdot \text{sign}(\nabla_{\mathbf{x}} \mathcal{L}_{\text{CE}}(f(\mathbf{x}), y))$$

In Network Intrusion Detection using Autoencoders (Kitsune), an alert is triggered when the reconstruction error exceeds the anomaly threshold ($\text{RMSE}(\mathbf{x}) > \theta$). Therefore, the offensive objective is to **minimize** the reconstruction error:
$$\min_{\delta} \quad \text{RMSE}(\mathbf{x} + \delta) \quad \text{subject to } \|\delta\|_{\infty} \le \epsilon \cdot \mathbf{s}_{\text{global}}$$

Applying first-order Taylor expansion around $\mathbf{x}$:
$$\text{RMSE}(\mathbf{x} + \delta) \approx \text{RMSE}(\mathbf{x}) + \nabla_{\mathbf{x}} \text{RMSE}(\mathbf{x})^T \delta$$
To minimize this quantity under the $L_\infty$ box constraint:
$$\delta^* = -\epsilon \cdot \mathbf{s}_{\text{global}} \odot \text{sign}(\nabla_{\mathbf{x}} \text{RMSE}(\mathbf{x}))$$
$$\mathbf{x}_{\text{adv}} = \text{project}_{\mathcal{C}}(\mathbf{x} + \delta^*)$$
where $\text{project}_{\mathcal{C}}$ applies physical domain constraints (signed Covariance preserved, PCC $\in [-1, 1]$, unsigned features $\ge 0.0$).

### 2.2. Multi-Step Projected Gradient Descent (PGD) & Manifold Shaping
To resolve the single-step overshooting and saturated sigmoid gradient issues of FGSM on extreme flood packets:
1. **Iterative PGD with Step Size $\alpha = \epsilon / 10$:**
   $$\mathbf{x}_{t+1} = \text{project}_{L_\infty(\epsilon)}\left(\mathbf{x}_t - \alpha \cdot \mathbf{s}_{\text{global}} \odot \text{sign}(\nabla_{\mathbf{x}} \text{RMSE}(\mathbf{x}_t))\right)$$
   Smoothly descends the reconstruction surface without catapulting features across optimal valleys.
2. **Traffic-Shaping Manifold Projection (unbudgeted):**
   In real networks, an attacker mitigates volumetric rate anomalies by rate-pacing packets into normal operating traffic envelopes $[\mathbf{norm\_min}, \mathbf{norm\_max}]$. When `manifold_projection=True`:
   $$\mathbf{x}_{t+1} = \text{clamp}\left(\mathbf{x}_t - \alpha \cdot \mathbf{s}_{\text{global}} \odot \text{sign}(\nabla_{\mathbf{x}} \text{RMSE}(\mathbf{x}_t)), \min=\mathbf{norm\_min}, \max=\mathbf{norm\_max}\right)$$
   No $\epsilon$ ball is applied in this mode. The clamp itself pulls every out-of-envelope feature to the benign boundary, so the measured $L_\infty$ (13–35 normalised units) is far above $\epsilon=2$. Phase 6 added a clamp-only baseline and an untrained-generator ablation because this projection, not the generator, is the dominant evasion mechanism.

### 2.3. DeepFool for Reconstruction Decision Boundaries
DeepFool iteratively projects a detected sample along the shortest orthogonal path toward the manifold surface defined by $\text{RMSE}(\mathbf{x}) = \theta$:
$$f(\mathbf{x}) = \text{RMSE}(\mathbf{x}) - \theta$$
At each iteration $k$:
1. Compute gradient $\mathbf{w}_k = \nabla_{\mathbf{x}} f(\mathbf{x}_k) / \mathbf{s}_{\text{global}}$.
2. Compute distance to boundary: $d_k = \max(f(\mathbf{x}_k), 0.0)$.
3. Update step:
   $$\Delta \mathbf{x}_k = -(1 + \eta) \cdot \frac{d_k}{\|\mathbf{w}_k\|_2^2 + 10^{-12}} \cdot \mathbf{w}_k \odot \mathbf{s}_{\text{global}}$$
4. Project candidate back into valid domain space until $f(\mathbf{x}_k) \le 0$ or $k = k_{\max}$.

---

## 3. Polymorphic WGAN-GP Architecture

### 3.1. Operating in Normalized Feature Space
To prevent gradient explosions across unnormalized AfterImage features (which span 12 orders of magnitude from $10^{-5}$ to $10^7$):
$$\tilde{\mathbf{x}} = \frac{\mathbf{x} - \mathbf{norm\_min}_{\text{global}}}{\mathbf{s}_{\text{global}}}$$
Both Generator and Critic operate in this normalized domain.

### 3.2. Generator (`PolymorphicGenerator`)
- **Input:** Latent vector $\mathbf{z} \sim \mathcal{N}(0, \mathbf{I}_{32})$ concatenated with normalized malicious features $\tilde{\mathbf{x}}_{\text{mal}} \in \mathbb{R}^{100}$ (total input dim: 132).
- **Hidden Layers:**
  - $\text{Linear}(132 \rightarrow 128) \rightarrow \text{LayerNorm} \rightarrow \text{LeakyReLU}(0.2)$
  - $\text{Linear}(128 \rightarrow 256) \rightarrow \text{LayerNorm} \rightarrow \text{LeakyReLU}(0.2)$
  - $\text{Linear}(256 \rightarrow 128) \rightarrow \text{LayerNorm} \rightarrow \text{LeakyReLU}(0.2)$
  - $\text{Linear}(128 \rightarrow 100) \rightarrow \tanh$
- **Budget Scaling (additive mode only):**
  $$\tilde{\delta} = \tanh(\text{MLP}(\mathbf{z}, \tilde{\mathbf{x}}_{\text{mal}})) \cdot \epsilon$$
  $$\tilde{\mathbf{x}}_{\text{adv}} = \tilde{\mathbf{x}}_{\text{mal}} + \tilde{\delta}$$
  Because $\tanh \in [-1, 1]$, the generated perturbation is guaranteed to stay within $\pm \epsilon$ in normalized space. This applies to additive mode only.

  Phase 6 changed the default $\epsilon$ from 0.05 to 2.0 and made training and evaluation share the same value, persisted in the checkpoint metadata. A second review round then removed the post-hoc clamp from manifold mode: it now predicts
  $$\tilde{\mathbf{x}}_{\text{adv}} = \tilde{\mathbf{x}}_{\text{mal}} + \tanh(\text{MLP}(\mathbf{z}, \tilde{\mathbf{x}}_{\text{mal}})) \odot \max(|\tilde{\mathbf{x}}_{\text{mal}}|, \epsilon)$$
  with no envelope clamp. The learned network must produce the target magnitude; an untrained network scores 0% while the trained one reaches 99.9–100% and beats a naive clamp. Manifold mode remains unbudgeted.

### 3.3. Critic (`WGANDiscriminator`)
- **Input:** Normalized feature candidate $\tilde{\mathbf{x}} \in \mathbb{R}^{100}$.
- **Hidden Layers:**
  - $\text{Linear}(100 \rightarrow 128) \rightarrow \text{LayerNorm} \rightarrow \text{LeakyReLU}(0.2)$
  - $\text{Linear}(128 \rightarrow 256) \rightarrow \text{LayerNorm} \rightarrow \text{LeakyReLU}(0.2)$
  - $\text{Linear}(256 \rightarrow 128) \rightarrow \text{LayerNorm} \rightarrow \text{LeakyReLU}(0.2)$
  - $\text{Linear}(128 \rightarrow 1)$ (scalar realism score)
- **1-Lipschitz Gradient Penalty (`compute_gradient_penalty`):**
  $$\hat{\mathbf{x}} = \alpha \tilde{\mathbf{x}}_{\text{real}} + (1 - \alpha) \tilde{\mathbf{x}}_{\text{fake}} \quad \text{where } \alpha \sim \mathcal{U}(0, 1)$$
  $$\mathcal{L}_{\text{GP}} = \mathbb{E}_{\hat{\mathbf{x}}}\left[ (\|\nabla_{\hat{\mathbf{x}}} D(\hat{\mathbf{x}})\|_2 - 1)^2 \right]$$

### 3.4. Hybrid Generator Loss for Phase 4 Closed-Loop Evasion
Standard WGAN trains the generator strictly against the Critic: $\mathcal{L}_G = -\mathbb{E}[D(G(\mathbf{z}, \tilde{\mathbf{x}}))]$. To ensure generated outputs explicitly drop Kitsune's reconstruction error below $\theta = 0.455986$, Phase 4 incorporates a composite objective:
$$\mathcal{L}_G = -\mathbb{E}[D(\tilde{\mathbf{x}}_{\text{adv}})] + \lambda_{\text{evade}} \cdot \mathbb{E}\left[\max(\text{RMSE}(\mathbf{x}_{\text{adv}}) - \theta, 0.0)\right] + \lambda_{\text{dist}} \cdot \|\tilde{\mathbf{x}}_{\text{adv}} - \tilde{\mathbf{x}}_{\text{mal}}\|_2$$
Because `PyTorchKitNET` is fully differentiable, the hinge loss $\max(\text{RMSE} - \theta, 0)$ backpropagates through the autoencoders into the Generator weights, forcing the Generator to synthesize stealthy features that actively minimize reconstruction anomaly scores.

---

## 4. Empirical Validation & Baseline Evasion Sweep

### 4.1. Sample Audit on Detected Packets
We tested FGSM and DeepFool on a batch of 500 detected Mirai attack packets ($\text{RMSE}_{\text{orig}} > \theta$):

| Epsilon Budget ($\epsilon_{L_\infty}$) | Conditional ASR ($\text{ASR}_{\text{cond}}$) | Mean Attack RMSE | Median Attack RMSE |
| :--- | :--- | :--- | :--- |
| **Clean Baseline** | $0.00\%$ | $2.9942$ | $1.6421$ |
| $\epsilon = 0.05$ | $0.00\%$ | $2.9710$ | $1.6144$ |
| $\epsilon = 0.10$ | $0.20\%$ | $2.9436$ | $1.5862$ |
| $\epsilon = 0.25$ | $0.20\%$ | $2.8681$ | $1.5034$ |
| $\epsilon = 0.50$ | $0.00\%$ | $2.7491$ | $1.3697$ |
| $\epsilon = 1.00$ | $0.00\%$ | $2.5548$ | $1.1592$ |
| $\epsilon = 2.00$ | $0.00\%$ | $2.4317$ | $0.9840$ |
| $\epsilon = 5.00$ | $0.00\%$ | $3.6743$ | $1.2185$ |

### 4.2. Comparative Baseline Results: FGSM vs. PGD vs. Manifold Shaping
We evaluated FGSM, standard multi-step PGD, and Manifold Traffic-Shaping PGD on detected Mirai attack batches:

| Attack Method | Perturbation Configuration | Mean Attack RMSE | Conditional ASR ($\text{ASR}_{\text{cond}}$) |
| :--- | :--- | :--- | :--- |
| **Clean Baseline** | Unperturbed | $1.6798$ | $0.00\%$ |
| **Single-Step FGSM** | $\epsilon = 0.25 \cdot \mathbf{s}_{\text{global}}$ | $1.6421$ | $0.20\%$ |
| **Single-Step FGSM** | $\epsilon = 2.00 \cdot \mathbf{s}_{\text{global}}$ (Overshooting) | $2.4317$ | $0.00\%$ |
| **Multi-Step PGD** | 50 steps, $\alpha = 0.05 \cdot \mathbf{s}, \epsilon = 2.0 \cdot \mathbf{s}$ | $0.8243$ | **$37.00\%$** |
| **Manifold PGD** | 50 steps, $\alpha = 0.05 \cdot \mathbf{s}$, $[\mathbf{min}, \mathbf{max}]$ | **$0.0974$** | **$100.00\%$** |
| **WGAN-GP (Prototype)** | 50 steps, $\lambda_{\text{evade}} = 30$, $\epsilon = 2.0$ | **$0.5355$** | **$53.12\%$** |

*Historical note:* this prototype number is in-sample and was not reproduced by the saved Phase 4 checkpoint. After the Phase 6 fixes (shared ε, disjoint split, holdout selection, `n_critic`), the budgeted WGAN reaches 1.90% conditional ASR on the scanning holdout and 0.00% on flood. See `phase_6_detailed.md`.

### 4.3. Analytical Insight
Single-step FGSM successfully reduces the median RMSE from $1.64$ down to $0.98$. However, because Kitsune's ensemble decomposes features across 16 correlated micro-autoencoders, single-step perturbations that reduce error in one micro-AE can inadvertently increase error in another (reflected when $\epsilon$ gets too large at $5.0$). 
This confirms the necessity of **multi-step generative modeling (WGAN-GP)** in Phase 4, which learns the joint benign manifold across all 16 micro-autoencoders simultaneously.

---

## 5. Test Suite Verification (`tests/test_attacker.py`)
- `test_fgsm_attack_reduces_rmse`: Confirms FGSM reduces mean RMSE on detected attack batches.
- `test_deepfool_attack_step`: Confirms iterative minimal projection moves samples toward boundary.
- `test_polymorphic_generator`: Confirms `mutate_raw` respects normalized $\epsilon$ bounds.
- `test_wgan_discriminator_and_gradient_penalty`: Confirms Critic scores shape `[B, 1]` and computes scalar $\mathcal{L}_{GP} \ge 0$.
- **Project Test Results:** 14 / 14 tests passing at the time; 28 / 28 after Phase 6.
