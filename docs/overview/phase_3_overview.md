# Phase 3 Summary: Offensive Attacker Engine (WGAN-GP & Gradient Perturbations)

This document provides a high-level executive summary of the offensive adversarial attack architecture, mathematical perturbation baselines, and WGAN-GP components developed during Phase 3.

---

## 1. Objectives Achieved
1. **White-Box Gradient Baselines (`src/attacker/perturbations.py`):**
   - **Fast Gradient Sign Method (FGSM):** Autoencoder loss minimization baseline.
   - **Multi-Step Projected Gradient Descent (PGD):** Resolves single-step overshooting and vanishing sigmoid gradients on extreme flood features, achieving 37% evasion under $\epsilon=2.0$.
   - **Learned Manifold Target (WGAN):** The generator predicts the negative step needed to land each feature in the benign envelope, with no post-hoc clamp. Trained: 99.9–100% evasion; untrained: 0%. The mode is **unbudgeted**; a clamp-only baseline remains for attribution.
   - **DeepFool Baseline:** Iterative projection calculating minimal orthogonal gradient steps toward the decision boundary ($\text{RMSE} \le \theta$).
   - **Epsilon Budget Sweep:** Evaluated across $\epsilon \in [0.05, 5.0]$ on detected packets to measure empirical resistance curves.
2. **Conditional Polymorphic WGAN-GP (`src/attacker/generator.py`):**
   - Operates in normalized feature space ($\tilde{\mathbf{x}} = (\mathbf{x} - \mathbf{min})/\mathbf{s}$).
   - Conditions on both latent Gaussian noise $\mathbf{z} \sim \mathcal{N}(0, \mathbf{I}_{32})$ and source malicious features.
   - Enforces $L_\infty$ perturbation limits via $\tanh$ output scaling:
     $$\tilde{\mathbf{x}}_{\text{adv}} = \tilde{\mathbf{x}}_{\text{mal}} + \tanh(\text{MLP}(\mathbf{z}, \tilde{\mathbf{x}}_{\text{mal}})) \cdot \epsilon$$
   - Provides `mutate_raw()` wrapper to seamlessly convert between raw and normalized feature spaces.
   - Hybrid loss ready for Phase 4: Incorporates an explicit hinge evasion loss $\lambda_{\text{evade}} \cdot \max(\text{RMSE}(\mathbf{x}_{\text{adv}}) - \theta, 0)$ alongside the Wasserstein Critic score, driving generator convergence to $> 50\%$ evasion in under 50 training iterations.
3. **Wasserstein Critic with 1-Lipschitz Penalty (`src/attacker/discriminator.py`):**
   - Evaluates realism of normalized generated candidates against normalized benign traffic.
   - Computes 1-Lipschitz Gradient Penalty ($\lambda_{GP} = 10.0$) using linear interpolation $\hat{\mathbf{x}} = \alpha \mathbf{x}_{\text{real}} + (1 - \alpha) \mathbf{x}_{\text{fake}}$, preventing gradient explosion across unnormalized dimensions.
4. **ASR Metric Isolation Guardrail:**
   - Differentiates **Conditional ASR** ($\text{ASR}_{\text{cond}}$) strictly computed over detected packets ($\text{RMSE}_{\text{orig}} > \theta$) from **Overall Evasion Rate** ($\text{ER}_{\text{total}}$), preventing the $76,123$ pre-evading packets from artificially inflating the attacker's success.
5. **Quality & Test Automation (`tests/test_attacker.py`):**
   - Implemented unit tests for FGSM, DeepFool, PolymorphicGenerator, and WGANDiscriminator.
   - Test suite at the time: **14 / 14 passing**; **28 / 28** after the Phase 6 remediation.

---

## 2. Attacker Engine Components

| Component | File Path | Inputs | Outputs / Constraints |
| :--- | :--- | :--- | :--- |
| **FGSM Attack** | `src/attacker/perturbations.py` | `model`, $\mathbf{x}_{\text{det}}$, $\epsilon$, $\mathbf{s}_{\text{global}}$ | $\mathbf{x}_{\text{adv}}$, $\text{RMSE}_{\text{adv}}$ with domain bounds |
| **PGD Multi-Step Attack** | `src/attacker/perturbations.py` | `model`, $\mathbf{x}_{\text{det}}$, $\theta$, $\alpha$, $\epsilon$, steps | Multi-step descent with box or manifold projection |
| **DeepFool Attack** | `src/attacker/perturbations.py` | `model`, $\mathbf{x}_{\text{det}}$, $\theta$, $\mathbf{s}_{\text{global}}$ | Minimal iterative step to $\text{RMSE} \le \theta$ |
| **Polymorphic Generator** | `src/attacker/generator.py` | $\mathbf{z} \in \mathbb{R}^{32}, \tilde{\mathbf{x}} \in \mathbb{R}^{100}$ | Additive: bounded in $[-\epsilon, +\epsilon]$. Manifold: unbudgeted learned target, no post-hoc clamp |
| **WGAN Critic** | `src/attacker/discriminator.py` | $\tilde{\mathbf{x}} \in \mathbb{R}^{100}$ | Scalar realism score + Gradient Penalty ($\lambda_{GP}=10$) |

---

## 3. Initial Baseline Observations
- Applying single-step FGSM on detected Mirai scanning packets steadily decreases their average reconstruction RMSE (e.g. from $2.97$ down to $2.43$), but single-step perturbations struggle to collapse multi-modal autoencoder correlation signatures under tight budgets without multi-step iterative loops or generative adaptation.
- This sets the stage for **Phase 4: Automated Evaluation & Feedback Loop**, where the WGAN-GP generator is trained against both the Critic and the PyTorchKitNET anomaly score in a closed-loop benchmarker.
