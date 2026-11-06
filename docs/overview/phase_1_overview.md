# Phase 1 Summary: Project Scaffolding, Domain Constraints & Differentiable PyTorch KitNET

This document provides a high-level executive summary of the repository architecture, configuration files, mathematical constraint boundaries, and the differentiable PyTorch KitNET wrapper established during Phase 1.

---

## 1. Objectives Achieved
1. **Modular Directory Structure:** Built standard Python package layout under `src/` (`defender`, `attacker`, `pipeline`, `utils`), `configs/`, `tests/`, and `scripts/`.
2. **Centralized Configuration:** Created YAML specifications for both the Kitsune anomaly detection parameters and the WGAN-GP / perturbation engine.
3. **Signed AfterImage Statistical Invariants:** Preserved legitimate negative values for Covariance and $[-1.0, 1.0]$ bounds for Pearson Correlation Coefficients across all 5 damped window scales.
4. **Relative Dynamic-Range Scaled Perturbations:** Upgraded $L_\infty$ and $L_2$ projections to scale with per-feature dynamic ranges $\mathbf{s} = \mathbf{x}_{\max} - \mathbf{x}_{\min}$ so that perturbation budgets have uniform effectiveness across variance features ($10^7$) and inter-arrival times ($10^{-5}$).
5. **Numerical Stability & Zero-Division Defense:** Protected dynamic range divisions and normalization denominators across `src/utils/constraints.py` and `src/defender/torch_kitnet.py` (clamped with $\max(\mathbf{s}, 10^{-12})$) to prevent `NaN` values and gradient collapses on constant/zero-variance features.
6. **Differentiable PyTorch KitNET Wrapper (`src/defender/torch_kitnet.py`):** Built a native PyTorch implementation of KitNET enabling CUDA batching (64–256) and autograd `.backward()` for gradient-based adversarial attack generators (FGSM, DeepFool, WGAN-GP).
7. **Quality & Test Automation:** Established Pytest harness (`pytest.ini`), Flake8 styling profile (`.flake8`), and 100% test pass on all constraints and PyTorch/NumPy parity checks with zero lint errors.

---

## 2. Directory Layout Established

```text
.
├── configs/
│   ├── kitsune_config.yaml
│   └── wgan_config.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── attack_samples/
├── docs/
│   ├── overview/
│   │   ├── phase_0_overview.md
│   │   ├── phase_0_5_overview.md
│   │   └── phase_1_overview.md
│   └── detailed/
│       ├── phase_0_detailed.md
│       ├── phase_0_5_detailed.md
│       └── phase_1_detailed.md
├── src/
│   ├── defender/
│   │   └── torch_kitnet.py
│   ├── attacker/
│   ├── pipeline/
│   └── utils/
│       └── constraints.py
├── tests/
│   ├── test_constraints.py
│   └── test_torch_kitnet.py
├── pytest.ini
└── .flake8
```

---

## 3. Configuration Profiles
- **`configs/kitsune_config.yaml`**: Standardizes `feature_dim: 100`, `max_ae: 10`, `fm_grace: 5000`, `ad_grace: 50000`, and `threshold_percentile: 99.9`.
- **`configs/wgan_config.yaml`**: Standardizes latent noise dimension (32), critic iterations ($n_{\text{critic}}=5$), gradient penalty weight ($\lambda_{GP}=10.0$), and relative perturbation boundaries ($\epsilon_{L_\infty}=0.05, \delta_{L_2}=1.0$).

---

## 4. Constraint Operators Implemented
In `src/utils/constraints.py`:
- `project_linf_ball`: Clamps modifications within $L_\infty \le \epsilon \cdot \mathbf{s}$ ball around baseline vectors.
- `project_l2_ball`: Normalizes and rescales Euclidean perturbation drift in normalized feature space.
- `enforce_feature_bounds`: Ensures non-negativity on unsigned physical features while strictly preserving signed Covariances in $(-\infty, +\infty)$ and PCCs in $[-1.0, 1.0]$.
- `apply_domain_constraints`: Unified projection pipeline chaining $L_\infty$, $L_2$, and domain boundary enforcements.

---

## 5. Verification Status
- **Pytest:** 8 / 8 tests passing (including numerical parity and autograd gradient backprop).
- **Black:** 100% compliant formatting.
- **Flake8:** Zero lint or line-length warnings.
