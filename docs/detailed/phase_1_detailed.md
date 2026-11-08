# Phase 1 Comprehensive Technical Log: Scaffolding, Configurations, Domain Constraints & Differentiable PyTorch KitNET

This document provides an exhaustive, granular engineering record of all directory tree scaffolding, YAML schemas, mathematical perturbation operators, signed feature invariant enforcement, the differentiable PyTorch KitNET wrapper, and unit test suites created during Phase 1.

---

## 1. Directory Tree & Module Scaffolding

### 1.1. Directory Structure Setup Command
Executed to structure the codebase into standard Python package tiers per [AGENTS.md](file:///home/brooklyng3/projects/grad/AGENTS.md):
```bash
mkdir -p configs src/defender src/attacker src/pipeline src/utils tests scripts docs/overview docs/detailed
touch src/__init__.py src/defender/__init__.py src/attacker/__init__.py src/pipeline/__init__.py src/utils/__init__.py tests/__init__.py
```

### 1.2. Architecture Role Assignment
- `src/defender/`: Unsupervised NIDS modules (AfterImage feature extractor wrapper, PyTorch KitNET neural network ensemble with autograd support, and threshold calibration).
- `src/attacker/`: Offensive mutation generators (WGAN-GP generator, critic, FGSM, and DeepFool perturbation modules).
- `src/pipeline/`: Closed-loop automated benchmarker and evaluator connecting attacker and defender.
- `src/utils/`: Domain constraints, scaled projection boundaries, and quantitative evaluation metrics.
- `configs/`: Declarative YAML parameter files separating experiments from code.
- `tests/`: Pytest unit test suites for verifying reproducibility and robustness.

---

## 2. Configuration Profiles

### 2.1. Kitsune NIDS Configuration (`configs/kitsune_config.yaml`)
```yaml
kitsune:
  feature_dim: 100
  max_ae: 10
  fm_grace: 5000
  ad_grace: 50000
  threshold_percentile: 99.9
  learning_rate: 0.1
  hidden_ratio: 0.75

data:
  raw_pcap: "data/raw/mirai.pcap"
  raw_tsv: "data/raw/mirai.pcap.tsv"
  raw_labels: "data/raw/mirai_labels.csv"
  processed_features: "data/processed/mirai_features.npy"
  model_weights: "data/processed/kitsune_mirai_model.pkl"
  packet_limit: null

device: "cuda"
batch_size: 128
```

### 2.2. Attacker Engine Configuration (`configs/wgan_config.yaml`)
```yaml
wgan:
  latent_dim: 32
  feature_dim: 100
  generator_hidden_dims: [128, 256, 128]
  critic_hidden_dims: [128, 256, 128]
  lambda_gp: 10.0
  n_critic: 5
  lr_g: 0.0001
  lr_c: 0.0001
  b1: 0.5
  b2: 0.9

perturbation:
  eps_linf: 2.0   # was 0.05 before Phase 6
  delta_l2: 1.0
  max_iterations: 100
  step_size: 0.01

data:
  processed_features: "data/processed/mirai_features.npy"
  labels: "data/raw/mirai_labels.csv"
  output_samples: "data/attack_samples/perturbed_mirai.npy"

training:
  batch_size: 128
  epochs: 50
  device: "cuda"
```

---

## 3. Network Constraints & Perturbation Operators (`src/utils/constraints.py`)

Section 5 of [AGENTS.md](file:///home/brooklyng3/projects/grad/AGENTS.md) mandates that all adversarial perturbations must respect physical protocol rules, value bounds, and explicit perturbation budgets.

### 3.1. Signed AfterImage Feature Invariants
Kitsune's 100 AfterImage features consist of 5 decay factor windows ($\lambda \in \{5, 3, 1, 0.1, 0.01\}$). Within each 20-feature window, two statistical metrics are signed:
- **Covariance ($Cov$):** Range $(-\infty, +\infty)$ at feature indices `[20, 27, 34, 41, 48, 70, 77, 84, 91, 98]`.
- **Pearson Correlation Coefficient ($PCC$):** Range $[-1.0, +1.0]$ at feature indices `[21, 28, 35, 42, 49, 71, 78, 85, 92, 99]`.

`enforce_feature_bounds` clamps strictly the unsigned physical features (packet counts, byte means, variances, jitter) to $\ge 0.0$ while preserving signed Covariance and clamping PCC strictly to $[-1.0, 1.0]$.

### 3.2. Relative Dynamic-Range Budgeting & Zero-Division Defense
Because unnormalized packet variances reach $10^7$ whereas inter-arrival times are $10^{-5}$, flat scalar perturbations ($\epsilon = 0.05$) fail on large-magnitude features.

We implemented **per-feature dynamic range scaling**:
$$\mathbf{s} = \mathbf{x}_{\max} - \mathbf{x}_{\min}$$
$$\mathbf{x}_{\text{proj}} = \text{clip}(\mathbf{x}_{\text{adv}} - \mathbf{x}_{\text{orig}}, -\epsilon \cdot \mathbf{s}, +\epsilon \cdot \mathbf{s}) + \mathbf{x}_{\text{orig}}$$
$$\mathbf{x}_{\text{proj}, L_2} = \mathbf{x}_{\text{orig}} + \left(\frac{\mathbf{x}_{\text{adv}} - \mathbf{x}_{\text{orig}}}{\mathbf{s}} \cdot \min\left(1, \frac{\delta}{\|\frac{\mathbf{x}_{\text{adv}} - \mathbf{x}_{\text{orig}}}{\mathbf{s}}\|_2 + 10^{-12}}\right)\right) \cdot \mathbf{s}$$

#### Zero-Variance / Denominator Protection:
In raw network captures, certain rare protocol statistics have $x_{\max} == x_{\min}$ (i.e. $\mathbf{s} = 0$). To prevent zero-division that would produce `NaN` values and break autograd `.backward()` gradient backpropagation, the dynamic range vector is clamped:
$$\mathbf{s}_{\text{safe}} = \max(\mathbf{s}, 10^{-12})$$
Similarly, in `DAE.__init__` inside `src/defender/torch_kitnet.py`, the normalization range is bounded:
$$\text{norm\_range} = \max(\text{norm\_max} - \text{norm\_min}, 10^{-12})$$

---

## 4. Differentiable PyTorch KitNET Wrapper (`src/defender/torch_kitnet.py`)

The original `Kitsune-py` KitNET implementation is written in pure NumPy, operating on one sample at a time on CPU with no gradient tracking. To enable GPU batching (64–256) and gradient-based adversarial attack generators (FGSM, DeepFool, WGAN-GP), `src/defender/torch_kitnet.py` was developed:

### 4.1. Core Components
1. **`DAE(nn.Module)`**:
   - Stores normalized bounds `norm_min` and `norm_range` as registered buffers.
   - Encodes via $\text{sigmoid}(\mathbf{x} \mathbf{W} + \mathbf{b}_h)$.
   - Decodes via $\text{sigmoid}(\mathbf{h} \mathbf{W}^T + \mathbf{b}_v)$.
   - Computes RMSE reconstruction error differentiable with respect to input $\mathbf{x}$.
2. **`PyTorchKitNET(nn.Module)`**:
   - `from_kitnet_model(kitnet_obj, device)`: Ingests the trained NumPy ensemble weights and maps them to PyTorch CUDA/CPU tensors with zero data loss.
   - `forward(x)`: Executes the ensemble layer in parallel, feeds intermediate reconstruction scores $\mathbf{S}_{l1}$ into the master autoencoder, and outputs the final anomaly reconstruction RMSE tensor.

---

## 5. Verification Test Suite & Parity Validation

### 5.1. Test Cases Implemented
- `tests/test_constraints.py`:
  - `test_linf_projection_numpy` & `test_linf_scaled_projection`
  - `test_l2_projection_numpy` & `test_l2_projection_torch`
  - `test_afterimage_signed_features_preserved`: Confirms negative Covariance and $[-1.0, 1.0]$ PCCs are preserved.
  - `test_apply_combined_constraints_torch`
- `tests/test_torch_kitnet.py`:
  - `test_pytorch_kitnet_numerical_parity`: Verifies that `PyTorchKitNET` output matches `Kitsune-py` NumPy `kitnet.execute(x)` within numerical tolerance ($\text{atol} = 10^{-4}$).
  - `test_pytorch_kitnet_autograd_gradients`: Verifies that calling `.backward()` on the reconstruction error computes valid, non-zero gradients on input features $\frac{\partial \text{RMSE}}{\partial \mathbf{x}}$.

### 5.2. Test Execution Log
```bash
source .venv/bin/activate
flake8 src/ tests/
black --check src/ tests/
pytest
```
#### Output:
```text
All done! 10 files would be left unchanged.
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-9.2.0, pluggy-1.7.0
rootdir: /home/brooklyng3/projects/grad
configfile: pytest.ini
testpaths: tests
collected 8 items

tests/test_constraints.py ......                                         [ 75%]
tests/test_torch_kitnet.py ..                                            [100%]

============================== 8 passed in 2.84s ===============================
```
