# Phase 0 Documentation: Kitsune-py Migration & Setup

This document records the exact steps, modifications, and environment configurations performed during Phase 0 to port and prepare the original `Kitsune-py` repository for modern Python 3.12 and our project pipeline.

---

## 1. Repository Source
- **Upstream Repository:** `https://github.com/ymirsky/Kitsune-py.git`
- **Local Directory:** `Kitsune-py/`
- **Base Commit:** `master` branch (`origin/master`)

---

## 2. Environment & System Tools Setup
- **Python Version:** `3.12.3`
- **Virtual Environment:** `.venv/`
- **CUDA / PyTorch:** PyTorch `2.5.1+cu121` with CUDA support enabled.
- **System Tools Installed:**
  - `tshark` (Wireshark 4.2.2 at `/usr/bin/tshark` for fast compiled native packet dissection)
  - `unzip` (`/usr/bin/unzip`)
- **Python Packages Installed:**
  - `cython==3.3.1` (Required to compile native extensions for AfterImage feature extraction)
  - `scipy==1.18.1` (Required by Kitsune / CorClust)
  - `scikit-learn==1.9.1` (Evaluation and preprocessing utilities)
  - `dpkt==1.9.8` (Streaming packet parsing utility)
  - `scapy==2.8.0` (Packet manipulation and fallback reader)
  - `pyyaml==6.0.3` (Configuration parsing)
  - `pytest==9.2.0`, `black==26.10.1`, `flake8==7.5.0` (Code quality and testing)
  - Pre-existing: `numpy==2.5.4`, `torch==2.5.1+cu121`

---

## 3. Code Modifications (Legacy Python 2 / NumPy 2.x Migration)

### 3.1. Replacement of Deprecated `numpy.Inf` / `np.Inf`
In NumPy 2.x, `numpy.Inf` and `np.Inf` have been completely removed in favor of lowercase `np.inf`.

- **Files Modified:**
  - `Kitsune-py/KitNET/dA.py`
  - `Kitsune-py/AfterImage.py`
  - `Kitsune-py/AfterImage_extrapolate.pyx`
  - `Kitsune-py/example.py`
- **Key Diffs:**
  - `dA.py`: Replaced `numpy.Inf` with `numpy.inf` in normalization vectors `norm_max` and `norm_min`.
  - `AfterImage.py`: Replaced `np.Inf` default arguments with `np.inf`.
  - `AfterImage_extrapolate.pyx`: Replaced all bounds and return anomalies `np.Inf` / `-np.Inf` with `np.inf` / `-np.inf`.
  - `example.py`: Updated `packet_limit = np.inf`.

---

## 4. Cython Native Extension Compilation
To support fast incremental damped window statistics in `AfterImage`:
- **Build Command:**
  ```bash
  cd Kitsune-py && python3 setup.py build_ext --inplace
  ```
- **Generated Artifacts:**
  - `Kitsune-py/AfterImage_extrapolate.c` (Transpiled C source)
  - `Kitsune-py/AfterImage_extrapolate.cpython-312-x86_64-linux-gnu.so` (Compiled native shared object library)

---

## 5. Tshark Detection & Feature Extractor Integration
- `FeatureExtractor.py` detects `/usr/bin/tshark` automatically via `_get_tshark_path()`.
- Verified that `FE` detects `/usr/bin/tshark` directly on Linux/WSL2 without needing AppArmor profile overrides.

---

## 6. Functional Verification Tests
Ran online training and scoring smoke tests using Python 3.12:
- **Test Scenarios:**
  1. Grace period state transition check: Verified that executing before feature mapping completes raises appropriate `RuntimeError`.
  2. Full lifecycle check: Ran 20 incremental samples through `KitNET(10, 3, 5, 10)`:
     - `Feature-Mapper: train-mode` $\rightarrow$ `Feature-Mapper: execute-mode, Anomaly-Detector: train-mode` $\rightarrow$ `execute-mode`.
     - Validated that `execute(x)` computes reconstruction RMSE (e.g., `RMSE: ~0.3861`) without updating model weights.
  3. Feature extractor import: Verified clean import of `FeatureExtractor.py` and recognition of `/usr/bin/tshark`.
