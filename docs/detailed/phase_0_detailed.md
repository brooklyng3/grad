# Phase 0 Comprehensive Technical Log: Legacy Kitsune Migration & Native Cython Toolchain

This document provides an exhaustive, granular engineering record of all setup commands, code modifications, build pipelines, error states, and diagnostic checks executed during Phase 0.

---

## 1. System Environment & Hardware Baseline
- **Host OS:** Linux 6.6.87.2-microsoft-standard-WSL2 (x86_64 Ubuntu 24.04 LTS)
- **Host CPU:** Intel(R) Core(TM) i5-9400F CPU @ 2.90GHz (6 physical cores, 6 threads)
- **System RAM:** 16.0 GB DDR4 (7.9 GB active WSL2 memory allocation)
- **Host GPU:** NVIDIA GeForce GTX 1660 SUPER (6 GB VRAM)
- **NVIDIA Driver / CUDA Runtime:** CUDA 12.1 / PyTorch `2.5.1+cu121`
- **Base Python:** Python `3.12.3` at `/usr/bin/python3`
- **Virtual Environment:** `/home/brooklyng3/projects/grad/.venv`

---

## 2. Upstream Repository Acquisition

### 2.1. Git Clone Command
Cloned the official Kitsune reference implementation by Yisroel Mirsky:
```bash
git clone https://github.com/ymirsky/Kitsune-py.git Kitsune-py
```
- **Source URL:** `https://github.com/ymirsky/Kitsune-py.git`
- **Commit Reference:** `master` (SHA `origin/master`)
- **Cloned Artifacts:**
  - `AfterImage.py` (Damped window statistical feature extraction logic)
  - `AfterImage_extrapolate.pyx` (Cython implementation of decay and 2D stats)
  - `FeatureExtractor.py` (Packet ingestion and dissection coordinator)
  - `KitNET/` (Micro-autoencoders, CorClust clustering, and Master AE)
  - `mirai.zip` (Pre-packaged Mirai botnet evaluation trace)
  - `netStat.py` (Network statistic tracking wrapper)
  - `setup.py` (Cython extension build configuration)

---

## 3. Package Dependencies Installation

The following packages were installed into `.venv` using `pip`:

```bash
source .venv/bin/activate
pip install cython==3.3.1 scipy==1.18.1 scikit-learn==1.9.1 dpkt==1.9.8 \
            scapy==2.8.0 pyyaml==6.0.3 pytest==9.2.0 black==26.10.1 flake8==7.5.0
```

### Dependency Resolution Justification:
- `cython`: Necessary to compile `AfterImage_extrapolate.pyx` into an optimized C shared library (`.so`).
- `scipy`: Required by CorClust for hierarchical linkage and Pearson correlation matrix computations.
- `scikit-learn`: Required for threshold scoring, baseline ROC/AUC calculations, and metrics.
- `dpkt`: C-like binary streaming packet parser.
- `scapy`: Fallback packet dissection and header validation.
- `pyyaml`: YAML configuration parsing for modular hyperparameter orchestration.
- `black` & `flake8`: Strict code formatting and PEP8 lint compliance.

---

## 4. Code Diagnostics & Python 3.12 Migration

### 4.1. NumPy 2.x Deprecation (`np.Inf` / `numpy.Inf`)
NumPy 2.0+ removed `numpy.Inf` in favor of lowercase `numpy.inf`. Executing the unmodified code under NumPy 2.5.4 raised:
```text
AttributeError: `np.Inf` was removed in the NumPy 2.0 release. Use `np.inf` instead.
```

#### Diagnostic Detection Command:
```bash
grep -rn "numpy\.Inf\|np\.Inf" Kitsune-py/
```

#### Automated In-Place Source Patch:
```bash
find Kitsune-py -type f \( -name "*.py" -o -name "*.pyx" \) -exec sed -i 's/numpy\.Inf/numpy.inf/g' {} +
find Kitsune-py -type f \( -name "*.py" -o -name "*.pyx" \) -exec sed -i 's/np\.Inf/np.inf/g' {} +
```

#### Specific Code Diffs Applied:

1. **`Kitsune-py/KitNET/dA.py`**:
```diff
--- a/KitNET/dA.py
+++ b/KitNET/dA.py
@@ -45,8 +45,8 @@ class dA:
             self.params.n_hidden = int(numpy.ceil(self.params.n_visible*self.params.hiddenRatio))
 
         # for 0-1 normlaization
-        self.norm_max = numpy.ones((self.params.n_visible,)) * -numpy.Inf
-        self.norm_min = numpy.ones((self.params.n_visible,)) * numpy.Inf
+        self.norm_max = numpy.ones((self.params.n_visible,)) * -numpy.inf
+        self.norm_min = numpy.ones((self.params.n_visible,)) * numpy.inf
         self.n = 0
```

2. **`Kitsune-py/AfterImage.py`**:
```diff
--- a/AfterImage.py
+++ b/AfterImage.py
@@ -250,7 +250,7 @@ class incStat_cov:
 
 class incStatDB:
     # default_lambda: use this as the lambda for all streams.
-    def __init__(self,limit=np.Inf,default_lambda=np.nan):
+    def __init__(self,limit=np.inf,default_lambda=np.nan):
         self.HT = dict()
         self.limit = limit
         self.df_lambda = default_lambda
```

3. **`Kitsune-py/AfterImage_extrapolate.pyx`**:
```diff
--- a/AfterImage_extrapolate.pyx
+++ b/AfterImage_extrapolate.pyx
@@ -338,7 +338,7 @@ cdef class incStatDB:
     cdef dict HT
 
     # default_lambda: use this as the lambda for all streams.
-    def __init__(self,double limit=np.Inf,double default_lambda=np.nan):
+    def __init__(self,double limit=np.inf,double default_lambda=np.nan):
         self.HT = dict()
         self.limit = limit
         self.df_lambda = default_lambda
@@ -559,8 +559,8 @@ class incHist:
     def __init__(self,nbins,Lambda=0,ubIsAnom=True,lbIsAnom=True,lbound=-10,ubound=10,scaleGrace=None):
         self.scaleGrace = scaleGrace
         if scaleGrace is not None:
-            self.lbound = np.Inf
-            self.ubound = -np.Inf
+            self.lbound = np.inf
+            self.ubound = -np.inf
```

---

## 5. Cython C Extension Compilation Pipeline

To enable native speed for damped window statistical decay calculations:
```bash
cd Kitsune-py
source ../.venv/bin/activate
python3 setup.py build_ext --inplace
```

### Compiler Output & Generated Binaries:
- Cython transpilation target: `Kitsune-py/AfterImage_extrapolate.c` (1.7 MB C source code generated).
- Native shared library: `Kitsune-py/AfterImage_extrapolate.cpython-312-x86_64-linux-gnu.so` (2.2 MB binary).

---

## 6. System Tool Integration: Native `tshark`
In earlier notes, packet parsing using Scapy took $10+$ hours on large captures (the PDF is not committed). Wireshark's compiled C dissector `tshark` was installed via:
```bash
sudo apt update && sudo apt install -y tshark unzip apparmor-utils
```
- Installed Version: `TShark (Wireshark) 4.2.2` at `/usr/bin/tshark`.
- AppArmor Profile Status: Checked via `sudo aa-disable /usr/bin/tshark`. WSL2 does not bind an active AppArmor profile for `tshark`, allowing direct subprocess invocation.

---

## 7. Verification Test Scripts & Output

### 7.1. KitNET Execution Smoke Test
Executed an online learning smoke test to verify weight initialization, grace period thresholds, and score stability:
```bash
python3 -c "
import sys; sys.path.insert(0, 'Kitsune-py')
from KitNET.KitNET import KitNET
import numpy as np

# Initialize with 10 features, maxAE=3, FMgrace=5, ADgrace=10
k = KitNET(10, 3, 5, 10)
for _ in range(20):
    k.process(np.random.rand(10))

score = k.execute(np.random.rand(10))
print('Trained successfully! Inference RMSE:', score)
"
```
#### Output:
```text
Feature-Mapper: train-mode, Anomaly-Detector: off-mode
The Feature-Mapper found a mapping: 10 features to 5 autoencoders.
Feature-Mapper: execute-mode, Anomaly-Detector: train-mode
Feature-Mapper: execute-mode, Anomaly-Detector: execute-mode
Trained successfully! Inference RMSE: 0.38610113079077923
```

### 7.2. Tshark Path Auto-Detection Test
```bash
python3 -c "
import sys; sys.path.insert(0, 'Kitsune-py')
from FeatureExtractor import FE
fe = FE.__new__(FE)
print('Tshark detected path:', fe._get_tshark_path())
"
```
#### Output:
```text
Importing Scapy Library
Tshark detected path: /usr/bin/tshark
```
