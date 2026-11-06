# Phase 1.5 Summary: Feature Extraction & KitNET Defender Training Run

This document provides a high-level executive summary of the dataset feature extraction pass, memory-mapped storage generation, and unsupervised KitNET defender training executed during Phase 1.5.

---

## 1. Objectives Achieved
1. **Streaming Feature Extraction:** Successfully streamed all 764,137 packets from `data/raw/mirai.pcap.tsv` through Cython-accelerated `AfterImage` (`FE.get_next_vector()`).
2. **Persistent Numerical Feature Tensor:** Built a memory-mapped NumPy matrix `data/processed/mirai_features.npy` ($764,137 \times 100$, `float32`, 291.50 MB), isolating raw packet strings from numerical machine learning pipelines.
3. **KitNET Online Training:** Trained the ensemble online according to benchmark grace periods:
   - `FMgrace = 5000`: CorClust learned feature correlations and partitioned the 100 features into **16 micro-autoencoders**.
   - `ADgrace = 50000`: Trained the 16 micro-autoencoders and the master output autoencoder on benign network traffic.
   - Transitions verified: `train-mode` $\rightarrow$ `Feature-Mapper: execute-mode, Anomaly-Detector: train-mode` $\rightarrow$ `Anomaly-Detector: execute-mode`.
4. **Isolated Neural Network Persistence:** Successfully serialized only the trained neural ensemble (`KitNET`) to `data/processed/kitsune_mirai_model.pkl` (187.36 KB), guaranteeing no unpicklable file handles or parser states.
5. **Differentiable PyTorch CUDA Verification:** Validated that `PyTorchKitNET.from_file()` loads the trained model onto CUDA and performs high-throughput mini-batch forward passes ($128$ samples/batch) yielding exact parity with NumPy inference ($0.009949$).

6. **Global Normalization Bounds Extraction (`get_global_feature_bounds`):** Implemented utility inside `PyTorchKitNET` to reassemble individual micro-autoencoder `norm_min` and `norm_max` vectors into global 100-dimensional bounds mapped by `kitnet.v`. This provides exact dynamic range vectors $\mathbf{s} = \mathbf{norm\_max} - \mathbf{norm\_min}$ for Phase 3 perturbation scaling.
7. **Grace Period Threshold Safeguard (Phase 2 Pre-Condition):** Established explicit rule that the first 55,000 packets (`FMgrace + ADgrace`) must be excluded from threshold calibration. Only post-convergence benign packets ($i \ge 55,000$, yielding 66,621 clean packets) will be used to calibrate $\theta = \text{percentile}(\text{benign\_rmses}, 99.9)$, preventing artificial $0.0$ scores and early SGD fluctuations from distorting the alert threshold.

---

## 2. Artifacts Produced

| File Path | Type | Dimensions / Size | Purpose |
| :--- | :--- | :--- | :--- |
| `data/processed/mirai_features.npy` | Binary (`open_memmap`) | $(764137, 100)$ `float32`, 291.50 MB | Continuous statistical feature matrix for sub-second slicing in evaluation and attack mutations. |
| `data/processed/kitsune_mirai_model.pkl` | Pickled Model | 187.36 KB | Trained KitNET ensemble weights (16 micro-AEs + 1 master AE). |
| `scripts/extract_and_train.py` | Python Script | 138 lines | Self-contained, reproducible pipeline script executing extraction and online training. |

---

## 3. Operational Performance
- **Total Packets Processed:** 764,137
- **Total Wall-Clock Time:** 1,808.47 seconds (~30.1 minutes)
- **Average Ingestion Throughput:** 422.5 packets/second
- **Memory Footprint:** Maintained strictly under 450 MB RAM throughout execution via memory-mapped streaming.
