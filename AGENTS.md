# AGENTS.md: Kitsune NIDS vs. Polymorphic WGAN Pipeline

## 1. Project Overview & Scope
This project is an automated adversarial machine learning evaluation pipeline testing the robustness of Network Intrusion Detection Systems (NIDS).

- Target (Defender): *Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection* (NDSS, with modern 2024 PyTorch implementations), an unsupervised machine learning system using an ensemble of autoencoders (KitNET) running on 2D/1D damped window statistics (AfterImage feature extractor) to detect anomalous network traffic.
- Weapon (Attacker): Polymorphic attack generator based on *Polymorphic Adversarial Cyberattacks Using WGAN* (Wasserstein GAN with Gradient Penalty / WGAN-GP) alongside recent 2024 frameworks for gradient perturbation baselines (FGSM, DeepFool network evasion) to subtly modify malicious network traffic so it appears benign to an ML model.
- Objective: An automated evaluation loop where the offensive script continuously mutates malicious attack traffic (e.g., Mirai, SYN DoS, or CIC-IDS) until Kitsune's reconstruction error falls below its anomaly threshold (RMSE <= threshold), effectively building an automated evaluation tool to test an ML-based defensive product's threshold for adversarial evasion.

---

## 2. Hardware Budget & System Limits
Agents must optimize all data loaders, batching, and training pipelines to execute strictly within the local host limits:

- CPU: Intel Core i5-9400F (6 physical cores, 6 threads). Multiprocessing workers must be set to 4 or 5 max to prevent system lockups.
- System RAM: 16.0 GB DDR4 (2666 MHz). Never load complete multi-gigabyte PCAP files into memory with scapy.all.rdpcap(). Stream packets iteratively via scapy.all.PcapReader, dpkt, or pre-extracted parquet/csv tables.
- GPU: NVIDIA GeForce GTX 1660 SUPER (6 GB VRAM).
  - Use batch sizes between 64 and 256.
  - Call torch.cuda.empty_cache() between evaluation and mutation cycles.
  - Keep VRAM usage under 4 GB to leave headroom for the OS.

---

## 3. Repository Directory Structure

```text
.
├── AGENTS.md
├── README.md
├── requirements.txt
├── configs/
│   ├── kitsune_config.yaml
│   └── wgan_config.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── attack_samples/
├── src/
│   ├── defender/
│   │   ├── afterimage.py
│   │   └── kitnet.py
│   ├── attacker/
│   │   ├── generator.py
│   │   ├── discriminator.py
│   │   └── perturbations.py
│   ├── pipeline/
│   │   ├── evaluator.py
│   │   └── feedback_loop.py
│   └── utils/
│       ├── constraints.py
│       └── metrics.py
└── tests/
    ├── test_kitnet.py
    └── test_attacker.py
```

---

## 4. Environment Setup

### Prerequisites
- Python 3.10 or higher
- NVIDIA CUDA 12.x drivers installed on the host system

### Virtual Environment Setup

On Windows Command Prompt or PowerShell:
```bash
python -m venv .venv
.venv\Scripts\activate
```

On Linux or WSL2:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### PyTorch CUDA Installation
```bash
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Additional Dependencies
```bash
pip install scikit-learn numpy pandas pyyaml scapy dpkt tqdm pytest black flake8
```

### Hardware Verification Command
```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

---

## 5. Domain Invariants & Network Constraints
Adversarial perturbations must produce valid, routable packets. The agent must enforce the following rules in `src/utils/constraints.py`:

1. Immutable Fields: Never mutate Ethernet source/destination MAC addresses, IP header checksums, protocol identifiers, or TCP handshake flags unless explicitly targeting that protocol layer.
2. Value Bounds:
   - Packet sizes must be non-negative integers <= 1500 bytes (standard MTU).
   - Inter-Arrival Times (IAT) must be non-negative floats (IAT >= 0.0).
   - Time-to-Live (TTL) values must be integers between 1 and 255.
3. Perturbation Budgets: Constrain perturbation magnitudes using L-infinity (L_inf <= epsilon) or L2 (L2 <= delta) bounds so the underlying malicious payload remains functional.

---

## 6. Coding Standards & Agent Workflows

- Ignored Files & Context Discipline: Strictly adhere to `.gitignore` and `.agentignore`. Never load, index, or dump raw PCAP/PCAPNG files (`*.pcap`, `*.pcapng`), full tabular dumps (`*.csv`, `*.tsv`), or serialized models/checkpoints (`*.pkl`, `*.pt`, `*.pth`, `*.onnx`, `*.npy`, etc.) directly into LLM prompts or git commits. Stream or inspect data in small targeted chunks using scripts.
- Type Annotations: All Python functions and methods must include typing annotations (e.g., from typing import List, Dict, Optional, Tuple).
- Formatting: Run black on all changed files before committing. Check code quality using flake8.
- Device Allocation: Never hardcode cuda:0. Always declare:
  ```python
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  ```
- Inference Optimization: Wrap all Kitsune anomaly detection and evaluation runs in `with torch.no_grad():` blocks to avoid memory leaks.
- Testing: Execute `pytest tests/` prior to proposing code updates or starting automated evaluation passes.
