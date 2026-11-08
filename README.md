# Kitsune NIDS vs. Polymorphic WGAN — Evaluation Pipeline

Automated adversarial-ML evaluation harness for the Kitsune/KitNET Network
Intrusion Detection System. It calibrates a benign alert threshold, trains and
runs FGSM / PGD / DeepFool / WGAN-GP attacks in AfterImage feature space, and
reports evasion, distortion, latency, and detection metrics.

## Important result qualification

The **budgeted** WGAN (additive mode, ε·s box) is weak: 1.9% scanning and 0%
flood ASR. Ten feedback-refinement steps raise scanning to 3.9% and cut flood
RMSE from 13.5 to 5.4, but aggregate perturbation grows to 11–20 ε.

The **manifold** WGAN is unbudgeted and now learns the target step itself (no
post-hoc clamp). It reaches 99.9–100% evasion with lower RMSE than the naive clamp
(0.037 vs 0.109 on flood) and closer benign-neighbour distance (0.62 vs 0.87),
while an untrained generator scores 0%. The pipeline still reports the controls:

- `Envelope Clamp Only` — projection with no generator at all
- `WGAN-GP (Manifold, untrained)` — random weights, same forward
- `WGAN-GP (Manifold, trained)` — the learned attack
- `WGAN-GP (Box)` — bounded learned attack, single/best-of-k/refinement

See `docs/detailed/phase_6_detailed.md` for the analysis.

## Repository layout

```
configs/                 YAML configuration
data/raw/                PCAP, TSV dissection, labels (not committed)
data/processed/          memmapped features and model (not committed)
data/attack_samples/     generator/critic checkpoints + training metadata
scripts/                 feature extraction and defender training
src/defender/            differentiable KitNET, threshold calibration
src/attacker/            FGSM, PGD, DeepFool, WGAN-GP generator/critic
src/pipeline/            evaluator and closed-loop feedback training
src/utils/               constraints, metrics, splits, seeding
tests/                   pytest suite
docs/overview/           executive phase summaries
docs/detailed/           full technical logs
reports/                 benchmark tables and JSON/CSV metrics
```

## Setup

Python 3.10+, CUDA 12.x, `tshark`, and the packages in the environment section of
`AGENTS.md`. Then:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install cython scipy scikit-learn dpkt scapy pyyaml pytest black flake8
python -c "import torch; print(torch.cuda.is_available())"
```

The pure-Python AfterImage extractor is the default. Set
`KITSUNE_USE_CYTHON=1` to use the compiled `AfterImage_extrapolate` extension;
if it is missing it is built on import.

## Data preparation

The repository does not commit the 60 MB capture, the 94 MB TSV, or the 305 MB
feature matrix. Follow `docs/detailed/phase_0_5_detailed.md` to extract
`mirai.pcap` and `mirai_labels.csv`, then:

```bash
python scripts/extract_and_train.py        # ~30 min, creates features + model
python src/defender/eval_threshold.py      # calibrates theta on a benign split
```

The extraction script writes `data/processed/mirai_features.npy` and
`data/processed/kitsune_mirai_model.pkl`.

## Reproduce the benchmark

```bash
# Train both WGAN modes (writes checkpoints + .meta.json with the budget)
python run_pipeline.py --mode train_wgan --eps 2.0 --seed 42
python run_pipeline.py --mode train_wgan --eps 2.0 --seed 42 --manifold_shaping

# Full comparative suite over both attack regimes
python run_pipeline.py --mode evaluate_all --samples 1000 --attack_phase both --seed 42

# Individual sweeps
python run_pipeline.py --mode evaluate_clean --seed 42
python run_pipeline.py --mode fgsm_sweep --attack_phase flood --samples 1000 --seed 42
python run_pipeline.py --mode pgd_attack --attack_phase flood --samples 1000 --seed 42
```

All evaluation runs are seeded. The `--eps` value used at training is stored in
`data/attack_samples/*.meta.json` and reused at evaluation, so the two cannot
drift apart silently.

## Evaluation protocol

- Benign packets after grace period 55,000 are split by parity: even indices
  calibrate θ, odd indices measure FPR out-of-sample. They never overlap.
- Attack packets are split by `(index - 121,621) % 3`: remainder 0 trains,
  remainder 1 evaluates. Training and evaluation attack samples never overlap.
- Training draws half its attacks from the early scanning window and half
  spread across the whole capture, so flood traffic is represented.
- The best generator epoch is selected by held-out ASR.

## Testing

```bash
pytest -q
flake8 src tests run_pipeline.py scripts
black --check src tests run_pipeline.py scripts
```

## Documentation

- `docs/implementation_plan.md` — milestone plan
- `docs/detailed/phase_0_detailed.md` … `phase_5_detailed.md` — per-phase logs
- `docs/detailed/phase_6_detailed.md` — review remediation log
- `docs/review_findings.md` — the first independent review
- `docs/final_review_findings.md` — the final review and remediation record
