# Phase 6 Comprehensive Technical Log: Review Remediation, Split Discipline & Attack Attribution

This document is the exhaustive engineering record of the remediation pass
performed after the independent review in `docs/review_findings.md`. It lists
every code change, new file, configuration change, test change, command,
failure encountered, and post-fix measurement, together with the reasoning.

The review established three critical facts:

1. The 100% "Polymorphic WGAN-GP (Manifold)" result was produced by
   `torch.clamp(..., 0.0, 1.0)` in the generator, not by the learned network; an
   untrained generator matched it.
2. The WGAN was trained with `eps_linf = 0.05` and evaluated at `max_eps = 2.0`;
   its "holdout" score was computed on training samples.
3. FPR was measured on the same benign packets used to calibrate θ.

Phase 6 fixes all three and the remaining medium/low findings.

---

## 1. Remediation map

| Review finding | Remediation |
|---|---|
| F1 clamp attribution | Post-hoc clamp removed from the manifold forward; clamp and untrained controls retained; benign-kNN realism metric added |
| F2 ε mismatch / overfit | Shared `--eps` persisted in metadata; disjoint stratified split; holdout selection; `n_critic` |
| F3 no retry loop | `wgan_best_of_k` plus `wgan_refine` feedback refinement |
| F4 methodology | Calibration/validation split; seed helper; unguarded tests; evasion tests |
| F5 constraints | MTU and IAT bounds on feature channels; TTL limitation documented |
| F6 artifacts/docs | Reports regenerated; README; link repair; Phase 6 docs |
| F7 Cython dead code | Opt-in `KITSUNE_USE_CYTHON=1`; on-demand build |
| F8 unused code | DeepFool benchmarked; `n_critic` executed; config wired |

---

## 2. Split infrastructure (new files)

### 2.1 `src/utils/data_splits.py`

New single source of truth for training/evaluation partitions. Previously the
feedback loop trained on the first 20,000 attacks while Phase A evaluated the
first 10,000 — complete overlap.

```python
ATTACK_START = 121621
ATTACK_SPLIT_STRIDE = 3
ATTACK_TRAIN_REMAINDER = 0
ATTACK_EVAL_REMAINDER = 1
BENIGN_GRACE = 55000
BENIGN_SPLIT_STRIDE = 2
BENIGN_CALIBRATION_REMAINDER = 0
BENIGN_VALIDATION_REMAINDER = 1
```

`attack_indices_for_split()` filters by
`(index - ATTACK_START) % 3 == remainder`; `benign_indices_for_split()` filters
by parity over the post-grace benign range. The unused third attack partition is
deliberately left as slack.

**Reasoning:** a modulo rule is deterministic, requires no shuffle state, is
easy to audit in code review, and guarantees that training and evaluation can
never share a packet. Parity does the same for benign calibration/validation.

### 2.2 `src/utils/seed.py`

`seed_everything()` seeds `random`, NumPy, Torch CPU and CUDA, and disables
cuDNN benchmarking. `run_pipeline.py` seeds at startup; the feedback loop seeds
at the start of `train_loop`; the best-of-k evaluation restores the run seed
after constructing the untrained ablation.

**Reasoning:** the pre-fix WGAN result varied between draws. Without seeds the
benchmark cannot be reproduced.

---

## 3. Threshold calibration versus FPR validation

### 3.1 Code change (`src/defender/eval_threshold.py`)

Before:

```python
benign_indices = np.where(
    (labels == 0) & (np.arange(total_packets) >= grace_period)
)[0]
...
theta = np.percentile(benign_rmses, 99.9)
fp_count = np.sum(benign_rmses > theta)
```

After:

```python
calibration_indices = benign_indices_for_split(labels, "calibration")
validation_indices = benign_indices_for_split(labels, "validation")
benign_rmses = score_indices(calibration_indices)
theta = np.percentile(benign_rmses, 99.9)
validation_rmses = score_indices(validation_indices)
fp_count = np.sum(validation_rmses > theta)
```

Metadata now stores `benign_calibration` and `benign_validation` separately.

**Reasoning:** the previous FPR was arithmetic (`99.9th percentile` guarantees
≈0.1% on the same sample). The validation half is scored only after θ is fixed,
so the reported FPR is out-of-sample.

### 3.2 Recalibration output

```text
Scoring 33,311 calibration and 33,310 validation benign packets on GPU...
Calculated Theta (θ):       0.455986
Calibration RMSE (mean ± std): 0.015203 ± 0.052498
Validation FPR (out-of-sample): 0.096% (32/33310), mean RMSE 0.014967
Attacks Detected:           566,393 (88.15%)
Attacks Evaded (Baseline):  76,123 (11.85%)
```

The new θ (0.455986) is close to the previous 0.453419 and the out-of-sample FPR
0.096% confirms the calibration is not an artefact.

---

## 4. Epsilon consistency and checkpoint metadata

### 4.1 Config

`configs/wgan_config.yaml` changed `perturbation.eps_linf` from `0.05` to `2.0`
and added `best_of_k_retries: 10`.

### 4.2 Feedback loop and CLI

`AdversarialFeedbackLoop.__init__` now accepts an explicit `eps` and falls back
to the config; `run_pipeline.py` exposes `--eps` (default 2.0) and passes it to
both training and evaluation. After training, `run_train_wgan` writes:

```json
{
  "eps": 2.0,
  "seed": 42,
  "mode": "additive",
  "best_epoch": 10,
  "best_selection_asr": 0.1,
  "test_asr": 0.0
}
```

`_load_generator()` in `run_pipeline.py` reads this metadata and constructs the
generator with the recorded budget. Evaluation can no longer silently use a
different epsilon.

**Reasoning:** the pre-fix code trained at 0.05 and constructed the evaluator
generator with `max_eps=2.0`, a 40× mismatch that the tables concealed behind an
"ε = 2.0" label.

---

## 5. Feedback-loop remediation (`src/pipeline/feedback_loop.py`)

### 5.1 Representative, disjoint training data

Before:

```python
attack_indices = np.where(labels == 1)[0]
candidate_attacks = features[attack_indices[:pool_size]]
```

After:

```python
all_attacks = np.where(labels == 1)[0]
train_attacks = attack_indices_for_split(all_attacks, "train")
early = train_attacks[train_attacks < ATTACK_START + 10000]
later = train_attacks[train_attacks >= ATTACK_START + 10000]
chosen = np.concatenate([early[:n_early], self._spread(later, n_late)])
```

Half the pool comes from scanning, half is evenly spread over the remaining
capture. Benign training uses the calibration parity split, leaving validation
untouched.

**Reasoning:** the old pool covered 3% of attacks, all early scanning; flood was
unseen by construction. The new pool covers both regimes without reusing
evaluation packets.

### 5.2 `n_critic` executed

Before, one critic update per generator update despite `n_critic: 5`. Now:

```python
for _ in range(self.n_critic):
    epoch_loss_c += self._train_critic_step(real_batch, mal_norm_batch)
```

**Reasoning:** WGAN-GP requires multiple critic steps between generator steps
to keep the critic near its Lipschitz optimum.

### 5.3 Held-out selection and untouched test split

The evaluation split is divided into two disjoint halves: a **selection**
holdout used every epoch, and an **untouched test** holdout scored once at the
end. Each epoch records selection ASR/mean RMSE:

```python
select_adv = self.generator.mutate_raw(x_select, ...)
select_rmse = self.evaluator.score_batches(select_adv)
...
if select_asr > best_asr or (select_asr == best_asr and select_mean < best_mean):
    best_asr, best_mean, best_epoch = select_asr, select_mean, epoch
    best_state = copy.deepcopy(self.generator.state_dict())
```

The best state is restored before final evaluation and saving. Final evaluation
uses `x_test`, never used for model selection.

**Reasoning:** the old final evaluation sliced `x_mal_raw[:1000]`, i.e. training
samples, while the doc called it a holdout. Selection on a disjoint split and a
separate test half prevent both training leakage and selection bias.

### 5.4 Training telemetry (seeded)

Additive, ε = 2.0, 10 epochs:

```text
Loaded 10,000 strictly detected attack samples (scan=5,000, spread=5,000).
Selection holdout: 1,000 | untouched test holdout: 1,000 (disjoint halves)
Epoch [01] | Loss C: -8.3121 | Loss G: 321.3015 | Batch ASR: 0.04% | Holdout ASR: 0.00%
Epoch [05] | Loss C: -37.2324 | Loss G: 324.6055 | Batch ASR: 0.24% | Holdout ASR: 0.10%
Epoch [10] | Loss C: -77.4663 | Loss G: 344.2202 | Batch ASR: 0.26% | Holdout ASR: 0.10%
Best selection ASR 0.10% at epoch 8. Untouched test: 0.00%, RMSE 13.1033.
```

Manifold, ε = 2.0, 10 epochs:

```text
Epoch [01] | Loss C: -6.3421 | Loss G: 133.9979 | Batch ASR: 11.56% | Holdout ASR: 54.70%
Epoch [05] | Loss C: -0.6948 | Loss G: -3.0472 | Batch ASR: 99.99% | Holdout ASR: 100.00%
Epoch [10] | Loss C: -0.5699 | Loss G: -3.5101 | Batch ASR: 99.97% | Holdout ASR: 100.00%
Best selection ASR 100.00% at epoch 8. Untouched test: 100.00%, RMSE 0.0397.
```

The additive critic loss magnitude grows to -77; performance plateaus at 0.10%
selection ASR and 0.00% on the untouched test half. The manifold generator
learns quickly (54.7% selection ASR after one epoch) and saturates by epoch 5.
The pool print now reports the actual post-detection 5,000/5,000 composition,
and the final score is measured on a half never used for model selection.

---

## 6. Evaluation remediation (`run_pipeline.py`)

### 6.1 New rows

`evaluate_attack_suite()` now emits, in order:

1. Clean baseline
2. **Envelope Clamp Only** (`envelope_projection`)
3. FGSM ε = 0.05 and 0.25
4. PGD box ε = 2.0, 50 steps
5. DeepFool, 50 iterations, ε ≤ 2.0
6. PGD manifold, unbudgeted, 50 steps
7. WGAN box, one draw
8. WGAN box, best-of-k retry (`wgan_best_of_k`)
9. WGAN box, 10-step feedback refinement (`wgan_refine`)
10. WGAN manifold, trained (unclamped learned target)
11. WGAN manifold, **untrained** ablation

Each row reports draws per packet, RMSE, ASR, ER, mean L∞, and latency.

### 6.2 Clamp baseline

New `envelope_projection()` in `src/attacker/perturbations.py`:

```python
x_adv = torch.clamp(x.detach(), min=norm_min, max=norm_max)
x_adv = enforce_feature_bounds(x_adv)
```

**Reasoning:** to measure the mechanism separately from the generator. Without
this row the manifold result cannot be attributed.

### 6.3 Best-of-k retry

`wgan_best_of_k()` draws a fresh latent for each pending sample, retains the
best per-sample RMSE, stops early when the batch has evaded, and returns
per-sample attempt counts:

```python
best = generator.mutate_raw(...)
best_rmse = model(best)
for _ in range(1, max_retries):
    pending = best_rmse > theta
    if not pending.any():
        break
    candidate = generator.mutate_raw(...)
    better = candidate_rmse < best_rmse
    best[better] = candidate[better]
    ...
    attempts[pending] += 1
```

**Reasoning:** supplies the inference-time "until bypass" semantics the original
single-pass WGAN lacked. `wgan_refine()` is a companion that feeds the current
mutated packet back as the next condition (`G(z_t, x_t)`) for K steps, testing
the reviewer's mode-collapse hypothesis. Both stop early when all samples evade
and retain the best per-sample RMSE.

### 6.4 Sampling

`get_detected_attack_sample()` now draws from the evaluation remainder only:

```python
eval_attacks = attack_indices_for_split(np.where(labels == 1)[0], "eval")
if phase == "scanning":
    candidates = eval_attacks[(eval_attacks - ATTACK_START) < num_samples * 10]
else:
    later = eval_attacks[(eval_attacks - ATTACK_START) >= 10000]
```

Clean detection uses the benign validation parity split, not the calibration
split.

### 6.5 Seeding

`main()` calls `seed_everything(args.seed)` before component loading; the
untrained ablation is seeded separately and the run seed restored afterwards.

### 6.6 Benign-manifold proximity metric

`PipelineEvaluator.benign_knn_distance()` reports the mean normalised L2
distance from each candidate to its nearest neighbour among 2,000 benign
validation packets. It is independent of the critic and quantifies the
reviewer's "does not lie on the benign manifold" objection. A benign-to-benign
baseline of 0.1383 is recorded in `reports/clean_detection_metrics.json`.

---

## 7. Domain-constraint enforcement (`src/utils/constraints.py`)

New index sets:

```python
SIZE_MEAN_INDICES = [1,4,7,10,13,16,23,30,37,44,66,73,80,87,94]
IAT_STAT_INDICES = [51,54,57,60,63,52,55,58,61,64]
```

Torch branch:

```python
out.index_copy_(-1, size_mean_idx,
    torch.clamp(out.index_select(-1, size_mean_idx),
                min=float(MIN_PACKET_BYTES), max=float(MTU_MAX_BYTES)))
out.index_copy_(-1, iat_idx,
    torch.clamp(out.index_select(-1, iat_idx), min=MIN_IAT))
```

The NumPy branch mirrors this. Previously the MTU/IAT/TTL constants were
declared but unused.

**Reasoning:** packet-size mean channels cannot exceed the MTU and IAT
statistics cannot be negative. TTL is not part of AfterImage (the extractor
does not request `ip.ttl`), so it remains documented rather than enforced.

---

## 8. Cython opt-in (`Kitsune-py/`)

### 8.1 `netStat.py`

```python
if os.environ.get("KITSUNE_USE_CYTHON") == "1":
    import AfterImage_extrapolate as af
else:
    import AfterImage as af
```

### 8.2 `FeatureExtractor.py`

The dead `use_extrapolation = False` flag now reads the environment and builds
the extension on demand via `importlib.util.find_spec`.

**Reasoning:** the compiled extension was built in Phase 0 but never imported.
Pure Python remains the default because the committed features and model were
produced with it; the opt-in makes the extension usable without changing
reproducibility.

Import check:

```text
>>> import AfterImage_extrapolate
>>> [n for n in dir(af) if 'incStat' in n]
['incStat', 'incStatDB', 'incStat_cov']
```

---

## 9. Test changes (`tests/`)

- Removed the `if metrics[...] == ...` guards in
  `test_evaluator_subsample_extrapolation`; both cases are now asserted.
- Added `test_benign_splits_are_disjoint`.
- Added `test_envelope_projection_evades_detected_samples` (asserts ≥90% ASR).
- Added `test_wgan_best_of_k_never_worse_than_single_draw` (same RNG state;
  asserts per-sample minimum ≤ single draw and attempt bounds).
- Added `test_mtu_and_iat_bounds_enforced`.
- Updated `test_threshold_config_metadata` for the two splits: 33,311
  calibration / 33,310 validation, validation FPR ≤0.5%, TPR >80%.

Suite result:

```text
..........................                                               [100%]
36 passed in 23.50s
```

Formatting/lint:

```text
black --check src tests run_pipeline.py scripts
All done! 25 files would be left unchanged.
flake8 src tests run_pipeline.py scripts
exit 0
```

---

## 10. Reporting and artifact regeneration

Commands executed after the code changes:

```bash
python src/defender/eval_threshold.py
python run_pipeline.py --mode train_wgan --eps 2.0 --seed 42 --train_epochs 10
python run_pipeline.py --mode train_wgan --eps 2.0 --seed 42 --train_epochs 10 --manifold_shaping
python run_pipeline.py --mode evaluate_all --samples 1000 --attack_phase both --seed 42
python run_pipeline.py --mode pgd_attack --attack_phase flood --samples 1000 --seed 42
python run_pipeline.py --mode fgsm_sweep --attack_phase flood --samples 1000 --seed 42
```

Regenerated artifacts:

| Path | Change |
|---|---|
| `data/processed/threshold_config.yaml` | New θ and split metadata |
| `reports/evaluation_table.md` | Clamp/untrained/DeepFool/retry rows, new numbers |
| `reports/clean_detection_metrics.json` | Validation FPR 0.0961% |
| `reports/pgd_benchmark.json` | Manifold ER corrected 12.0 → 100.0 |
| `reports/fgsm_tradeoff.csv` | Regenerated; now tracked via `.gitignore` exception |
| `data/attack_samples/*.meta.json` | Persisted budget/seed/best epoch |
| `data/attack_samples/wgan_manifold_critic.pt` | Separate manifold critic |
| `reports/wgan_training_summary.json` | Additive training history and holdout summary |
| `reports/wgan_manifold_summary.json` | Manifold training history and holdout summary |

The stale `pgd_benchmark.json` now reads:

```text
Manifold Shaping (20 steps) ASR= 100.0 ER= 100.0
Manifold Shaping (50 steps) ASR= 100.0 ER= 100.0
```

---

## 11. Post-fix results

### Phase A — early scanning, 1,000 detected holdout packets

| Attack | Draws/pkt | Mean RMSE | ASR | ER | Mean L∞ |
|---|---:|---:|---:|---:|---:|
| Clean baseline | 0.00 | 5.3381 | 0.00% | 11.85% | 0.0000 |
| Envelope clamp only | 1.00 | 0.1136 | 100.00% | 100.00% | 12.9735 |
| FGSM ε=0.05 | 1.00 | 5.3076 | 0.00% | 11.85% | 0.0500 |
| FGSM ε=0.25 | 1.00 | 5.1910 | 0.00% | 11.85% | 0.2500 |
| PGD box ε=2 | 1.00 | 4.1601 | 2.30% | 13.88% | 1.9983 |
| DeepFool ε≤2 | 1.00 | 4.5108 | 0.00% | 11.85% | 1.9683 |
| PGD manifold | 1.00 | 0.0909 | 100.00% | 100.00% | 13.0325 |
| WGAN box, 1 draw | 1.00 | 4.1925 | 1.60% | 13.52% | 1.9981 |
| WGAN box, best-of-10 | 9.85 | 4.1883 | 1.60% | 13.52% | 1.9983 |
| WGAN box, 10-step refine | 9.68 | 1.5275 | 3.80% | 15.29% | 10.4344 |
| WGAN manifold, trained | 1.00 | 0.0647 | 100.00% | 100.00% | 13.9198 |
| WGAN manifold, untrained | 1.00 | 5.6284 | 0.00% | 11.85% | 6.1065 |

### Phase B — volumetric flood, 1,000 detected holdout packets

| Attack | Draws/pkt | Mean RMSE | ASR | ER | Mean L∞ |
|---|---:|---:|---:|---:|---:|
| Clean baseline | 0.00 | 13.4764 | 0.00% | 11.85% | 0.0000 |
| Envelope clamp only | 1.00 | 0.1092 | 100.00% | 100.00% | 34.9663 |
| FGSM ε=0.05 | 1.00 | 13.4459 | 0.00% | 11.85% | 0.0500 |
| FGSM ε=0.25 | 1.00 | 13.3258 | 0.00% | 11.85% | 0.2500 |
| PGD box ε=2 | 1.00 | 12.2761 | 0.00% | 11.85% | 2.0000 |
| DeepFool ε≤2 | 1.00 | 12.9098 | 0.00% | 11.85% | 2.0000 |
| PGD manifold | 1.00 | 0.0896 | 100.00% | 100.00% | 34.9754 |
| WGAN box, 1 draw | 1.00 | 12.2900 | 0.00% | 11.85% | 1.9998 |
| WGAN box, best-of-10 | 10.00 | 12.2890 | 0.00% | 11.85% | 1.9998 |
| WGAN box, 10-step refine | 10.00 | 5.1945 | 0.00% | 11.85% | 19.7303 |
| WGAN manifold, trained | 1.00 | 0.0391 | 100.00% | 100.00% | 35.3428 |
| WGAN manifold, untrained | 1.00 | 15.0501 | 0.00% | 11.85% | 15.8726 |

### Clean detection (validation split)

```text
Theta: 0.455986
ROC-AUC: 0.941826
TPR: 88.13%  FPR: 0.0961%  Precision: 99.93%  F1: 0.9366
Benign kNN baseline (normalised L2): 0.1383
```

---

## 12. Analysis of post-fix results

1. **Removing the post-hoc clamp reverses the attribution.** The untrained
   generator now scores 0% and RMSE 5.63/15.05, while the trained generator
   reaches 99.9–100% and RMSE 0.078/0.037. The trained generator **beats the
   clamp** (0.114/0.109). This answers the reviewer's central objection: the
   learned network, not the projection, is doing the work.
2. **Realism moves in the right direction but is not solved.** Mean nearest-
   benign distance is 0.138 for benign-to-benign, 0.91/0.87 for clamp,
   and 0.67/0.62 for the trained generator. Training improves manifold
   proximity; the attack still is not a true benign sample. The critic is not
   used as the metric because it is trained against the generator and is
   therefore gameable; kNN is independent.
3. **The budgeted WGAN is no longer zero on scanning**, rising from the broken
   checkpoint's 0.00% to 1.60%. It remains 0.00% on flood, consistent with the
   ε = 2 box being unable to bridge a ~35× dynamic range.
4. **Best-of-k does not help because output variance is near zero**, confirming
   conditional mode collapse. Ten feedback-refinement steps raise scanning ASR
   to 3.80% and cut flood RMSE to 5.19, but total L∞ grows to 10.4/19.7
   normalised units. The reviewer's predicted >80% was not reached with a
   generator trained for one step.
5. **DeepFool is ineffective here.** At ε ≤ 2 it matches PGD box on flood (0%)
   and is worse than PGD box on scanning (0% vs 2.30%), despite ~2 s latency.
   It is retained as a documented negative baseline.
6. **The manifold attack remains unbudgeted.** Its mean L∞ (14–36 normalised
   units) is far above ε = 2; the table shows L∞ and labels the constraint
   "Unbudgeted".

---

## 13. Residual limitations

- TTL is absent from AfterImage and cannot be enforced at feature level.
- No packet reconstruction or joint feature-consistency validation was added;
  the attacks remain statistical-feature attacks.
- The manifold mode is an unbudgeted learned-target attack; it is not a bounded
  adversarial perturbation and is now labelled accordingly.
- Feedback refinement accumulates budgets (10 steps ≈ 20 ε aggregate) and is
  not comparable to one-step ε = 2 rows.
- Mirai and one detector only. The generator's flood failure is a property of
  the ε = 2 box, but other datasets/budgets are untested.
- Best-of-k uses a fixed retry cap; a success-weighted stopping rule or
  larger search could improve coverage at higher latency.

---

## 14. File manifest

New:

- `src/utils/data_splits.py`
- `src/utils/seed.py`
- `tests/test_cli.py`
- `tests/test_kitnet.py`
- `requirements.txt`
- `README.md`
- `docs/review_findings.md`
- `docs/final_review_findings.md`
- `docs/overview/phase_6_overview.md`
- `docs/detailed/phase_6_detailed.md`

Modified:

- `configs/wgan_config.yaml`
- `run_pipeline.py`
- `src/attacker/generator.py`
- `src/attacker/perturbations.py`
- `src/pipeline/evaluator.py`
- `src/defender/eval_threshold.py`
- `src/pipeline/feedback_loop.py`
- `src/utils/constraints.py`
- `tests/test_attacker.py`
- `tests/test_constraints.py`
- `tests/test_pipeline.py`
- `tests/test_threshold.py`
- `Kitsune-py/netStat.py`
- `Kitsune-py/FeatureExtractor.py`
- `scripts/extract_and_train.py`
- `.gitignore`
- `docs/review_findings.md`
- `docs/implementation_plan.md`
- `docs/overview/phase_1_5_overview.md`, `phase_3_overview.md`, `phase_4_overview.md`, `phase_5_overview.md`
- `docs/detailed/phase_3_detailed.md`, `phase_4_detailed.md`, `phase_5_detailed.md`
- `reports/evaluation_table.md`, `reports/clean_detection_metrics.json`, `reports/pgd_benchmark.json`, `reports/fgsm_tradeoff.csv`
- `data/processed/threshold_config.yaml`

---

## 15. Second review round (`img/15.png`–`17.png`)

A second review round raised two substantive concerns:

1. The manifold WGAN was still a clamp in disguise; removing the post-hoc
   `torch.clamp(..., 0.0, 1.0)` would make an untrained generator fail and force
   the learned network to earn the result. The reviewer proposed
   `x_adv = x + tanh(MLP(z, x)) * max(|x|, eps)`.
2. Best-of-k had identical ASR across draws because the latent `z` is dominated
   by the 100 large conditional features (conditional mode collapse); feeding
   the mutated vector back for K refinement steps was predicted to raise ASR
   from 1.60% to >80%.

### 15.1 Validation of the clamp criticism

Measured on 500 detected flood packets:

```text
normalised flood features > norm_max per packet: 14.9 / 100
features pinned at exactly 1.0 after clamp:      14.9 / 100
benign normalised mean / median / p95:           0.173 / 0.045 / 0.766
benign fraction > 1.0:                            0.02%
```

The "40 features at 1.0" figure is sample-dependent but directionally correct:
roughly 15 features are pinned exactly at the ceiling, while benign values
average 0.17. The clamp vector is therefore not a benign sample, even though it
scores well against the autoencoder. Nearest-benign-neighbour distance confirms
it: benign-to-benign 0.138, clamp 0.91 (scan) / 0.87 (flood).

### 15.2 Generator change

`src/attacker/generator.py` manifold branch before:

```python
dynamic_budget = torch.clamp(x_mal_norm, min=budget)
delta_norm = delta_raw * dynamic_budget
x_adv_norm = torch.clamp(x_mal_norm + delta_norm, min=0.0, max=1.0)
```

After:

```python
dynamic_budget = torch.clamp(x_mal_norm.abs(), min=budget)
delta_norm = delta_raw * dynamic_budget
x_adv_norm = x_mal_norm + delta_norm
```

The raw-space `enforce_feature_bounds` still applies (unsigned >= 0, PCC,
MTU, IAT), so this is not unbounded in physical terms, but there is no envelope
clamp in normalised space.

### 15.3 Retraining and controls

Retrained with the same protocol (seed 42, ε = 2.0, 10 epochs, best selection
epoch 8). The untrained control uses seed 999 and the same forward.

| Phase | Method | ASR | Mean RMSE | kNN→benign |
|---|---|---:|---:|---:|
| A | Clamp | 100% | 0.1136 | 0.9096 |
| A | Manifold, untrained | 0.00% | 5.6284 | 32.2566 |
| A | Manifold, trained | 100.00% | **0.0647** | **0.6739** |
| B | Clamp | 100% | 0.1092 | 0.8739 |
| B | Manifold, untrained | 0.00% | 15.0501 | 83.0818 |
| B | Manifold, trained | 100% | **0.0391** | **0.6161** |

The trained generator now beats the clamp on both RMSE (1.5× scan, 3× flood)
and benign proximity, while the untrained generator fails completely. This
answers the "why use a WGAN" question within the unbudgeted mode.

### 15.4 Refinement test

`wgan_refine()` feeds the current mutated vector back into the generator for up
to K steps, retaining only improvements. Ten steps:

| Phase | 1 draw ASR / RMSE | 10-step refine ASR / RMSE | Aggregate L∞ |
|---|---:|---:|---:|
| A | 1.60% / 4.1925 | **3.80% / 1.5275** | 10.43 |
| B | 0.00% / 12.2900 | **0.00% / 5.1945** | 19.81 |

Mode collapse is real — independent draws gave identical RMSE — and refinement
materially lowers reconstruction error, but the predicted >80% ASR does not
materialise on flood. The generator was trained for one-step composition;
training it unrolled over K steps is the obvious next experiment.

### 15.5 New metric

`PipelineEvaluator.benign_knn_distance()` reports mean normalised L2 distance
to the nearest of 2,000 benign validation packets. It is independent of the
critic and lets the benchmark distinguish "low RMSE" from "near benign
traffic". A benign-to-benign baseline of 0.138 is recorded in
`reports/clean_detection_metrics.json`.

### 15.6 Resulting table changes

- Manifold trained/untrained rows updated.
- New `WGAN-GP (Box, feedback refine)` row with draw count and aggregate L∞.
- New `kNN->benign` column on every attack row.
- README, Phase 3/4/5/6 docs and review statuses updated.

### 15.7 Remaining gap

The reviewer's stronger claim — refinement should reach >80% — is not
supported by the measured one-step-trained generator. It is documented as a
negative result rather than tuned to the expected number. Unrolled training is
left as a follow-up.

---

## 16. Final-review remediation (`docs/final_review_findings.md`)

The final review found that `run_pipeline.py` was omitted from commit `cf27792`
even though every other Phase 6 file was committed. The reports were genuine
but produced by an uncommitted working copy, so the delivered CLI could not
reproduce them. This section records the repair.

### 16.1 CLI recovery

The missing edits were recovered from the Phase 6 session transcript by
replaying all 40 file edits (including six intervening `black` passes) against
the `d0322b1` version. Duplicated edit pairs produced repeated `--refine_steps`
and `--deepfool_*` argument definitions; these were deduplicated. Recovered
capabilities now verified in the executable file:

- `--seed`, `--eps`, `--retries`, `--refine_steps`, `--train_epochs`,
  `--train_batch_size`, `--deepfool_max_eps`, `--deepfool_steps`,
  `--deepfool_step`, `--checkpoint_dir`
- eval-split sampler for both phases
- validation-only clean evaluation and benign-kNN baseline
- metadata write/read via `_load_generator`
- clamp, untrained, DeepFool, best-of-k, refinement, and kNN rows
- per-1,000 latency scaling and small-sample shortage warnings

### 16.2 Remaining review findings

| Finding | Change |
|---|---|
| Checkpoint clobber | Mode-specific names and `--checkpoint_dir`; one-epoch manifold temp run wrote only `wgan_manifold_generator.pt`, `wgan_manifold_critic.pt`, meta and summary |
| Ignored train args | Threshold/model/eps/seed/epochs/batch/checkpoint dir all passed to the loop |
| Pool composition | Early window widened to 30k attacks; detection applied per regime before a real 50/50 take; print reports 5,000/5,000 |
| Holdout reuse | Selection and test are disjoint halves; final ASR uses the untouched half |
| Weak tests | Added `tests/test_cli.py` and `tests/test_kitnet.py`; theta loaded from config; full suite 36 passing |
| Unused config | Config values now provide CLI defaults; dead keys removed |
| Hard-coded population | Read from threshold attack metadata into evaluator attributes |
| Latency label | Scaled by `1000 / n`; scanning candidate floor is 10k; shortage warning |
| Missing files | `requirements.txt` and `tests/test_kitnet.py` added |
| Stale docs | Test counts, Phase 4 epoch note, telemetry, and review status updated |

### 16.3 Verification

```text
python run_pipeline.py --help                     all Phase 6 flags present
pytest -q                                          36 passed
flake8 src tests run_pipeline.py scripts           exit 0
black --check src tests run_pipeline.py scripts    clean
evaluate_all --samples 1000 --seed 42              initial RMSE 5.3381;
                                                   WGAN box 1.60% / 4.1925
evaluate_clean --seed 42                           86,853 samples; FPR 0.0961%;
                                                   kNN 0.1383
train_wgan --epochs 1 --checkpoint_dir /tmp        mode-specific files only
```

The committed Phase A report is again reproducible from the delivered CLI.
