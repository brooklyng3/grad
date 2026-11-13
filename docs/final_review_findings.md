# Final Independent Review — Kitsune NIDS vs. Polymorphic WGAN

**Date:** 2026-11-09
**Reviewer:** Final reviewing agent
**Scope:** Committed HEAD `cf27792` ("finished") plus working tree; all source, tests, configs, docs, reports, and artifacts.
**Constraint:** No repository source, docs, or artifacts were modified during the review. Test scripts were written to `/tmp` and deleted afterwards. This report file is the only addition.

> **Remediation status (2026-11-09): all critical and high findings fixed.**
> `run_pipeline.py` was recovered from the Phase 6 transcript, restored, and
> extended. The sampler now uses the eval split, clean evaluation uses the
> validation split, metadata is written and read, clamp/untrained/DeepFool/
> best-of-k/refine/kNN rows are wired, manifold checkpoints no longer clobber
> additive ones, population counts come from threshold metadata, latency is
> per-1,000, and requirements/tests were added. Reports were regenerated from
> the restored CLI and match the committed references. See section 8 below and
> `docs/detailed/phase_6_detailed.md` section 16.

---

## 1. Verdict

The underlying research is real and mostly sound: data integrity holds, the PyTorch defender is numerically faithful, the committed checkpoints behave exactly as their summaries claim, the untrained control genuinely scores 0%, and the committed Phase A benchmark table **can be reproduced exactly** using the evaluation protocol described in `phase_6_detailed.md`.

**However, the committed repository does not contain the `run_pipeline.py` that produced the Phase 6 reports.** The final commit `cf27792` includes every other Phase 6 file but omits `run_pipeline.py`, which is byte-identical to its pre-Phase-6 state (`d0322b1`). As a result:

- Every documented command that uses Phase 6 flags fails.
- The CLI emits 7 of the 12 committed benchmark rows.
- The evaluation sampler leaks training packets into the evaluation set.
- The clean FPR is half in-sample.
- Checkpoint metadata is neither written nor read.
- The retry-until-bypass objective is not reachable from the CLI.

The reports are genuine — they were produced by an uncommitted newer version of the pipeline — but the delivered repository cannot regenerate them.

Severity scale: **Critical** = invalidates a headline deliverable or reproducibility; **High** = invalidates a comparison or key property; **Medium** = methodology/reporting weakness; **Low** = hygiene.

---

## 2. Critical findings

### C1 — The delivered CLI is the pre-Phase-6 version; committed reports are not reproducible

**Evidence:**

- `git diff d0322b1 HEAD -- run_pipeline.py` is empty; `run_pipeline.py` has not changed since `d0322b1`.
- `phase_6_detailed.md:564` lists `run_pipeline.py` among modified files, but commit `cf27792` does not contain it.
- The documented command fails:

  ```text
  $ python run_pipeline.py --mode train_wgan --eps 2.0 --seed 42 --train_epochs 10
  run_pipeline.py: error: unrecognized arguments: --eps 2.0 --seed 42 --train_epochs 10
  ```

- `--seed`, `--eps`, `--retries`, `--train_epochs`, and `--deepfool_max_eps` are all absent (`run_pipeline.py:49-133`).
- `evaluate_attack_suite()` emits only **7 rows** (`run_pipeline.py:514-793`); `reports/evaluation_table.md` has **12 rows per phase**. Missing:
  - Envelope Clamp Only
  - DeepFool
  - WGAN best-of-k
  - WGAN feedback refinement
  - Untrained manifold generator
  - The entire `kNN→benign` column
- `envelope_projection`, `wgan_best_of_k`, `wgan_refine`, `deepfool_attack`, and `benign_knn_distance` exist in the source but are never imported or called by `run_pipeline.py` (`run_pipeline.py:30`).
- There is no `seed_everything()` in `run_pipeline.py`, despite README:92 and `phase_6_detailed.md:66-69` claiming seeded evaluation.
- There is no metadata writing/reading and no `_load_generator()` function, despite `phase_6_detailed.md:133-148`.

**Reproduction evidence:**

Running the committed CLI:

```text
python run_pipeline.py --mode evaluate_all --samples 1000 --attack_phase scanning --output_dir /tmp/...
[*] [SCANNING] Selected 1000 detected packets. Initial mean RMSE: 3.7961
| Polymorphic WGAN-GP (Box) | ... | 2.7135 | 6.00% | ...
| Polymorphic WGAN-GP (Manifold) | ... | 0.1083 | 100.00% | ...
```

Committed table: WGAN-box 1.90% / 4.1836; manifold 99.90% / 0.0782.

Using the eval-split sampler plus `seed_everything(42)` (the missing implementation), the committed Phase A table reproduces **exactly**: initial RMSE **5.3381**, WGAN-box **1.90% / 4.1836 / L∞ 1.9990**. This proves the reports came from the missing code and are not fabricated.

**Recovery:** the missing `run_pipeline.py` changes (including `_load_generator`, the eval-split sampler, and seed calls) are present in the Phase 6 session transcript:

```text
~/.claude/projects/-home-brooklyng3-projects-grad/ac0b2f65-ec08-4e21-aa68-5e7c74bd187c.jsonl
```

### C2 — Current evaluation sampling leaks training data into Phase A

`get_detected_attack_sample()` samples `np.arange(ATTACK_START, ATTACK_START + samples*10)` for scanning and an all-attack `linspace` for flood (`run_pipeline.py:198-205`), not the modulo evaluation remainder defined in `src/utils/data_splits.py`.

Measured with the exact committed sampler at `--samples 1000`:

| Phase | Selected | In WGAN training pool | From `%3==1` eval split |
|---|---:|---:|---:|
| Scanning | 1,000 | **337 (33.7%)** | 332 (33.2%) |
| Flood | 1,000 | 24 (2.4%) | 336 (33.6%) |

This is why the current CLI reports 6.00% scanning ASR versus the 1.90% held-out figure. README:100-101 and `phase_6_detailed.md:60-62` claim the split guarantees disjoint evaluation; the split module does, but the evaluator ignores it.

### C3 — Clean FPR is half in-sample, and the committed clean report is unreproducible

`run_evaluate_clean()` scores the entire post-grace benign slice 55,000–121,621 (`run_pipeline.py:245-249`), which includes the 33,311 calibration packets that set θ. It does not use the validation parity split and never computes `benign_knn_baseline`.

| Quantity | Committed report | Current CLI |
|---|---:|---:|
| Benign packets scored | 33,310 (validation only) | 66,621 (calibration + validation) |
| Total samples | 86,853 | 120,164 |
| FPR | 0.0961% | 0.0991% |
| Precision | 99.93% | 99.86% |

The current `--mode evaluate_clean` therefore cannot regenerate `reports/clean_detection_metrics.json`, despite Phase 6 claiming out-of-sample FPR.

### C4 — "Shared epsilon persisted in metadata" is not implemented

The `.meta.json` files exist, but **no code writes or reads them**. `run_pipeline.py` hard-codes `max_eps=2.0` when constructing both generators (`run_pipeline.py:705-711,749-755`). The config default was changed to 2.0, which masks the defect until someone trains at a different budget. The claim that "evaluation can no longer silently use a different epsilon" (`phase_6_detailed.md:147-149`) is false.

### C5 — The core "mutate until bypass" objective is not in the executable pipeline

`wgan_best_of_k()` and `wgan_refine()` exist (`src/attacker/perturbations.py:53-151`) but `run_pipeline.py` performs single-pass attacks only. The shipped tool does not run the objective's retry loop, and the committed best-of-k/refine rows cannot be regenerated. The only tests for these functions use an untrained generator.

---

## 3. High and medium findings

1. **Manifold training clobbers the additive checkpoint.** `train_loop()` defaults to `checkpoint_name="wgan_generator.pt"` (`feedback_loop.py:235-236,409-415`). `run_train_wgan()` never passes a mode-specific name (`run_pipeline.py:485-499`), so a manifold run overwrites `wgan_generator.pt` and `wgan_critic.pt`, and never creates `wgan_manifold_critic.pt`. Verified with a one-epoch manifold run into `/tmp`: only `wgan_generator.pt` and `wgan_critic.pt` were written.
2. **`run_train_wgan` ignores `--threshold_config` and `--model_path`**, hard-codes 10 epochs and batch 128 (`run_pipeline.py:478-490`), and passes neither `eps` nor `seed`. F8 was only partly fixed.
3. **Training pool is not "half early / half spread".** The final 10,000-sample pool measured **25.3% early / 74.7% later**. `feedback_loop.py:188-192` prints pre-detection, pre-truncation counts (3,334/16,666), not the actual mix.
4. **Tests do not protect the headline.** No test loads either trained WGAN checkpoint or asserts trained-WGAN evasion; best-of-k/refine tests use random weights; `theta` is stale (0.453419 vs the configured 0.455986). The suite passes 28/28 even though the CLI is broken and emits half the table.
5. **Holdout is reused after model selection.** The best epoch is selected on the 1,000-packet holdout and the final ASR is reported on that same holdout. It is disjoint from training but not an untouched test set.
6. **Unused config keys:** `delta_l2`, `max_iterations`, `step_size`, `best_of_k_retries`, `training.epochs`, `training.batch_size`, `data.output_samples`.
7. **Hard-coded population constants.** `NUM_PRE_EVADED=76123` and `TOTAL_ATTACKS=642516` (`run_pipeline.py:45-46`) are not read from threshold metadata; ER becomes stale after recalibration.
8. **Latency is mislabelled for non-default sample counts.** It is raw milliseconds for the current batch but reported as "ms / 1k pkts"; correct only when `--samples 1000`. At `--samples 100` scanning, only 6 detected packets are selected without warning.
9. **Missing layout files:** `requirements.txt` and `tests/test_kitnet.py` are required by AGENTS §3 but absent.
10. **Stale or inconsistent docs:** implementation plan says 22 tests; phase 4 still states 15 epochs/6.70%; `phase_6_detailed.md` §5.4 manifold telemetry is from the pre-clamp-removal run (batch ASR 100% epoch 1) while the artifact summary says 7.81%. `docs/review_findings.md` marks F1/F2/F3/F4/F6/F8 "Fixed" more strongly than the committed code supports.

---

## 4. What verified well

- **Data integrity:** `(764137, 100) float32`; 121,621 benign then 642,516 attacks; no benign packet after attack onset; labels `{0,1}`; sampled values finite.
- **Quality gates:** `pytest` → 28 passed; `flake8` → exit 0; `black --check` → clean.
- **Defender parity:** PyTorch KitNET numerical-parity and autograd tests pass; evaluator ASR/ER arithmetic is correct.
- **Threshold calibration:** `eval_threshold.py` genuinely calibrates on 33,311 benign and validates on 33,310 disjoint benign (FPR 0.0961% in the YAML).
- **Checkpoint behaviour matches the summaries:** additive holdout **0.10%** ASR; manifold **100%**, RMSE **0.0387**; untrained control **0%** (RMSE 5.98/14.79).
- **Committed table is genuine:** with the eval-split sampler and seed 42, Phase A reproduces exactly.
- **Constraint index map:** MTU/IAT indices correctly match AfterImage's layout (MI 0–14, HH 15–49, jitter 50–64, HpHp 65–99).
- **Cython opt-in** is wired (`KITSUNE_USE_CYTHON=1`); signed covariance/PCC handling is preserved.

---

## 5. Requirement scorecard

| Requirement | Status | Evidence |
|---|---|---|
| Automated evaluation pipeline against a NIDS | **Met** | CLI, evaluator, metrics |
| Calibrate and test an evasion threshold | **Met mechanically** | Calibrator split-based; CLI FPR half in-sample (C3) |
| Offensive script mutates until bypass | **Partial** | Helpers exist, not wired (C5) |
| WGAN-GP as the weapon | **Implemented; weak when budgeted** | 1.9%/0% at ε=2; strong unbudgeted manifold |
| Disjoint evaluation | **Fails in CLI** | 33.7% training overlap (C2) |
| Reproducible, artifact-backed results | **Fails** | CLI cannot regenerate reports; flags absent (C1) |
| Physical/domain constraints | **Partial** | MTU/IAT enforced; TTL absent and documented |
| Documentation accuracy | **Partial** | Describes uncommitted code |

---

## 6. Recommended fixes, in order

1. Restore/re-apply the Phase 6 `run_pipeline.py` (recoverable from the transcript path in C1), commit it, and regenerate every report from it.
2. Make the sampler use `attack_indices_for_split(..., "eval")` for both phases; make clean evaluation use the validation parity split only.
3. Wire in the clamp, untrained, DeepFool, best-of-k, refinement, and kNN rows, plus `--seed`, `--eps`, `--retries`, `--train_epochs`, and `--deepfool_max_eps`.
4. Implement metadata write and read with a single budget source; remove hard-coded `max_eps=2.0`.
5. Pass explicit checkpoint names/directories per mode; never let manifold training write `wgan_generator.pt`.
6. Add CLI-level regression tests (tiny `evaluate_all`, assert the full row set and evasion), a metadata round-trip test, and a sampler-disjointness test. Load trained checkpoints in tests.
7. Reconcile docs with code, fix the pool-mix and latency issues, add `requirements.txt`, and move `NUM_PRE_EVADED`/`TOTAL_ATTACKS` into threshold metadata.

---

## 7. Verification appendix

Commands used (outputs directed to `/tmp`; project artifacts untouched):

```bash
source .venv/bin/activate
pytest -q                                   # 28 passed
flake8 src tests run_pipeline.py scripts    # exit 0
black --check src tests run_pipeline.py scripts

python run_pipeline.py --help
python run_pipeline.py --mode train_wgan --eps 2.0 --seed 42 --train_epochs 10
python run_pipeline.py --mode evaluate_clean --output_dir /tmp/...
python run_pipeline.py --mode evaluate_all --samples 100 --attack_phase both --output_dir /tmp/...
python run_pipeline.py --mode evaluate_all --samples 1000 --attack_phase scanning --output_dir /tmp/...
python run_pipeline.py --mode fgsm_sweep --attack_phase flood --samples 1000 --output_dir /tmp/...
```

Additional checks (temporary scripts in `/tmp`, since deleted):

- Eval-split sampler reproduction with `seed_everything(42)` — matched committed Phase A exactly.
- Training-pool overlap and pool-composition measurement.
- Checkpoint holdout scoring, five independent draws per checkpoint.
- Untrained-generator control scoring on both regimes.
- One-epoch manifold training into a temp directory to observe checkpoint names.
- Data shape/label/finiteness checks.
- Direct inspection of `AfterImage.py` / `netStat.py` / `FeatureExtractor.py` to validate the MTU/IAT index map.

Final repository state: `git status` unchanged except the pre-existing untracked screenshot `bdd514ab-f786-4dcc-b6dc-28c17dfd069e.png` (this report was added after the check).

---

## 8. Remediation record (post-review)

| Finding | Status | Change |
|---|---|---|
| C1 missing CLI | **Fixed** | `run_pipeline.py` restored from the Phase 6 transcript (40 edits, 6 black passes), deduplicated, and syntax/CLI-verified. Documented commands work. |
| C2 sampler leak | **Fixed** | Both phases draw from `attack_indices_for_split(..., "eval")`; candidate windows expand for small `--samples`. |
| C3 in-sample FPR | **Fixed** | Clean evaluation uses the validation parity split only; benign-kNN baseline restored. Report reproduces 86,853 / 0.0961% / 0.1383. |
| C4 metadata | **Fixed** | Training writes `*.meta.json`; `_load_generator` reads it; hard-coded `max_eps=2.0` removed. |
| C5 retry objective | **Fixed** | Clamp, untrained, DeepFool, best-of-k, refinement, and kNN rows all wired. |
| H1 checkpoint clobber | **Fixed** | Mode-specific generator/critic names; `--checkpoint_dir` added. Verified with a one-epoch temp run. |
| H2 ignored train args | **Fixed** | `--threshold_config`, `--model_path`, `--eps`, `--seed`, `--train_epochs`, `--train_batch_size`, `--checkpoint_dir` passed through. |
| H3 pool mix | **Fixed** | Detection applied per regime, then an actual 50/50 post-detection composition is enforced and printed. |
| H4 tests | **Fixed** | Added `tests/test_cli.py` (CLI flags, sampler disjointness, metadata round-trip, trained checkpoint evasion, full-row smoke) and `tests/test_kitnet.py`; theta loaded from config. 37 tests pass. |
| H5 holdout reuse | **Fixed** | Selection and test are disjoint halves; final ASR is on the untouched half. |
| H6 unused config | **Fixed** | Config keys now supply CLI defaults; dead keys removed. |
| H7 hard-coded population | **Fixed** | Read from `threshold_config.yaml` attack metadata into evaluator attributes. |
| H8 latency/small samples | **Fixed** | Latency scaled by `1000 / n`; scanning window floor 10k; explicit shortage warning. |
| H9 missing files | **Fixed** | Added `requirements.txt` and `tests/test_kitnet.py`. |
| H10 stale docs | **Fixed** | Test counts, Phase 4 epochs, Phase 6 manifold telemetry, review statuses, and this record updated. |

Verification after remediation:

```text
pytest -q                        37 passed
flake8 src tests run_pipeline.py scripts    exit 0
black --check src tests run_pipeline.py scripts   clean
evaluate_all --samples 1000      initial RMSE 5.3381; WGAN box 1.60% / 4.1925
evaluate_clean                   86,853 samples; FPR 0.0961%; kNN 0.1383
one-epoch manifold to /tmp       wgan_manifold_generator.pt + critic + meta, eps 1.5
```
