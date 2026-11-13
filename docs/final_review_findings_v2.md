# Final Review (Round 3) — Kitsune NIDS vs. Polymorphic WGAN

**Date:** 2026-11-13
**Reviewer:** Final reviewing agent
**Scope:** HEAD `f9435f6` plus the working tree — all source (`src/`, `scripts/`, `run_pipeline.py`, `Kitsune-py/`), tests, configs, docs, reports, and artifacts.
**Constraint:** Read-only review. Every experiment wrote to `/tmp`; the temporary scripts and outputs were deleted afterwards. The only repository change is this report. `git status` otherwise shows only the pre-existing untracked `bdd514ab-f786-4dcc-b6dc-28c17dfd069e.png`.

> **Remediation status (2026-11-13): all findings fixed or explicitly accepted.**
> The Cython estimator was brought to bit parity and covered by a test; the
> holdout was moved to the unused `%3 == 2` partition; DeepFool/PGD labels are
> parameter-driven; zero guards and paths were added; small artifacts are
> committed; layout docs and ER labels corrected. See section 7 below and
> `docs/detailed/phase_6_detailed.md` section 17.

---

## 1. Verdict

**The deliverable is genuine, honest, and reproducible. No critical or high-severity defect survived verification.**
Every committed benchmark artifact reproduces exactly, including the threshold configuration, which came out byte-identical. The headline attributions are real and seed-robust. The scientific caveats already in the README (the manifold attack is unbudgeted; the ε = 2 WGAN is weak) are accurate and stand.

The remaining problems are: one advertised opt-in feature is numerically broken, documentation numbers have drifted from the committed report, the reported benchmark slightly reuses the model-selection holdout, and there are edge-case/robustness and hygiene gaps. None invalidate the delivered results.

Severity scale: **Critical** = invalidates a deliverable/reproducibility; **High** = invalidates a comparison or headline property; **Medium** = methodology/reporting weakness; **Low** = robustness/hygiene.

---

## 2. What was verified as working

- **Exact reproduction.**
  - `python run_pipeline.py --mode evaluate_all --samples 1000 --attack_phase scanning --seed 42` and the same with `--attack_phase flood` reproduce every committed RMSE, ASR, ER, mean-L∞ and kNN value in `reports/evaluation_table.md`; only the single-run latency column varies.
  - `--mode evaluate_clean` reproduces `reports/clean_detection_metrics.json` field-for-field: 86,853 samples, FPR 0.0961%, confusion matrix, ROC-AUC 0.9418255, kNN baseline 0.1383412.
  - `--mode pgd_attack --attack_phase flood --samples 1000` reproduces every value in `reports/pgd_benchmark.json`, including the 100% manifold ASR/ER correction.
  - `--mode fgsm_sweep --attack_phase flood --samples 1000` reproduces `reports/fgsm_tradeoff.csv` **byte-identically**.
  - `python src/defender/eval_threshold.py` reproduces `data/processed/threshold_config.yaml` **byte-identically**; the file was backed up and restored.
- **Quality gates.** `pytest -q` → 36 passed. `flake8 src tests run_pipeline.py scripts` → exit 0. `black --check src tests run_pipeline.py scripts` → 27 files unchanged.
- **Data integrity.** `mirai_features.npy` is `(764137, 100) float32`; labels are `{0, 1}`; first attack index is 121,621; 642,516 attacks and 121,621 benign; zero benign packets after attack onset; every value finite.
- **Reproducible extraction.** Re-running the pure-Python extractor over the same packets is bit-identical to the committed feature matrix (max absolute difference 0 after float32).
- **Splits and threshold.**
  - Benchmark packets vs WGAN training split: **0** overlap.
  - Selection holdout vs untouched test half: 0 overlap.
  - Benign calibration vs validation: disjoint.
  - θ is genuinely the 99.9th percentile of the 33,311 calibration packets (0.4559856).
  - Validation FPR is genuinely 32/33,310 (0.0961%).
- **Artifacts match their summaries.** Reconstructing the test holdout exactly as `load_holdout(block=(0.5, 1.0))` gives initial mean RMSE 14.281322479248047 — identical in both committed summaries. The additive checkpoint yields mean 13.10328, ASR 0%, and the manifold checkpoint 0.03966, ASR 100%, matching `13.1033 / 0.0` and `0.03975 / 99.999…`. The reports are not fabricated.
- **Attribution is seed-robust.** Over five seeds on 200 detected flood packets: trained manifold 100% ASR at RMSE 0.0383–0.0391; envelope clamp 100% at 0.1098; untrained generator 0% at RMSE 10.4–13.5.
- **Mode collapse is real and measured.** Ten independent additive draws on 32 flood packets give mean RMSE 7.7818–7.7827, standard deviation **0.0003**, so the near-zero best-of-k gain is a property of the generator, not the retry code.
- **The network learned more than a projection.** The trained manifold output has normalised mean 0.193 / median 0.086 / p95 0.725 versus benign 0.173 / 0.045 / 0.766. A trivial "set every feature to `norm_min`" attack scores 0% ASR at RMSE 0.557, compared with 100% at 0.038 for the trained generator.
- **Operational behavior.**
  - `--eps` is written to `*.meta.json` and read back by `_load_generator`; evaluation honours 1.5 as trained.
  - One-epoch training is bit-reproducible (identical loss curve and holdout metrics).
  - Manifold training writes only `wgan_manifold_generator.pt`, `wgan_manifold_critic.pt` and its metadata.
  - Training correctly refuses to clobber the additive checkpoint.
  - CPU fallback (`--device cpu`) completes.
  - The Cython opt-in imports.
- **Constraint map correctness.** Verified against `Kitsune-py/netStat.py:74-107` and `AfterImage.py`: covariance indices `[20,27,34,41,48,70,77,84,91,98]`, PCC `[…+1]`, size means `SIZE_MEAN_INDICES`, jitter means/stds `IAT_STAT_INDICES` all match the real 100-D layout (MI 0–14, HH 15–49, jitter 50–64, HpHp 65–99). The model's 16 micro-AE clusters cover all 100 indices with no duplicates.

---

## 3. Findings

### F1 — Medium: The advertised Cython extractor is numerically broken and yields `NaN` scores

**Where:** `Kitsune-py/netStat.py:10-13`, `Kitsune-py/FeatureExtractor.py:3-15`, `Kitsune-py/AfterImage_extrapolate.pyx:219-276`, README:58-60, `docs/detailed/phase_6_detailed.md` section 8, `docs/final_review_findings.md:147`.

`KITSUNE_USE_CYTHON=1` selects the compiled extractor, but its covariance estimator diverges from the pure-Python one:

- Pure-Python extraction of 300 packets at offset 0 and 300 at offset 60,000 is **bit-identical** to the committed matrix (max diff 0 after float32).
- Cython extraction differs in radius/covariance/PCC by up to **2.68e27** at offset 0 and **9.16e31** at offset 60,000. PCC reaches ~1e26 although it must lie in [-1, 1].
- Scoring Cython features with the committed model gives **`NaN` RMSE for 44/300** mid-stream packets and flags **17%** of benign packets versus 0.67% for pure Python.

**Cause:** `incStat_cov.update_cov` in the `.pyx` uses an unbounded Lagrange `extrapolator` (`extrapolator`, `.pyx:754-800`) and never decays the peer stream; the pure-Python implementation (`AfterImage.py:159-196`) uses a bounded `lastRes` residual product.

**Impact:** the opt-in cannot be used interchangeably with the committed model, and no test covers it. The default path and all committed results are unaffected, so severity is Medium rather than High.

### F2 — Low/Medium: The benchmark reuses the model-selection holdout

**Where:** `src/pipeline/feedback_loop.py:225-245`; `run_pipeline.py:300-314`; unused `%3 == 2` partition at `src/utils/data_splits.py:14-16`.

`load_holdout` takes selection and test sets from the same eval split that `get_detected_attack_sample` later samples for the published table. Measured overlap with the current seed:

| Published set | In selection holdout | In untouched test half |
|---|---:|---:|
| Phase A (scanning), 1,000 pkts | **27** | 0 |
| Phase B (flood), 1,000 pkts | **10** | 3 |

Training overlap is zero and selection/test do not overlap each other, so most of the split claim holds. But Phase A/B are presented as "holdout packets", and ~2.7% of Phase A contributed to checkpoint selection. Using the deliberately unused `(index − ATTACK_START) % 3 == 2` partition for selection/test would make the published set fully untouched.

### F3 — Medium: Documentation numbers are stale and internally inconsistent

The committed report and artifacts are correct; several docs are not:

- `docs/overview/phase_5_overview.md:62-64`, `docs/detailed/phase_5_detailed.md:140-142`, `docs/detailed/phase_6_detailed.md:477-479`, and `docs/overview/phase_6_overview.md:98-100` report WGAN box best-of-k **1.60% ASR / 13.52% ER**; the report says **1.80% / 13.43%**. An ER of 13.52% is arithmetically impossible at 1.60% ASR: `(76123 + 566393·0.016) / 642516 = 13.26%`.
- The refine ER is quoted as 15.29% where the report says 15.20%.
- Prose quotes manifold RMSE **0.078 / 0.037** (`phase_6_overview.md:37`, `phase_6_detailed.md:515`, `docs/overview/phase_5_overview.md:104`, `docs/detailed/phase_5_detailed.md:164-165`, `docs/overview/phase_4_overview.md:36`, `docs/review_findings.md:73`) while the artifacts are **0.0647 / 0.0391**.
- `docs/overview/phase_6_overview.md:67` says tests were strengthened "from 22 to 28"; the suite has 36.
- `docs/detailed/phase_3_detailed.md:132` still says the results "confirm the necessity" of WGAN-GP, which Phase 6 itself qualified.

### F4 — Low: DeepFool CLI flags are not reflected in the table

**Where:** `run_pipeline.py:810-828`.

`--deepfool_steps 5 --deepfool_max_eps 0.5` correctly changes behavior (observed mean L∞ 0.5000), but the row still prints `50 iters, eps <= 2.0 * s`. PGD's ε is hard-coded at 2.0 (`run_pipeline.py:795`) with no CLI flag, although the row is also hard-coded to `eps = 2.0`.

### F5 — Low: Edge-case crashes

- `--samples 0` → `ValueError: need at least one array to concatenate` (`src/pipeline/evaluator.py:59`, reached from `run_pipeline.py:324-329`).
- `--batch_size 0` → `ValueError: range() arg 3 must not be zero` (`src/pipeline/evaluator.py:54`).

### F6 — Low: Data paths are partly hard-coded

**Where:** `run_pipeline.py:219,300,362`.

`load_labels()` is always called on its default `data/raw/mirai_labels.csv`; `--config` does not redirect evaluation-side paths, and `--features_path` never reaches the training loop (which reads the config's own paths). Relocating a dataset leaves labels pointing at the old tree without an error.

### F7 — Low: The benchmark depends on untracked artifacts

**Where:** `.gitignore:3` (ignores all of `data/`).

`git ls-files` contains no `threshold_config.yaml`, no `*.meta.json`, no model and no checkpoints. A fresh clone cannot reproduce the benchmark without roughly 30 minutes of extraction and retraining. The README documents data preparation but does not say that the WGAN artifacts are also untracked.

### F8 — Low: Layout and documentation drift, broken references

- `AGENTS.md` section 3 and `docs/implementation_plan.md:82-83` list `src/defender/afterimage.py` and `src/defender/kitnet.py`, which do not exist.
- `docs/implementation_plan.md:52,99` says only `K.AnomDetector` is serialized; `scripts/extract_and_train.py:131` pickles the whole `KitNET`. Harmless (the detailed doc clarifies) but the plan is stale.
- `docs/detailed/phase_0_detailed.md:151` links the absent `Note to myself.pdf`.
- Docs reference `img/15-17`, but `img/` is gitignored, so those references also break in a clone.

### F9 — Low: Dead and duplicated code

- `run_evaluate_clean` computes `benign_knn_baseline` twice (`run_pipeline.py:392-399`).
- `evaluate_perturbation_sweep` and `aggregate_tradeoff_curve` are unused by the pipeline.
- `TTL_MIN` / `TTL_MAX` remain unused (documented limitation).
- Config keys `device` and `batch_size` in `configs/kitsune_config.yaml` are unused.

### F10 — Low: "Overall Evasion Rate" is a per-phase extrapolation

**Where:** `src/pipeline/evaluator.py:122-129`; header at `run_pipeline.py:716`.

`evaluate_evasion` extrapolates the current phase's regime-specific ASR across the whole detected population. Phase A and Phase B therefore report two different "overall" estimates for the same attack (for the box single draw: 13.26% and 11.85%), not a single overall figure. The label overstates what is measured.

---

## 4. Requirement scorecard

| Requirement | Status | Evidence |
|---|---|---|
| Automated evaluation pipeline against a NIDS | **Met** | 5 CLI modes; every report reproduces |
| Calibrate and test an evasion threshold | **Met** | Disjoint calibration; out-of-sample FPR genuine |
| Offensive script mutates until bypass | **Partial** | Best-of-k exists but adds 0.2 points; box/flood stays 0% |
| WGAN-GP as the weapon | **Implemented; strong only when unbudgeted** | 100% manifold, 0% untrained, weak at ε = 2, documented |
| Physical/domain validity | **Partial** | MTU/IAT/PCC/sign enforced; TTL absent; feature-level only |
| Reproducible, artifact-backed results | **Met for reports; needs untracked data** | All five committed artifacts reproduced |
| Documentation accuracy | **Partial** | F3, F8 |
| Quality gates and AGENTS conventions | **Pass with layout drift** | 36 pass; lint clean; F8 |

---

## 5. Recommendations, in order

1. Fix or disable the Cython opt-in: port the pure-Python residual covariance into the `.pyx` (or remove the Lagrange `extrapolator`) and add a parity test asserting max absolute difference `< 1e-6` over a packet slice.
2. Draw selection/test from the `%3 == 2` partition so published packets are fully untouched; alternatively exclude the 27/10 shared packets from the benchmark.
3. Regenerate the stale doc tables from `reports/evaluation_table.md`, correct the ER arithmetic, update the test count, and soften the Phase 3 "necessity" claim.
4. Derive the DeepFool/PGD constraint labels from the actual parameters and expose a PGD ε flag.
5. Guard `--samples >= 1` and `--batch_size >= 1` (or clamp them).
6. Thread labels/features/model paths through `--config` and add a `--labels_path`.
7. Commit the small reproducibility artifacts (`*.meta.json`, `threshold_config.yaml`) and state explicitly that checkpoints require training.
8. Add `tests/test_extractor_parity.py`; remove the duplicated kNN block and dead helpers.
9. Relabel the ER column as "ER (extrapolated, this regime)".

---

## 6. Limitations of this review

- One dataset (Mirai) and one trained detector.
- CUDA host; single-run latencies carry the documented ±20% variance.
- No packet reconstruction or joint feature-consistency analysis was performed.
- Adversarial claims were checked only in the project's white-box, feature-space setting.
- Findings are limited to the current seed for the overlap measurement in F2; the count is deterministic for `--seed 42`.

---

## 7. Remediation record (round 3)

| Finding | Status | Change |
|---|---|---|
| F1 broken Cython | **Fixed** | Ported the pure-Python residual covariance into `incStat_cov.update_cov`, squared variances in `incStat.radius`, removed the extrapolator path, rebuilt. Pure/Cython are bit-identical over 300 packets at offset 0 and 100 after a 10k warm-up; `tests/test_extractor_parity.py` asserts <1e-6 and finite output. |
| F2 selection overlap | **Fixed** | Added the `%3 == 2` holdout partition for selection/test; published eval (`%3 == 1`) is now fully untouched. Retrained both checkpoints and regenerated reports. |
| F3 stale docs | **Fixed** | ER values corrected to 13.26/11.85 (single draw), 13.43/11.85 (best-of-k), 15.20/11.85 (refine); manifold prose and tables set to 0.0779/0.0475; test count 36; Phase 3 necessity softened. |
| F4 label drift | **Fixed** | `--pgd_eps` added; PGD alpha derives from it; PGD and DeepFool labels are built from the actual parameters. |
| F5 zero crashes | **Fixed** | `samples`, `batch_size`, `train_batch_size`, `retries`, `refine_steps`, and `deepfool_steps` clamp to ≥1; regression test. |
| F6 hard-coded paths | **Fixed** | `--labels_path` added and threaded through evaluation/training; `--features_path` reaches the feedback loop; config paths can be overridden. |
| F7 untracked artifacts | **Fixed** | `threshold_config.yaml` and `*.meta.json` unignored and committed; README states checkpoints require regeneration. |
| F8 layout/doc drift | **Fixed** | AGENTS tree and plan updated; pickle wording corrected; missing PDF link replaced; review images copied to `docs/review_images/` and references updated. |
| F9 dead code | **Fixed** | Duplicate kNN block removed; unused `evaluate_perturbation_sweep` and `aggregate_tradeoff_curve` + test removed; unused kitsune config keys removed. TTL constants remain documented as unused. |
| F10 ER label | **Fixed** | Benchmark header is now `ER (this regime, extrap.) (%)`; docs explain it is a per-regime extrapolation. |

Verification: 37 tests pass, flake8/black clean, every report regenerated from
the new holdout split, and all five committed artifacts reproduce from the CLI.
