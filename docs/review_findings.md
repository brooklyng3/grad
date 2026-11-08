# Independent Review: Kitsune NIDS vs. Polymorphic WGAN Pipeline

**Date:** 2026-11-08
**Reviewer:** Claude Code
**Scope:** Full repository — all docs (`docs/`, including `docs/detailed/`), all source (`src/`, `scripts/`, `run_pipeline.py`, `tests/`), configs, reports, and artifacts.
**Constraint:** No repository source or artifact was modified during the review. Test and pipeline outputs were written to `/tmp`; checkpoint mtimes were verified unchanged.

> **Remediation status (2026-11-08): all findings addressed.** The fixes, reasoning, diffs, and post-fix results are documented in `docs/overview/phase_6_overview.md` and `docs/detailed/phase_6_detailed.md`. The numbers below are the pre-fix state and are retained as the review record.
>
> **Post-final-review correction (2026-11-09):** the Phase 6 `run_pipeline.py`
> was omitted from commit `cf27792` even though the other fixes landed. It has
> been restored and extended; the sampler, clean split, metadata, retry rows,
> checkpoint naming, population constants, latency scaling and tests are now in
> the delivered file. See `docs/final_review_findings.md` section 8 and
> `phase_6_detailed.md` section 16. The "Fixed" statuses below now hold for the
> committed code, not only the working tree.

| Finding | Status |
|---|---|
| F1 manifold result is a clamp | Fixed — clamp-only and untrained-generator ablation rows added; attribution corrected |
| F2 epsilon mismatch / overfitting | Fixed — shared ε persisted in checkpoint metadata; disjoint stratified training split; holdout model selection; `n_critic` executed |
| F3 no retry-until-bypass loop | Fixed — best-of-k retry attack added and benchmarked |
| F4 methodology weaknesses | Fixed — separate calibration/validation splits, seeds, unguarded tests, evasion assertions, 36 tests |
| F5 declarative physical constraints | Fixed — MTU/IAT bounds enforced; TTL limitation documented |
| F6 stale/missing artifacts | Fixed — reports regenerated, README and Phase 6 docs added, links repaired |
| F7 Cython unused | Fixed — opt-in `KITSUNE_USE_CYTHON=1` path wired; default documented |
| F8 unused code/config | Fixed — DeepFool benchmarked, `n_critic` used, config wired |

---

## 1. Verdict

**The evaluation harness is real and largely works. The central scientific claim about the WGAN does not hold up.**

The project delivers a functioning data pipeline, a numerically faithful differentiable defender, correct threshold calibration, decoupled evasion metrics, and a reproducible CLI benchmark. 22/22 tests pass; `flake8` and `black --check` are clean; the headline benchmark table reproduces exactly.

However, the headline result — "Polymorphic WGAN-GP (Manifold) — 100.00% conditional ASR" — is produced almost entirely by a `clamp()` into the defender's benign normalisation envelope, not by the learned generator. An **untrained**, randomly initialised generator achieves the same 100% ASR and lower RMSE than the trained one. The only perturbation-budgeted WGAN attack is non-functional on held-out data and is documented using an in-sample measurement. There is also an unreported train/evaluation epsilon mismatch that makes the documented budget label incorrect.

The project therefore demonstrates that **white-box envelope clamping evades Kitsune**, not that a polymorphic WGAN does. The docs should be re-centred on that distinction.

---

## 2. What the review verified

### 2.1 Positive findings

- **Data integrity:** `data/processed/mirai_features.npy` is `(764137, 100) float32`; labels are `(764137,)` and align 1:1. The first attack index is 121,621; the preceding 121,621 packets are all benign and there are no benign packets after the attack onset. All sampled feature values are finite.
- **Defender numerical parity:** `src/defender/torch_kitnet.py` faithfully mirrors the NumPy KitNET implementation (normalisation, encoder/decoder, output-layer flow). The parity test in `tests/test_torch_kitnet.py` passes.
- **Constraint index map is correct:** AfterImage's per-window 7-tuple is `[weight, mean, std, radius, magnitude, covariance, pcc]`; the covariance/PCC indices in `src/utils/constraints.py` are correct.
- **Evaluation arithmetic:** `ASR_cond` and `ER_total` are correctly decoupled and extrapolated in the current evaluator (`src/pipeline/evaluator.py:92-99`). Distortion norms are computed in normalised space.
- **Reproducibility of the committed table:** `run_pipeline.py --mode evaluate_all` reproduced `reports/evaluation_table.md` exactly, including every reported RMSE and L∞ value.
- **Quality gates:** `pytest` → 22 passed; `flake8 src tests run_pipeline.py scripts` → exit 0; `black --check` → unchanged.
- **The clamp result is genuine at full scale:** scoring all 642,516 attack packets through the clamp path evaded all 566,393 detected attacks; maximum post-attack RMSE was 0.305, below θ = 0.453.

### 2.2 Artifacts and environment confirmed

- Trained defender `data/processed/kitsune_mirai_model.pkl` (16 micro-AEs, `n_trained = 55,001`).
- Threshold metadata `data/processed/threshold_config.yaml` (θ = 0.4534189701080322).
- Generator checkpoints `data/attack_samples/wgan_manifold_generator.pt` and `wgan_generator.pt`.
- `torch.cuda.is_available()` is `True` on the host.

---

## 3. Findings

Severity: **Critical** = invalidates a headline claim; **High** = invalidates a baseline or comparison; **Medium** = methodology/reporting weakness; **Low** = hygiene.

### F1 — Critical: The 100% "WGAN manifold" result is a clamp, not a GAN

**Status:** Fixed, then strengthened after a second review round (`img/15-17`).
The post-hoc `clamp` was removed from the manifold forward; the generator now
learns the target step itself. Post-fix numbers: untrained 0% ASR (RMSE
5.63/15.05), trained 99.9–100% (RMSE 0.078/0.037), which **beats the clamp
baseline** (0.114/0.109). Benign-kNN distance also improves (0.67/0.62 vs clamp
0.91/0.87). The clamp row remains for attribution. See `phase_6_detailed.md`
section 15.

**Where:** `src/attacker/generator.py:91-96`

```python
if active_mode == "manifold":
    dynamic_budget = torch.clamp(x_mal_norm, min=budget)
    delta_norm = delta_raw * dynamic_budget
    x_adv_norm = torch.clamp(x_mal_norm + delta_norm, min=0.0, max=1.0)
```

For flood features whose normalised value is around 35, the final clamp to `[0, 1]` saturates nearly every feature regardless of the network output. The learned tanh perturbation barely survives.

**Evidence (500 detected packets per regime):**

| Attack | Scan ASR | Scan mean RMSE | Flood ASR | Flood mean RMSE |
|---|---:|---:|---:|---:|
| Trained manifold generator | 100% | 0.229 | 100% | 0.232 |
| **Trivial clamp only** | **100%** | **0.130** | **100%** | **0.110** |
| Untrained generator, seed 0 | 100% | 0.157 | — | — |
| Untrained generator, seed 1 | 100% | 0.189 | — | — |
| Untrained generator, seed 2 | 100% | 0.211 | — | — |

The trained generator is *worse* than the clamp and approximately equal to random initialisation. Output varies across random `z` by only about 0.01 normalised units. PGD-manifold at 20 vs 50 steps gives 0.0902 vs 0.0899 RMSE, showing that iteration adds nothing either.

**Impact:** The following claims are overstated:

- `docs/detailed/phase_5_detailed.md:133` — "a **1,237× speedup** — proving the decisive operational necessity of generative perturbation in real-time adversarial ML."
- `docs/overview/phase_5_overview.md:81` — "deep generative mutation is the only practical mechanism for real-time, wire-speed evasion."

A white-box clamp requires no generator forward pass. The result proves that **single-pass envelope projection** is fast and effective; it does not establish that a GAN is needed.

The L∞ column also shows the manifold attack is **unbudgeted**: mean 8.8 normalised units in scanning and 35.0 in flood, versus the ε = 2.0 rows it is compared against. The table labels the constraint only as "Envelope Shaping", but the comparison invites a false like-for-like reading.

---

### F2 — Critical: The budgeted WGAN is broken by a train/evaluation epsilon mismatch and severe overfitting

**Status:** Fixed for consistency, but the attack remains budget-limited.
Training and evaluation share `--eps`, persisted in `*.meta.json`; the split is
disjoint and stratified; the best epoch is selected by held-out ASR; `n_critic`
is executed. ASR stays 1.60% scanning / 0.00% flood. Independent draws had no
variance (conditional mode collapse), so a feedback-refinement attack was added:
10 steps raise scanning to 3.80% and cut flood RMSE to 5.19, but not to the
reviewer's predicted >80% (aggregate L∞ ≈ 10.4/19.7, not ε = 2).

**Where:**

- Training uses `eps_linf` from `configs/wgan_config.yaml:14`, which is **0.05** (`src/pipeline/feedback_loop.py:79,212,229`).
- Evaluation constructs the generator with `max_eps=2.0` and calls `mutate_raw` without an epsilon, so it evaluates at **2.0** (`run_pipeline.py:705-723`).
- The tables label the attack "ε = 2.0" (`docs/detailed/phase_4_detailed.md:78`, `docs/detailed/phase_5_detailed.md:116,129`).

The generator was trained at a 40× smaller perturbation scale than it was tested at.

**Additional training defects:**

- `n_critic: 5` is loaded but never used (`feedback_loop.py:81`); the loop performs one critic update per generator update (`feedback_loop.py:200-248`).
- Training data is the first 20,000 attack packets — 3% of the attack population, all early scanning (`feedback_loop.py:137-143`). Volumetric flood is out-of-distribution by construction.
- The "holdout" final evaluation is **in-sample**: `x_mal_raw[:1000]` is sliced from the training tensor loaded at `feedback_loop.py:148` and evaluated at `feedback_loop.py:290-302`. The doc calls it "1,000 Holdout Detected Attacks" (`phase_4_detailed.md:68`).

**Evidence:**

| Test | ASR (conditional) |
|---|---:|
| Shipped `wgan_generator.pt`, every epsilon 0.05–2.0, including training-window samples | **0.00%** |
| Fresh retrain at ε = 2.0, in-sample first 1,000 detected | **6.20%** |
| Fresh retrain at ε = 2.0, held-out scanning (attacks 20k–30k) | **0.20%** |
| Fresh retrain at ε = 2.0, held-out flood | **0.00%** |

The documented 6.70% is not reproducible from the committed checkpoint. When the config bug is removed, the in-sample figure is reproduced (~6.2%) but collapses on the adjacent holdout window. The budgeted generator is effectively memorising the first 20,000 attacks.

**Impact:** The "WGAN-GP (Box)" rows are not valid held-out measurements. The Phase 4 comparison matrix mixes an in-sample WGAN result with otherwise held-out baselines.

---

### F3 — High: There is no per-sample "mutate until bypass" loop

**Status:** Fixed. `wgan_best_of_k` draws fresh latents per sample until
`RMSE <= θ`, retains the best draw, and reports mean draws per packet in the
benchmark table.

**Where:** `src/pipeline/feedback_loop.py:190-256` trains the generator with a hinge loss, but each sample receives exactly one generator pass. `src/attacker/generator.py:104-137` (`mutate_raw`) is single-shot.

The screenshot objective and the plan mermaid (`docs/implementation_plan.md:30-37`, `A4 -->|No: Detected| A2`) imply iterative mutation until `RMSE ≤ θ`. Only PGD and DeepFool actually iterate; the WGAN does not.

**Impact:** The project objective "continuously mutates attack traffic until it successfully bypasses" is only partially met. The shipped budgeted WGAN never bypasses at all; the manifold variant bypasses in one pass because of the clamp.

---

### F4 — High: Evaluation methodology weaknesses

**Status:** Fixed. Benign calibration/validation are disjoint (FPR 0.096%
out-of-sample); attack train/eval splits are disjoint; all runs are seeded;
guarded test assertions removed; evasion assertions added (36 tests pass).

1. **FPR is in-sample by construction.** θ is the 99.9th percentile of 66,621 benign packets, then FPR is counted on those same packets (`src/defender/eval_threshold.py:91-125`; `run_pipeline.py:245-248`). FPR = 0.1006% is arithmetic, not independent validation. `tests/test_threshold.py:28` asserts FPR stays near 0.1%, which it must.
2. **Phase A is not a holdout for the WGAN.** The scanning evaluation range is the first 10,000 attacks (`run_pipeline.py:198-205`), entirely inside the 20,000-attack WGAN training pool.
3. **ER_total is extrapolated from 1,000 samples.** The formula is correct and I verified 100% over the full population for the clamp, but the reported WGAN ER itself was never measured across the detected population.
4. **No random seeds anywhere** except one NumPy fixture (`tests/test_torch_kitnet.py:23`). Evaluation draws fresh `z` each run; ASR varied 6.1–6.7% across trials.
5. **No per-epoch holdout evaluation or model selection** in `train_loop`; the reported training ASR is computed on the batches being trained.
6. **`test_evaluator_subsample_extrapolation`** (`tests/test_pipeline.py:62-79`) guards its assertions with `if metrics_100[...] == 100.0` and can silently no-op.
7. **No test asserts evasion.** `test_polymorphic_generator` (`tests/test_attacker.py:113-118`) uses a fresh, untrained generator and only checks shape/budget. The suite passes with a useless generator.

---

### F5 — Medium: Physical-realism constraints are largely declarative

**Status:** Fixed partially. Packet-size mean channels are clamped to MTU
(≤1500) and IAT channels to ≥0. TTL remains unenforceable because AfterImage
does not extract it; documented.

- `MTU_MAX_BYTES`, `MIN_PACKET_BYTES`, `MIN_IAT`, `TTL_MIN`, `TTL_MAX` are declared at `src/utils/constraints.py:17-21` and **never referenced anywhere else**.
- Enforced constraints are only: unsigned features ≥ 0, PCC in [-1, 1], covariance unconstrained. The AGENTS-mandated MTU/IAT/TTL rules are not mapped onto the feature vector.
- Attacks mutate 100 aggregate statistics. Nothing reconstructs packets or checks that the mutated feature vector is jointly consistent (e.g. mean/variance/rate coupling across damped windows).
- The manifold attack additionally requires the defender's internal `norm_min` / `norm_max` buffers (white-box setting), which is fine but should be stated as a limitation.

---

### F6 — Medium: Stale and missing artifacts

**Status:** Fixed. All reports regenerated (PGD manifold ER now 100%);
`reports/*.csv` unignored; WGAN summaries and training metadata produced;
README added; implementation-plan links repaired.

1. **`reports/pgd_benchmark.json` is stale.** Manifold rows report `overall_evasion_rate_pct = 12.003` alongside 100% ASR (lines 53 and 76) — one raw success count rather than population extrapolation. The JSON predates the extrapolation fix and was never regenerated; `reports/evaluation_table.md` correctly says 100.00%.
2. **`reports/fgsm_tradeoff.csv` is stale** (mean RMSE 13.329 vs current 13.500 at ε = 0.01). `*.csv` in `.gitignore` means the file is untracked.
3. **No WGAN training-summary JSONs exist** despite `run_train_wgan` writing them; the Phase 4 telemetry is unsupported by artifacts.
4. **No `README.md`.** The implementation plan links to deleted `docs/phase_0_changes.md` and `docs/phase_0_5_changes.md`.
5. **Documentation inconsistencies:** Phase 3 reports a WGAN prototype at 53.12% (`phase_3_detailed.md:121`) while Phase 4 reports 6.70%; Phase 4 claims 15 epochs while the CLI runs 10.

---

### F7 — Medium: The Cython extension is never used

**Status:** Fixed. `KITSUNE_USE_CYTHON=1` selects `AfterImage_extrapolate` in
`netStat.py` and the extractor builds it on demand; pure Python remains the
documented default for the committed features.

`Kitsune-py/FeatureExtractor.py:12` imports `netStat`; `Kitsune-py/netStat.py:7` imports pure-Python `AfterImage`; `use_extrapolation = False` in `FeatureExtractor.py:5`. The compiled `AfterImage_extrapolate` extension is never imported.

The claim of "Cython-accelerated `AfterImage`" (`docs/overview/phase_1_5_overview.md:8`) is false. The measured 422 pkts/s extraction rate is consistent with pure Python.

---

### F8 — Low: Unused code and configuration

**Status:** Fixed. DeepFool is benchmarked; `n_critic` is used; config values
are wired to CLI flags; checkpoint metadata records the budget.

- `deepfool_attack` (`src/attacker/perturbations.py:162`) is never invoked by the pipeline; `run_pipeline.py:30` imports only FGSM and PGD. A named baseline is unbenchmarked.
- `n_critic` is dead config (see F2).
- `wgan_config.yaml` `max_iterations` and `step_size` are unused.
- `run_train_wgan` ignores `--threshold_config` and `--model_path`; `feedback_loop` hardcodes `data/processed/threshold_config.yaml`.
- The pickled model includes the feature-mapper (`FM`) state, not only detector weights; harmless, but the doc's "only `K.AnomDetector`" wording is imprecise.

---

## 4. Objective scorecard

| Objective (screenshot / `docs/implementation_plan.md`) | Status | Evidence |
|---|---|---|
| Automated evaluation pipeline against a NIDS | **Achieved** | CLI with 5 modes, metrics, reports; table reproduces |
| Calibrate and test an evasion threshold | **Achieved mechanically** | θ and ASR/ER code correct; FPR is in-sample (F4) |
| Offensive script continuously mutates until bypass | **Partial** | Train-time feedback only; no retry loop (F3); budgeted WGAN fails (F2) |
| WGAN-GP as the weapon | **Not demonstrated** | Manifold success is a clamp (F1); box overfits (F2) |
| Physical/domain validity of perturbations | **Partial** | Feature-level bounds only; MTU/TTL/IAT unused (F5) |
| Reproducible, artifact-backed results | **Partial** | Table reproduces; no seeds, stale/missing artifacts (F4, F6) |

---

## 5. Recommendations

Ordered by impact.

1. **Add a three-row ablation to the headline table:** clamp-only / untrained generator / trained generator. This makes the GAN's marginal contribution explicit. On current evidence it is approximately zero for manifold mode.
2. **Fix the budget semantics.** Either project to an ε·s ball after the envelope step, or rename the mode "unbudgeted envelope projection" and stop comparing it head-to-head with ε = 2.0 rows.
3. **Fix the train/eval epsilon mismatch.** Use one budget source, expose it on the CLI, remove the ambiguous config default, and retrain/evaluate at matching budgets.
4. **Train on a representative attack distribution** (or per regime) and always evaluate on disjoint indices. Reserve a fixed holdout split in code.
5. **Implement a retry/best-of-k loop** if "mutate until bypass" is the objective, and report attempts and distortion per success.
6. **Split benign calibration from benign test** so FPR is measured out-of-sample. Seed Python/NumPy/Torch and add deterministic evaluation.
7. **Regenerate committed reports** (`pgd_benchmark.json`, FGSM CSV) after metric fixes and commit the WGAN summary JSONs.
8. **Add regression tests that assert evasion** for each attack mode and load the trained checkpoints; fix the guarded assertions in the extrapolation test.
9. **Use or remove** DeepFool, `n_critic`, and the MTU/IAT/TTL constants. If packet-level realism is claimed, either map constraints to features explicitly or state the limitation.
10. **Update the docs** to remove the "generative necessity" claim until an ablation supports it, and add a README.

---

## 6. Reproduction appendix

Commands used (outputs redirected to `/tmp`; project artifacts untouched):

```bash
# Full test suite
source .venv/bin/activate && pytest -q
# 22 passed

# Lint/format
flake8 src tests run_pipeline.py scripts   # exit 0
black --check src tests run_pipeline.py scripts

# Reproduce the reported table exactly
python run_pipeline.py --mode evaluate_all --samples 1000 --attack_phase both \
  --output_dir /tmp/grad_reports

# Demonstrate stale pgd_benchmark ER values
python run_pipeline.py --mode pgd_attack --attack_phase flood --samples 1000 \
  --output_dir /tmp/grad_reports2

# Manifold generator vs clamp vs untrained generator (500 samples/regime)
# Full-population clamp verification: 566,393 / 566,393 detected attacks evaded,
# max adversarial RMSE 0.305 < theta
# In-memory retrain at eps=2.0: 6.20% in-sample, 0.20% held-out scan, 0.00% flood
```

### Key source references

| Claim | Location |
|---|---|
| Manifold clamp is the active mechanism | `src/attacker/generator.py:91-96` |
| Training epsilon 0.05 | `configs/wgan_config.yaml:14`; `src/pipeline/feedback_loop.py:79` |
| Evaluation epsilon 2.0 | `run_pipeline.py:705-723` |
| In-sample "holdout" evaluation | `src/pipeline/feedback_loop.py:148,290-302` |
| Training pool is first 20k attacks | `src/pipeline/feedback_loop.py:137-143` |
| `n_critic` unused | `src/pipeline/feedback_loop.py:81,200-248` |
| Single critic step per generator step | `src/pipeline/feedback_loop.py:200-248` |
| In-sample FPR calibration | `src/defender/eval_threshold.py:91-125` |
| Phase A sampling | `run_pipeline.py:198-205` |
| Stale ER values | `reports/pgd_benchmark.json:53,76` |
| Cython never imported | `Kitsune-py/FeatureExtractor.py:5,12`; `Kitsune-py/netStat.py:7` |
| DeepFool unused | `src/attacker/perturbations.py:162`; `run_pipeline.py:30` |
| Unused MTU/TTL/IAT constants | `src/utils/constraints.py:17-21` |
| Overstated generative claims | `docs/detailed/phase_5_detailed.md:133`; `docs/overview/phase_5_overview.md:81` |

---

## 7. Limitations of this review

- Only the Mirai dataset and the project's single trained defender were reviewed; no external dataset or second detector was tested.
- White-box attacks are the project's stated setting; the clamp's dependence on defender normalisation buffers is consistent with that setting.
- The in-memory retrain used 10 epochs to match the CLI default, not the 15 epochs claimed in Phase 4. It is sufficient to show the in-sample/holdout gap, but not a full hyperparameter study.
- Review results are stochastic where the code is unseeded; ASR figures varied slightly across repeated draws (e.g. 6.1–6.7% in-sample).
