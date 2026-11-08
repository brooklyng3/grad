# Phase 6 Summary: Review Remediation, Budget Consistency & Honest Attack Attribution

This document summarises the remediation pass triggered by the independent
review in `docs/review_findings.md`. Every finding was addressed; the full code
log, diffs, commands, and post-fix telemetry are in
`docs/detailed/phase_6_detailed.md`.

> **Final-review follow-up:** commit `cf27792` omitted `run_pipeline.py`. The
> file was recovered from the session transcript, deduplicated, extended with
> the remaining review fixes (sampler, split, metadata, retry rows, population
> constants, latency, checkpoint naming), and verified. See
> `docs/final_review_findings.md` section 8 and `phase_6_detailed.md` section
> 16.

---

## 1. Why this phase exists

The review confirmed that the harness worked but that its headline claim was
misattributed: the 100% "WGAN manifold" evasion was produced by a clamp into
the defender's benign min/max envelope, and an untrained generator achieved the
same result. The budgeted WGAN was also trained at ε = 0.05 while evaluated at
ε = 2.0, and its "holdout" score was computed on training samples.

Phase 6 makes the benchmark honest and re-establishes reproducibility.

---

## 2. Objectives achieved

1. **Attack attribution made explicit, and the generator now wins.** The
   post-hoc `clamp` was removed from the manifold forward, so the network must
   learn the target step instead of being forced into `[0,1]`. The table keeps
   three controls:
   - `Envelope Clamp Only` (no neural network, RMSE 0.114/0.109),
   - `WGAN-GP (Manifold, untrained)` (0% ASR, RMSE 5.63/15.05),
   - `WGAN-GP (Manifold, trained)` (99.9–100% ASR, RMSE **0.078/0.037**).

   The trained generator now beats the naive clamp on RMSE and on benign-kNN
   distance (0.67/0.62 vs 0.91/0.87; benign baseline 0.14).

2. **Budget consistency.** `--eps` is a single CLI value used for both training
   and evaluation and persisted in `*.meta.json` beside each checkpoint. The
   evaluation loader reads it back. The config default is now 2.0, matching the
   tables.

3. **Disjoint, representative splits.**
   - Attack packets: `(index − 121,621) % 3` — remainder 0 trains, remainder 1
     evaluates; never overlap.
   - Training samples half from the early scanning window and half spread over
     the full capture, so flood is represented.
   - Benign packets: parity split; even calibrates θ, odd measures FPR.

4. **Model selection on held-out data.** Every epoch scores a 1,000-packet
   disjoint holdout; the best checkpoint by holdout ASR is restored and saved.

5. **Retry-until-bypass.** A best-of-k generator loop draws fresh latents per
   sample until `RMSE ≤ θ` or the retry cap, retains the best draw, and reports
   mean draws per packet. A feedback-refinement variant feeds the current
   mutation back for K steps.

6. **WGAN-GP correctness.** `n_critic = 5` is now executed; the critic and
   generator optimisers alternate correctly.

7. **Methodology and reproducibility.** All RNGs are seeded (`--seed`, default
   42). Benign FPR is measured out-of-sample (0.096% on validation). Tests were
   strengthened from 22 to 28 and now assert evasion, best-of-k, and refinement
   invariants. A benign-kNN distance metric measures manifold proximity
   independently of the critic (benign baseline 0.138; clamp 0.91/0.87; trained
   0.67/0.62).

8. **Domain constraints.** Packet-size mean channels are clamped to the MTU
   (≤1500 bytes) and jitter channels to IAT ≥ 0. TTL is documented as not
   representable in AfterImage.

9. **Artifacts and documentation.** All reports regenerated; PGD manifold ER
   corrected from the stale 12.0% to 100.0%. README added. The Cython extractor
   is now a documented opt-in (`KITSUNE_USE_CYTHON=1`). DeepFool is benchmarked.
   Broken documentation links repaired.

---

## 3. Post-fix benchmark (seeded, disjoint holdout)

Latencies are single-run measurements and vary by roughly ±20%; relative order
is stable. kNN is mean normalised L2 to the nearest benign validation packet
(benign-to-benign baseline 0.1383).

### Phase A — Early scanning (1,000 detected)

| Attack | Draws/pkt | Mean RMSE | ASR | ER | Mean L∞ | kNN→benign |
|---|---:|---:|---:|---:|---:|
| Envelope clamp only | 1.00 | 0.1136 | 100.00% | 100.00% | 12.97 | 0.9096 |
| PGD manifold | 1.00 | 0.0909 | 100.00% | 100.00% | 13.03 | 1.2577 |
| WGAN manifold, trained | 1.00 | 0.0647 | 100.00% | 100.00% | 13.81 | 0.6739 |
| WGAN manifold, untrained | 1.00 | 5.6284 | 0.00% | 11.85% | 6.11 | 32.2566 |
| PGD box, ε=2 | 1.00 | 4.1601 | 2.30% | 13.88% | 2.00 | 23.9633 |
| WGAN box, 1 draw | 1.00 | 4.1925 | 1.60% | 13.52% | 2.00 | 24.2740 |
| WGAN box, best-of-10 | 9.85 | 4.1883 | 1.60% | 13.52% | 2.00 | 24.2401 |
| WGAN box, 10-step refine | 9.68 | 1.5275 | 3.80% | 15.29% | 10.43 | 14.5611 |

### Phase B — Volumetric flood (1,000 detected)

| Attack | Draws/pkt | Mean RMSE | ASR | ER | Mean L∞ | kNN→benign |
|---|---:|---:|---:|---:|---:|
| Envelope clamp only | 1.00 | 0.1092 | 100.00% | 100.00% | 34.97 | 0.8739 |
| PGD manifold | 1.00 | 0.0896 | 100.00% | 100.00% | 34.98 | 1.2060 |
| WGAN manifold, trained | 1.00 | 0.0391 | 100.00% | 100.00% | 35.62 | 0.6161 |
| WGAN manifold, untrained | 1.00 | 15.0501 | 0.00% | 11.85% | 15.87 | 83.0818 |
| PGD box, ε=2 | 1.00 | 12.2761 | 0.00% | 11.85% | 2.00 | 67.3309 |
| WGAN box, 1 draw | 1.00 | 12.2900 | 0.00% | 11.85% | 1.99 | 67.5695 |
| WGAN box, best-of-10 | 10.00 | 12.2890 | 0.00% | 11.85% | 2.00 | 67.5325 |
| WGAN box, 10-step refine | 10.00 | 5.1945 | 0.00% | 11.85% | 19.81 | 35.2957 |

Clean detection: θ = 0.455986, ROC-AUC 0.9418, TPR 88.13%, out-of-sample FPR
0.0961%, precision 99.93%, F1 0.9366, benign kNN baseline 0.1383.

---

## 4. Conclusion

The honest thesis statement is now:

> The manifold WGAN can learn a 100% evasive target without a post-hoc clamp;
> it beats the naive clamp on reconstruction error and benign-neighbour
> distance, while an untrained generator scores 0%. Genuinely budget-bounded
> attacks at ε = 2.0 remain weak, and generator retries/refinement do not
> overcome the budget.

The budgeted WGAN improved from 0.00% (pre-fix checkpoint) to 1.60% on the
scanning holdout after fixing the epsilon and overfitting defects, but it still
does not evade flood traffic at ε = 2.0. Best-of-k does not change ASR; ten
feedback-refinement steps raise scanning to 3.80% and cut flood RMSE to 5.19,
but aggregate perturbation reaches 11.5–19.8 ε, so it is no longer an ε = 2
result. The reviewer's predicted >80% was not reached.

---

## 5. Residual limitations

- TTL cannot be enforced because AfterImage does not extract it.
- Mutations remain feature-level; no packet reconstruction or joint
  feature-consistency check exists.
- The manifold attack is unbudgeted by design; its L∞ (13–36 normalised units)
  is not comparable to ε = 2 rows without reading the L∞ column. It is a learned
  target, not a bounded perturbation.
- Feedback refinement accumulates per-step budgets (10 steps ≈ 20 ε aggregate).
- One detector and one capture (Mirai) are evaluated.
