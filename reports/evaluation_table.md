# Final Comparative Benchmark Summary

## Phase A: Reconnaissance / Scanning Traffic (Early Attack)

| Paradigm / Attack | Perturbation Constraint | Mean Attack RMSE | Conditional ASR (%) | Overall Evasion Rate (%) | Mean Linf | Latency (ms / 1k pkts) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Clean Baseline | None (Unperturbed) | 3.7961 | 0.00% | 11.85% | 0.0000 | 0.0 ms |
| Single-Step FGSM | eps = 0.05 * s | 3.7664 | 0.00% | 11.85% | 0.0500 | 284.4 ms |
| Single-Step FGSM | eps = 0.25 * s | 3.6568 | 0.10% | 11.94% | 0.2500 | 35.2 ms |
| Multi-Step PGD (Box) | 50 steps, eps = 2.0 * s | 2.6743 | 7.20% | 18.19% | 1.9968 | 1674.6 ms |
| Multi-Step PGD (Manifold) | 50 steps, [min, max] | 0.0845 | 100.00% | 100.00% | 8.8060 | 1513.7 ms |
| Polymorphic WGAN-GP (Box) | eps = 2.0 * s, closed loop | 2.9508 | 0.00% | 11.85% | 1.9999 | 19.4 ms |
| Polymorphic WGAN-GP (Manifold) | Envelope Shaping, single pass | 0.2275 | 100.00% | 100.00% | 9.5688 | 1.5 ms |

## Phase B: Volumetric Flooding Traffic (Heavy Attack)

| Paradigm / Attack | Perturbation Constraint | Mean Attack RMSE | Conditional ASR (%) | Overall Evasion Rate (%) | Mean Linf | Latency (ms / 1k pkts) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Clean Baseline | None (Unperturbed) | 13.5061 | 0.00% | 11.85% | 0.0000 | 0.0 ms |
| Single-Step FGSM | eps = 0.05 * s | 13.4756 | 0.00% | 11.85% | 0.0500 | 36.8 ms |
| Single-Step FGSM | eps = 0.25 * s | 13.3556 | 0.00% | 11.85% | 0.2500 | 36.7 ms |
| Multi-Step PGD (Box) | 50 steps, eps = 2.0 * s | 12.3060 | 0.00% | 11.85% | 2.0000 | 1526.1 ms |
| Multi-Step PGD (Manifold) | 50 steps, [min, max] | 0.0899 | 100.00% | 100.00% | 35.0449 | 1733.1 ms |
| Polymorphic WGAN-GP (Box) | eps = 2.0 * s, closed loop | 12.4102 | 0.00% | 11.85% | 1.9998 | 1.7 ms |
| Polymorphic WGAN-GP (Manifold) | Envelope Shaping, single pass | 0.2283 | 100.00% | 100.00% | 35.0845 | 1.4 ms |
