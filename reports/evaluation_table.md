# Final Comparative Benchmark Summary

## Phase A: Reconnaissance / Scanning Traffic (Early Attack)

| Paradigm / Attack | Perturbation Constraint | Draws/pkt | Mean Attack RMSE | Conditional ASR (%) | Overall Evasion Rate (%) | Mean Linf | kNN->benign (norm. L2) | Latency (ms / 1k pkts) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Clean Baseline | None (Unperturbed) | 0.00 | 5.3381 | 0.00% | 11.85% | 0.0000 | 30.3094 | 0.0 ms |
| Envelope Clamp Only | Unbudgeted [min, max] | 1.00 | 0.1136 | 100.00% | 100.00% | 12.9735 | 0.9096 | 5.1 ms |
| Single-Step FGSM | eps = 0.05 * s | 1.00 | 5.3076 | 0.00% | 11.85% | 0.0500 | 30.1469 | 59.7 ms |
| Single-Step FGSM | eps = 0.25 * s | 1.00 | 5.1910 | 0.00% | 11.85% | 0.2500 | 29.5408 | 56.9 ms |
| Multi-Step PGD (Box) | 50 steps, eps = 2.0 * s | 1.00 | 4.1601 | 2.30% | 13.88% | 1.9991 | 23.9633 | 1844.8 ms |
| DeepFool (bounded) | 50 iters, eps <= 2.0 * s | 1.00 | 4.5108 | 0.00% | 11.85% | 1.9683 | 25.8554 | 2016.6 ms |
| Multi-Step PGD (Manifold) | Unbudgeted [min, max], 50 steps | 1.00 | 0.0909 | 100.00% | 100.00% | 13.0325 | 1.2577 | 1941.2 ms |
| Polymorphic WGAN-GP (Box, 1 draw) | eps = 2 * s | 1.00 | 4.1836 | 1.90% | 13.52% | 1.9990 | 24.1680 | 13.8 ms |
| Polymorphic WGAN-GP (Box, best-of-k) | eps = 2 * s, retry until theta | 9.83 | 4.1807 | 1.90% | 13.52% | 1.9991 | 24.1361 | 140.0 ms |
| Polymorphic WGAN-GP (Box, feedback refine) | eps = 2 * s x 10 steps (aggregate) | 9.67 | 1.1778 | 3.90% | 15.29% | 11.4744 | 12.1770 | 161.8 ms |
| WGAN-GP (Manifold, trained) | Unbudgeted learned target, 1 draw | 1.00 | 0.0782 | 99.90% | 99.91% | 13.8111 | 0.8572 | 8.6 ms |
| WGAN-GP (Manifold, untrained) | Same forward, random weights | 1.00 | 5.6284 | 0.00% | 11.85% | 6.1065 | 32.2566 | 2.7 ms |

## Phase B: Volumetric Flooding Traffic (Heavy Attack)

| Paradigm / Attack | Perturbation Constraint | Draws/pkt | Mean Attack RMSE | Conditional ASR (%) | Overall Evasion Rate (%) | Mean Linf | kNN->benign (norm. L2) | Latency (ms / 1k pkts) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Clean Baseline | None (Unperturbed) | 0.00 | 13.4764 | 0.00% | 11.85% | 0.0000 | 73.7491 | 0.0 ms |
| Envelope Clamp Only | Unbudgeted [min, max] | 1.00 | 0.1092 | 100.00% | 100.00% | 34.9663 | 0.8739 | 1.0 ms |
| Single-Step FGSM | eps = 0.05 * s | 1.00 | 13.4459 | 0.00% | 11.85% | 0.0500 | 73.5862 | 55.7 ms |
| Single-Step FGSM | eps = 0.25 * s | 1.00 | 13.3258 | 0.00% | 11.85% | 0.2500 | 72.9632 | 68.4 ms |
| Multi-Step PGD (Box) | 50 steps, eps = 2.0 * s | 1.00 | 12.2761 | 0.00% | 11.85% | 2.0000 | 67.3309 | 1963.7 ms |
| DeepFool (bounded) | 50 iters, eps <= 2.0 * s | 1.00 | 12.9098 | 0.00% | 11.85% | 2.0000 | 70.6259 | 1914.4 ms |
| Multi-Step PGD (Manifold) | Unbudgeted [min, max], 50 steps | 1.00 | 0.0896 | 100.00% | 100.00% | 34.9754 | 1.2060 | 1990.8 ms |
| Polymorphic WGAN-GP (Box, 1 draw) | eps = 2 * s | 1.00 | 12.2905 | 0.00% | 11.85% | 1.9999 | 67.5523 | 1.7 ms |
| Polymorphic WGAN-GP (Box, best-of-k) | eps = 2 * s, retry until theta | 10.00 | 12.2890 | 0.00% | 11.85% | 1.9999 | 67.5325 | 134.4 ms |
| Polymorphic WGAN-GP (Box, feedback refine) | eps = 2 * s x 10 steps (aggregate) | 10.00 | 5.3781 | 0.00% | 11.85% | 19.8139 | 36.4872 | 148.0 ms |
| WGAN-GP (Manifold, trained) | Unbudgeted learned target, 1 draw | 1.00 | 0.0367 | 100.00% | 100.00% | 35.6243 | 0.6228 | 1.8 ms |
| WGAN-GP (Manifold, untrained) | Same forward, random weights | 1.00 | 15.0501 | 0.00% | 11.85% | 15.8726 | 83.0818 | 5.1 ms |
