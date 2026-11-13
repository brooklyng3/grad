# Final Comparative Benchmark Summary

## Phase A: Reconnaissance / Scanning Traffic (Early Attack)

| Paradigm / Attack | Perturbation Constraint | Draws/pkt | Mean Attack RMSE | Conditional ASR (%) | ER (this regime, extrap.) (%) | Mean Linf | kNN->benign (norm. L2) | Latency (ms / 1k pkts) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Clean Baseline | None (Unperturbed) | 0.00 | 5.3381 | 0.00% | 11.85% | 0.0000 | 30.3094 | 0.0 ms |
| Envelope Clamp Only | Unbudgeted [min, max] | 1.00 | 0.1136 | 100.00% | 100.00% | 12.9735 | 0.9096 | 1.4 ms |
| Single-Step FGSM | eps = 0.05 * s | 1.00 | 5.3076 | 0.00% | 11.85% | 0.0500 | 30.1469 | 37.9 ms |
| Single-Step FGSM | eps = 0.25 * s | 1.00 | 5.1910 | 0.00% | 11.85% | 0.2500 | 29.5408 | 41.2 ms |
| Multi-Step PGD (Box) | 50 steps, eps = 2 * s | 1.00 | 4.1601 | 2.30% | 13.88% | 1.9991 | 23.9633 | 1479.7 ms |
| DeepFool (bounded) | 50 iters, eps <= 2 * s | 1.00 | 4.5108 | 0.00% | 11.85% | 1.9683 | 25.8554 | 1582.0 ms |
| Multi-Step PGD (Manifold) | Unbudgeted [min, max], 50 steps | 1.00 | 0.0909 | 100.00% | 100.00% | 13.0325 | 1.2577 | 1445.1 ms |
| Polymorphic WGAN-GP (Box, 1 draw) | eps = 2 * s | 1.00 | 4.1925 | 1.60% | 13.26% | 1.9981 | 24.2740 | 13.7 ms |
| Polymorphic WGAN-GP (Box, best-of-k) | eps = 2 * s, retry until theta | 9.85 | 4.1883 | 1.80% | 13.43% | 1.9983 | 24.2401 | 147.5 ms |
| Polymorphic WGAN-GP (Box, feedback refine) | eps = 2 * s x 10 steps (aggregate) | 9.68 | 1.5275 | 3.80% | 15.20% | 10.4344 | 14.5611 | 166.7 ms |
| WGAN-GP (Manifold, trained) | Unbudgeted learned target, 1 draw | 1.00 | 0.0779 | 99.90% | 99.91% | 13.8967 | 0.7661 | 4.5 ms |
| WGAN-GP (Manifold, untrained) | Same forward, random weights | 1.00 | 5.6284 | 0.00% | 11.85% | 6.1065 | 32.2566 | 2.6 ms |

## Phase B: Volumetric Flooding Traffic (Heavy Attack)

| Paradigm / Attack | Perturbation Constraint | Draws/pkt | Mean Attack RMSE | Conditional ASR (%) | ER (this regime, extrap.) (%) | Mean Linf | kNN->benign (norm. L2) | Latency (ms / 1k pkts) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Clean Baseline | None (Unperturbed) | 0.00 | 13.4764 | 0.00% | 11.85% | 0.0000 | 73.7491 | 0.0 ms |
| Envelope Clamp Only | Unbudgeted [min, max] | 1.00 | 0.1092 | 100.00% | 100.00% | 34.9663 | 0.8739 | 1.0 ms |
| Single-Step FGSM | eps = 0.05 * s | 1.00 | 13.4459 | 0.00% | 11.85% | 0.0500 | 73.5862 | 41.3 ms |
| Single-Step FGSM | eps = 0.25 * s | 1.00 | 13.3258 | 0.00% | 11.85% | 0.2500 | 72.9632 | 42.7 ms |
| Multi-Step PGD (Box) | 50 steps, eps = 2 * s | 1.00 | 12.2761 | 0.00% | 11.85% | 2.0000 | 67.3309 | 1511.8 ms |
| DeepFool (bounded) | 50 iters, eps <= 2 * s | 1.00 | 12.9098 | 0.00% | 11.85% | 2.0000 | 70.6259 | 1480.9 ms |
| Multi-Step PGD (Manifold) | Unbudgeted [min, max], 50 steps | 1.00 | 0.0896 | 100.00% | 100.00% | 34.9754 | 1.2060 | 1490.8 ms |
| Polymorphic WGAN-GP (Box, 1 draw) | eps = 2 * s | 1.00 | 12.2900 | 0.00% | 11.85% | 1.9998 | 67.5695 | 1.6 ms |
| Polymorphic WGAN-GP (Box, best-of-k) | eps = 2 * s, retry until theta | 10.00 | 12.2893 | 0.00% | 11.85% | 1.9998 | 67.5598 | 116.9 ms |
| Polymorphic WGAN-GP (Box, feedback refine) | eps = 2 * s x 10 steps (aggregate) | 10.00 | 5.1945 | 0.00% | 11.85% | 19.7303 | 35.2957 | 128.1 ms |
| WGAN-GP (Manifold, trained) | Unbudgeted learned target, 1 draw | 1.00 | 0.0475 | 100.00% | 100.00% | 35.3521 | 0.6724 | 1.7 ms |
| WGAN-GP (Manifold, untrained) | Same forward, random weights | 1.00 | 15.0501 | 0.00% | 11.85% | 15.8726 | 83.0818 | 1.9 ms |
