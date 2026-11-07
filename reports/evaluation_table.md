# Final Comparative Benchmark Summary

| Paradigm / Attack | Perturbation Constraint | Mean Attack RMSE | Conditional ASR (%) | Overall Evasion Rate (%) | Mean Linf |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Clean Baseline | None (Unperturbed) | 13.3351 | 0.00% | 11.85% | 0.0000 |
| Single-Step FGSM | eps = 0.05 * s | 13.3047 | 0.00% | 11.85% | 0.0500 |
| Single-Step FGSM | eps = 0.25 * s | 13.1846 | 0.00% | 11.85% | 0.2500 |
| Multi-Step PGD | 50 steps, eps = 2.0 * s | 12.1354 | 0.00% | 11.85% | 2.0000 |
| Manifold Traffic-Shaping | 50 steps, [min, max] | 0.0903 | 100.00% | 12.00% | 34.6495 |
| Polymorphic WGAN-GP | eps = 2.0 * s, closed loop | 12.1385 | 0.00% | 11.85% | 2.0000 |
