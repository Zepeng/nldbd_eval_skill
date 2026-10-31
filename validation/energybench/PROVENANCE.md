# Frozen EnergyBench evaluator (do not edit)

`metrics.py`, `evaluate.py`, `configs/*.json` and `tests/test_metrics.py` are byte-for-byte
copies of `reproducibility/benchmark/` in the 0vBB-Benchmark release
(`0vBB-Benchmark_github_publish_20260923`, MIT License, Copyright (c) 2026 0vBB-Benchmark
contributors). That code produced Table 1 of "Learning Physics, Not Energy: A Multimodal
Benchmark for Neutrino Detectors". `SHA256SUMS` records the hashes; `tests/test_frozen.py`
in the parent folder fails if any file changes.

This project uses the `strict600` profile for every dataset: energies in keV, 600 fixed 5 keV
bins on 0-3000 keV, energies outside the range excluded (never clipped), energy matching with
>= 20 events per class per bin, >= 50 % coverage per class and >= 2 valid bins for a reportable
matched AUC. Only inclusive AUC and matched AUC (AUC_w) are reported in this project.
