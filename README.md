# NLDBD foundation-model project: validation harness

Start with **[USAGE.md](USAGE.md)** (step-by-step instruction) and **[validation_skill.md](validation_skill.md)** (the protocol, version 1.2). It is the complete, reusable evaluation
protocol (energy-matched AUC, strict600 profile, frozen splits, coarse and full modes, integrity
gates) and tells any LLM or person how to score a model and report it.

* `validation/` — the harness (frozen paper evaluator, dataset readers, frozen manifests, prediction
  runner, evaluator with gates, report generator, self-check).
* `examples/` — tiny reference models that exercise the whole pipeline end to end.
* `runs/`, `results/` — outputs of the worked example.
* `nersc_upload.sh` — mirrors this folder to `/global/cfs/cdirs/gen0032/foundation_model`.

Quick check of a copy: `bash validation/selfcheck.sh <python-with-numpy-and-h5py>` (add dataset
names to also verify the readers against the data).
