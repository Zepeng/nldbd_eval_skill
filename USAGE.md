# How to use the NLDBD validation skill (first-time instruction)

This is the short, ordered instruction for a person who wants to benchmark a model, alone or with
a coding agent (Claude Code, Codex, or any LLM that can run shell commands). The full protocol is
`validation_skill.md`; this page only tells you what to do, in order. Benchmark version 1.1.

## 0. What you get at the end

One table: energy-matched AUC (AUC_w) per dataset (MJD, EXO-200, NEXT, SuperNEMO) and per label
regime (1 %, 10 %, 100 % of the training labels), each number produced on the frozen test split
with all integrity gates passed. Coarse-mode numbers (validation subset) are for your own model
selection and never appear in the result table as benchmark numbers.

## 1. Get a writable copy (2 minutes)

```bash
# on the lab workstation
cp -r /home/tidmad/foundation_model ~/my_benchmark && cd ~/my_benchmark
# on NERSC Perlmutter (the source is read-only for everyone)
cp -r /global/cfs/cdirs/gen0032/foundation_model $CFS/<your project>/my_benchmark && cd $CFS/<your project>/my_benchmark
```

Do not run inside the read-only copy: `runs/`, `results/` and `cache/` are written next to the code.

## 2. Environment and data (5 minutes)

You need Python >= 3.11 with `numpy` and `h5py`; `torch` only for your model.

```bash
bash validation/setup_env.sh --with-torch        # creates ./.venv (uv or venv); omit --with-torch if you have torch
export PY=.venv/bin/python
# or reuse an existing environment:
#   workstation: export PY=/home/tidmad/SIDERIUS-tutorial/.venv/bin/python
#   NERSC:       export PY=/global/cfs/cdirs/gen0032/envs/fm/bin/python   (no torch)

# workstation
export NLDBD_DATA_ROOT=/home/klz/Data/zeronu_benchmark   # folder that contains MJD/ EXO-200/ NEXT/ SuperNEMO/
export NLDBD_NEXT_COMBINED=/home/klz/Data/NEXT_combined/next_combined.h5   # optional, makes NEXT fast
# NERSC (same layout, read-only)
export NLDBD_DATA_ROOT=/global/cfs/cdirs/gen0032/data/zeronu_benchmark
export NLDBD_NEXT_COMBINED=/global/cfs/cdirs/gen0032/data/NEXT_combined/next_combined.h5
export NLDBD_CACHE_DIR=/some/disk/with/10GB/free/cache   # optional; default ./cache (use $PSCRATCH on NERSC)
```

## 3. Self-check (1 minute; mandatory)

```bash
bash validation/selfcheck.sh $PY mjd exo200 supernemo next      # with data
bash validation/selfcheck.sh $PY                                # without data (steps 1 to 3 only)
```

It must end with `SELFCHECK PASS`. If it does not, stop: the copy or the data are not the frozen
ones, and any number you would produce is not comparable.

## 4. Write your model to the two-function contract (your work)

Create `my_model.py`:

```python
import numpy as np, torch

def load_model(dataset: str):
    """Build the model for this dataset and load your trained weights."""
    ...

def score(model, inputs) -> np.ndarray:
    """One float per event, higher = signal-like (raw logits are fine)."""
    ...
```

`inputs` is exactly what the reader yields (see `validation_skill.md` section 6, "Model input"):
MJD float32 `[B, 1, 3800]`, EXO-200 float32 `[B, 226, 300]`, NEXT and SuperNEMO a list of float32
`(n_hits, 4)` arrays. `examples/tiny_models.py` is a complete example for all four.

## 5. Train (your work, with three rules)

Read inputs only through the reader; take labels only from the frozen regime subset; never touch
the test split. Template in `examples/train_tiny.py`; the essentials:

```python
import sys; sys.path.insert(0, ".")
from validation.readers import get_reader
from validation.freeze_splits import labeled_subset_positions
import numpy as np

r = get_reader("mjd")
train, val = r.table("train"), r.table("validation")
pos = labeled_subset_positions(len(train["event_id"]), fraction=0.1, seed=42)   # the 10 % regime
# inputs: r.iter_inputs("train", batch_size=256, indices=pos, table=train)  -> (positions, inputs)
# labels: train["label"][pos]      validation for early stopping: val["label"], r.iter_inputs("validation", ...)
np.savez_compressed("runs/mymodel/mjd/L0.1/s42/training_ids.npz",
                    event_id=np.concatenate([train["event_id"][pos], val["event_id"]]),   # every event whose inputs you used
                    labeled_event_id=train["event_id"][pos])                              # every event whose label you used
```

Pretraining on unlabeled inputs may use the whole train split (and validation inputs) in any
regime; list those events in `event_id` too.

## 6. Score and evaluate

Coarse mode while developing (seconds per dataset after the first run):

```bash
R=runs/mymodel/mjd/L0.1/s42
$PY -B -m validation.make_predictions --dataset mjd --model-module my_model.py --subset coarse --output $R/coarse/predictions.npz
$PY -B -m validation.evaluate_predictions --dataset mjd --mode coarse --predictions $R/coarse/predictions.npz \
    --training-ids $R/training_ids.npz --label-fraction 0.1 --seed 42 --model mymodel --output $R/coarse
```

Full mode once per final model (this is the benchmark number):

```bash
$PY -B -m validation.make_predictions --dataset mjd --model-module my_model.py --output $R/predictions.npz
$PY -B -m validation.evaluate_predictions --dataset mjd --mode full --predictions $R/predictions.npz \
    --training-ids $R/training_ids.npz --label-fraction 0.1 --seed 42 --model mymodel --output $R
```

Each evaluation prints one line, for example
`{"dataset": "mjd", "mode": "full", "matched_auc": 0.8848, "benchmark_valid": true, "failed_gates": [], ...}`
and exits with 0 only when every gate passed. If a gate fails, `metrics.json` in the output folder
says which one and why (section 4.5 of the skill lists the usual causes).

Repeat for the four datasets, the three regimes (`L0.01`, `L0.1`, `L1.0`) and your seeds.

## 7. Report

```bash
$PY -B -m validation.report --runs runs --output results
cat results/results.md
```

Hand over `results/results.md`, the list of non-valid runs it prints, the benchmark version, your
model description and parameter count, the training recipe, which data each stage used, and the
`SELFCHECK PASS` output. Do not present inclusive AUC or accuracy as results.

## 7b. If what you have is a foundation model (pretrained encoder)

Skip steps 4 to 6. Write the three-function module described in `validation/fm_contract.md`
(`EMBED_DIM`, `to_tensors`, `load_encoder(dataset, pretrained)`), save the ids of every event your
pretraining used as `pretraining_ids.npz` (array `event_id`), and let the skill do the rest:

```bash
for d in mjd exo200 next supernemo; do for f in 0.01 0.1 1.0; do for m in probe full scratch; do
  $PY -B -m validation.finetune --fm-module my_fm.py --name myfm --dataset $d --label-fraction $f --mode $m \
      --pretraining-ids pretraining_ids.npz
done; done; done
$PY -B -m validation.report --runs runs --output results
```

Rows `myfm__full`, `myfm__probe` and `myfm__scratch` appear in `results/results.md`; `full` minus
`scratch` is what pretraining bought you. `examples/tiny_fm.py` is a working example of the module;
`validation_skill.md` section 5b states the frozen recipe.

## 8. Doing it with an agent

Give the agent the folder and one instruction, for example:

> Read validation_skill.md and USAGE.md. Run the self-check first and stop if it fails. Then
> implement my model from `model_notes.md` to the two-function contract, train it on all four
> datasets at the 1 %, 10 % and 100 % label regimes with seed 42 using coarse mode for model
> selection, run the full evaluation once per final model, and give me results/results.md with the
> reporting checklist.

or, for a foundation model:

> Read validation_skill.md and USAGE.md. Wrap the encoder in `my_fm.py` to the foundation-model
> contract, then run the fine-tuning protocol (probe, full, scratch) on all four datasets and all
> three label regimes with seed 42, and give me results/results.md with the reporting checklist.

The agent may choose the architecture, the training recipe and the number of seeds. It may not
change anything under `validation/energybench/`, `validation/manifests/` or
`validation/benchmark.json`, use test events anywhere in training, or pick labels outside the
frozen regime; the gates and the self-check will catch it if it does. Installing
`validation_skill.md` as the agent's skill (Claude Code: `~/.claude/skills/nldbd-validation/SKILL.md`)
makes it apply the protocol without being asked.

## 9. Adding a dataset or a metric

Follow section 8 of `validation_skill.md`: a dataset is a reader module plus an entry in
`validation/benchmark.json` plus one frozen manifest; a metric is one registered function plus an
entry in the same file. Both require a version bump, so results of different versions are never
mixed.

## 10. Common problems

| Symptom | Cause and fix |
|---|---|
| `SELFCHECK` fails at step 1 | a file under `validation/energybench/`, `manifests/` or `benchmark.json` was changed, or the manifests are from another version: restore the copy |
| `SELFCHECK` fails at step 4 | the data folder differs from the frozen one (wrong `NLDBD_DATA_ROOT`, missing or altered files) |
| `G2_test_identity` fails | you scored a subset or the wrong split/mode; use `make_predictions` without `--limit`, with `--subset coarse` only for `--mode coarse` |
| `G4a` fails | some training id is a test event, or an id is unknown: check how `training_ids.npz` was built |
| `G4b` fails | the labeled set is not the frozen regime subset: use `labeled_subset_positions` with seed 42 and the exact fraction |
| `manifest is benchmark version X, code declares Y` | mixed copies; take the whole folder from one source |
| NEXT reads take minutes | set `NLDBD_NEXT_COMBINED` to the combined file (or build it with `validation/build_next_combined.py`) |
| a process dies silently while reading NEXT or SuperNEMO | memory: keep per-event inputs as copies, not views into whole-file arrays (the shipped readers already do) |
