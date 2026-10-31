---
name: nldbd-energy-matched-auc-validation
description: Reproducible evaluation of any neutrinoless-double-beta-decay (NLDBD) event classifier or foundation model on the benchmark datasets MJD, EXO-200, NEXT and SuperNEMO with the paper's frozen energy-matched AUC (EnergyBench strict600), frozen official splits, three label regimes (1 %, 10 %, 100 %), a coarse mode (frozen validation subset, for development) and a full mode (complete test split, the benchmark), a frozen fine-tuning protocol (linear probe, full fine-tune, scratch control) that turns any pretrained encoder into benchmark rows, integrity gates that make a result benchmark-valid, and a registry for adding datasets and metrics. Use it whenever a model trained on these datasets has to be scored, fine-tuned, compared with the paper "Learning Physics, Not Energy", reported, or when the benchmark has to be extended.
---

# NLDBD validation skill, protocol 1.2

This file is the complete protocol. An LLM or a person following it from top to bottom, with the
harness folder beside it, reproduces the evaluation exactly. "Harness" means the `validation/`
folder next to this file. The benchmark is declared in `validation/benchmark.json` (name, version,
profile, primary and secondary metrics, label regimes, coarse rule, registered datasets); the code
reads that declaration, so adding a dataset or a metric is a registration (section 8), not a code
change in the protocol.

Part A (sections 1 to 5) is the **protocol**: it does not change within a benchmark version.
Section 5b is the **fine-tuning protocol** for foundation models: the user supplies only a
pretrained encoder; the skill owns the head, the recipe, the regimes, the controls and the bookkeeping.
Part B (sections 6 to 7) holds the **registrations**: one card per dataset and per metric.
Part C (sections 8 to 11) covers extending the benchmark, costs, using it with an agent, files.

---

# Part A: the protocol (fixed within version 1.2; frozen artefacts at version 1.1)

## 1. The metrics per dataset

Each dataset has its own primary and secondary metrics. Where a dataset has two primary metrics,
they are listed in order of precedence:

| Dataset | Primary | Secondary | Metric cards |
|---|---|---|---|
| EXO-200 | 1. energy RMSE over the full energy range; 2. sigma/E at 2615 keV, which decides when the RMSEs differ by less than 2 keV | single-site against multi-site (SS/MS) energy-matched AUC | 7.4, 7.1 |
| SuperNEMO | energy-matched AUC | opening-angle RMSE against the true escape angle `phiS` | 7.1, 7.5 |
| NEXT | energy-matched AUC (the only metric) | none | 7.1 |
| MJD | energy-matched AUC | inclusive AUC | 7.1, 7.2 |

The **energy-matched AUC**, written **AUC_w**, is computed by the frozen EnergyBench evaluator
of the paper *Learning Physics, Not Energy: A Multimodal Benchmark for Neutrino Detectors*
(ICLR 2027 submission) under its **strict600** profile:

* events are binned in 600 fixed 5 keV energy bins on 0 to 3000 keV (energies outside are
  excluded, never clipped);
* the common energy support of the two classes is the overlap of their central 99 % ranges
  (0.5 % trimmed on each side); only bins with at least 20 events of each class inside that
  support are kept;
* in every kept bin the signal-like and background-like classes are reweighted to the same
  target mass (the minimum of the two class-normalised bin masses), so the two classes have
  identical energy spectra after weighting;
* AUC_w is the pooled weighted pairwise AUC over all retained events (ties count one half);
* AUC_w is reportable only if each class keeps at least 50 % of its original finite-energy mass
  and at least two bins are valid. Otherwise the value is `null` and must be reported as
  "not estimable", never replaced by a diagnostic value.

A classifier that uses only energy obtains AUC_w = 0.5 by construction (section 7 lists the
floors). Secondary metrics can be tabulated, but they are not the benchmark result and must be
labelled as secondary wherever shown.

## 2. Two evaluation modes

| Mode | Population | Purpose | Counts as benchmark |
|---|---|---|---|
| **coarse** | a frozen, seeded subset of the **validation** split (`manifests/<ds>.json["coarse"]`): the smallest of 0.5/1/2/5/10/25/50/100 % that is at least the dataset's declared floor, on which AUC_w is estimable and each class keeps >= 50 % of the energy coverage the full validation split keeps | quick comparison of candidates during development, model selection, sanity checks; seconds per dataset once cached | no |
| **full** | the complete frozen **test** split | the reported number | yes |

Both modes use the same evaluator, profile, gates and output format; only the population differs.
Coarse numbers rank candidates; only full-mode numbers are reported. Current coarse populations are
in the dataset cards (section 6) and the measured coarse-versus-full agreement in section 9.

## 3. Hard rules

1. **Never modify** `validation/energybench/` (evaluator, profiles, tests), `validation/manifests/`
   or `validation/benchmark.json` within a version. `validation/tests/test_frozen.py` fails if the
   evaluator changes or if a manifest does not carry the declared version. A change is a new
   benchmark version (section 8.3).
2. **The test split is touched once** per (model, dataset, label regime, seed): to produce the
   full-mode `predictions.npz`. No model selection, early stopping, hyper-parameter choice,
   threshold or normalisation statistic may use test events. Model selection uses the validation
   split (coarse mode exists for exactly that).
3. **No test event in any training step**, including self-supervised pretraining, contrastive
   pretraining, statistics for normalisation, tokenizer fitting or data-driven augmentation.
   Gate G4a checks this from the `training_ids.npz` you must write.
4. **Labels come only from the frozen regime subset.** The 1 %, 10 % and 100 % regimes are
   nested, seeded subsets of the official train split fixed in the manifests (section 4). Gate
   G4b checks that the labeled set equals the frozen subset exactly. Unlabeled inputs from the
   whole train split (and validation inputs) may be used for pretraining in any regime.
5. **Energy, labels, categories and their proxies never enter the model.** The readers only hand
   the model the paper's inputs; each dataset card lists the forbidden fields.
6. **MJD_NPML shards are excluded** from this benchmark entirely (not even for pretraining).
7. **Every reported number is produced by** `validation/evaluate_predictions.py` **with all gates
   passing** (`"benchmark_valid": true`). A run with a failed or skipped gate may be discussed but
   is not a benchmark result; the report lists it separately with its failed gates.
8. **Report all three label regimes** (1 %, 10 %, 100 %) for every model and dataset, the seed or
   seeds used, and the benchmark version. There is no pass/fail threshold: the table is the result.

## 4. Frozen contracts

### 4.1 Reader contract (every dataset)

```python
from validation.readers import get_reader
r = get_reader("mjd")                 # any name registered in benchmark.json
t = r.table("test")                   # also "train", "validation"
# t["event_id"] str, t["label"] int8 (1 = signal-like = evaluation-positive), t["energy_keV"] float64,
# t["group"] str, plus locator fields. Canonical order = the paper's order.
for positions, inputs in r.iter_inputs("test", batch_size=256, indices=None, table=t):
    ...                               # positions index t; inputs are the paper's preprocessed inputs
```

### 4.2 Frozen manifests and label regimes

`validation/manifests/<ds>.json` (generated once by `freeze_splits.py`, never edited) records the
benchmark version, and for each split the event count, label counts and SHA-256 hashes of the event
ids (sorted and in canonical order), labels and energies; the nested label regimes; the coarse
subset with every candidate fraction it considered; the energy-only floors.
`validation/manifests/<ds>_ids.npz` holds the event ids of all three splits and, for the test
split and the coarse subset, labels, energies and groups, so predictions can be evaluated in both
modes on a machine without raw data.

Label regimes (seed 42): `positions = np.random.default_rng(42).permutation(n_train)[:round(f * n_train)]`
into the canonical train order, `f` in {0.01, 0.1, 1.0}; subsets are nested.

```python
from validation.freeze_splits import labeled_subset_positions
train = get_reader(ds).table("train")
pos = labeled_subset_positions(len(train["event_id"]), fraction=0.1, seed=42)
labeled_ids = train["event_id"][pos]          # labels: train["label"][pos]
```

### 4.3 File schemas

| File | Arrays | Written by |
|---|---|---|
| `predictions.npz` | `event_id` (str, one per event of the evaluation population), `score` (float, higher = signal-like; raw logits are fine, the metric is rank-based). Optional `label`, `energy_keV` are checked against the manifest, never used. | `validation/make_predictions.py` (or your own code) |
| `training_ids.npz` | `event_id`: every event whose inputs were used in ANY training step on this dataset (train and validation events are allowed); `labeled_event_id`: every train event whose label was used for a gradient step. | your training code (see `examples/train_tiny.py`) |
| `metrics.json` | benchmark name and version, mode, population, `primary_metric`, `primary_value`, `metrics{name: value,status,detail}`, gates, diagnostics, the full evaluator record | `validation/evaluate_predictions.py` |

### 4.4 Run layout

`runs/<model>/<dataset>/L<fraction>/s<seed>/` for full mode and
`runs/<model>/<dataset>/L<fraction>/s<seed>/coarse/` for coarse mode, `fraction` in `0.01`,
`0.1`, `1.0`. `validation/report.py` depends on this layout.

### 4.5 Gates (all must be PASS)

| Gate | Checks | Typical cause of failure |
|---|---|---|
| G1_schema | `event_id` and `score` present, aligned, ids unique, finite scores counted | wrong keys, NaN scores |
| G2_test_identity | prediction ids equal the frozen population (test split in full mode, coarse subset in coarse mode) as a set; order is recorded | scored a subset, wrong split or mode, stale data copy |
| G3_supplied_metadata_agrees | optional `label` / `energy_keV` in the file equal the frozen ones | label orientation flipped, float32 energies |
| G4a_no_test_contamination | no training id is a test id; all training ids belong to train or validation; in coarse mode additionally no coarse event had its label used | pretraining on all files, leaked test shard, validation labels in the loss |
| G4b_label_regime | labeled ids equal the frozen subset for `--label-fraction` and lie in the train split; without `--label-fraction` the gate is SKIPPED and the run is not benchmark-valid | different seed or sampling rule, validation labels used for gradients, regime not declared |
| G5_primary_metric_reportable | the primary metric's status is `ok` | too few events or disjoint support (only with tiny populations) |

`--no-training-ids` skips G4; the result is then explicitly not benchmark-valid.

Diagnostics in `metrics.json` (never benchmark numbers): `label_shuffle_null_primary_metric` (AUC_w
after shuffling labels, expected 0.5), `energy_only_floor` (score = ±energy, from the manifest,
expected 0.5), and the full evaluator record under `energybench` (common support, valid bins,
coverage per class, effective sample sizes).

## 5. Procedure for one model

1. **Set up and self-check** (section 10 has the paths):

   ```bash
   cd <folder containing this file>
   bash validation/setup_env.sh [--with-torch]      # optional: creates ./.venv (numpy, h5py)
   export PY=.venv/bin/python                       # or any python >= 3.11 with numpy + h5py
   export NLDBD_DATA_ROOT=/path/to/zeronu_benchmark # contains MJD/ EXO-200/ NEXT/ SuperNEMO/
   bash validation/selfcheck.sh $PY mjd exo200 supernemo next   # must end with SELFCHECK PASS
   ```

   The self-check proves (1) evaluator, manifests and declaration are the frozen ones and every
   declared metric is registered, (2) the evaluator passes the paper's own unit tests, (3) the gates
   behave on synthetic predictions, (4) the readers on this machine reproduce every frozen test
   split. Without data, run it with no dataset names (steps 1 to 3).
2. **Train** with your own code, reading inputs only through `get_reader(ds).iter_inputs(...)` on
   the train split (any events, unlabeled) and the labeled subset of 4.2 for supervised losses.
   Use the validation split for early stopping and model selection. Keep the seed.
3. **Write `training_ids.npz`** (schema 4.3) into the run folder.
4. **Score the evaluation population** through a module exposing `load_model(dataset)` and
   `score(model, inputs) -> np.ndarray` (one float per event, higher = signal-like):

   ```bash
   # coarse (validation subset), as often as you like while developing
   $PY -B -m validation.make_predictions --dataset mjd --model-module path/to/your_model.py \
       --subset coarse --output runs/<model>/mjd/L0.1/s42/coarse/predictions.npz
   # full (test split), once per final model
   $PY -B -m validation.make_predictions --dataset mjd --model-module path/to/your_model.py \
       --output runs/<model>/mjd/L0.1/s42/predictions.npz
   ```

   It writes `predictions.npz` and `predictions_provenance.json` (module hash, time, event count).
   `examples/tiny_models.py` shows the contract for all four input types. In coarse mode the inputs
   come from a one-time local cache (section 9); the first coarse run builds it.
5. **Evaluate** (same command for both modes):

   ```bash
   $PY -B -m validation.evaluate_predictions --dataset mjd --mode coarse \
       --predictions runs/<model>/mjd/L0.1/s42/coarse/predictions.npz \
       --training-ids runs/<model>/mjd/L0.1/s42/training_ids.npz \
       --label-fraction 0.1 --seed 42 --model <model> --output runs/<model>/mjd/L0.1/s42/coarse
   $PY -B -m validation.evaluate_predictions --dataset mjd --mode full \
       --predictions runs/<model>/mjd/L0.1/s42/predictions.npz \
       --training-ids runs/<model>/mjd/L0.1/s42/training_ids.npz \
       --label-fraction 0.1 --seed 42 --model <model> --output runs/<model>/mjd/L0.1/s42
   ```

   Prints one JSON line with the mode, the primary metric value and `benchmark_valid`; exit code 0
   only when all gates pass. In coarse mode `benchmark_valid` means "all gates passed", not
   "benchmark".
6. **Repeat** for every dataset, the three regimes and your seeds, then build the tables:

   ```bash
   $PY -B -m validation.report --runs runs --output results            # primary metric
   $PY -B -m validation.report --runs runs --output results --metric inclusive_auc   # a secondary one
   ```

`examples/train_tiny.py` runs steps 2 to 5 end to end (both modes) for a tiny model and is the
executable reference: `$PY -B examples/train_tiny.py --dataset exo200 --label-fraction 0.01 --epochs 8 --mode both`.

**Reporting checklist.** A benchmark report contains: the `results/results.md` table (AUC_w per
dataset and regime, mean ± std over seeds when more than one), the runs that were not
benchmark-valid with their failed gates, the benchmark version, the model description and
parameter count, the training recipe (optimiser, schedule, epochs, early-stopping rule, seeds),
what data each stage used (pretraining events, labeled events, validation usage), the harness
copy used and the output of `selfcheck.sh`. Secondary metrics, accuracy or the independence score
are never presented as results.

## 5b. Fine-tuning protocol for foundation models

Recipe 1.2.1 (2026-10-23): in `probe` mode the frozen encoder is evaluated in eval mode and every input is encoded exactly once; the head trains on cached pooled embeddings. Same arithmetic as re-encoding each pass (the encoder never changes), minutes become seconds. `full` and `scratch` are unchanged.

**Who does what.** The foundation-model author pretrains an encoder however they like (any
events of the train and validation splits, never test events, and they record the ids they used).
The skill turns that encoder into benchmark rows with one command per dataset, regime and mode.

**The contract** (`validation/fm_contract.md`): one Python module with `EMBED_DIM` (int, or one
per dataset), `to_tensors(dataset, inputs)` (the author's tokenizer or stem: reader inputs to
encoder tensors, never touching labels or energies) and `load_encoder(dataset, pretrained=True)`
returning a `torch.nn.Module` whose forward gives tokens `[B, N, D]` plus a boolean mask, or an
embedding `[B, D]`; `pretrained=False` must build the same architecture with random weights.
`examples/tiny_fm.py` is a complete example.

**The frozen recipe** (`validation/finetune_recipe.json`, stamped into every run):

| Item | Value |
|---|---|
| Modes, all run for every foundation model | `probe` (encoder frozen, head trains), `full` (everything trains), `scratch` (same architecture, random weights: the control) |
| Head | masked mean pooling, LayerNorm(D), Linear(D, 64), GELU, Linear(64, 1); fresh per run |
| Loss | binary cross-entropy with logits; no class weights, no augmentation |
| Optimiser | AdamW, weight decay 1e-4; lr 5e-4 for the head; 1e-4 for a pretrained encoder (`full`), 5e-4 for `scratch`; gradient clipping 1 |
| Schedule | 5 % warm-up, cosine to zero over the planned steps; batch 64; fp32 |
| Length | at most 50 epochs; early stopping (patience 5) is armed only after 3,000 optimizer steps, small label sets are cycled to get there |
| Model selection | inclusive AUC on the frozen coarse subset (its inputs come from the cache, so it costs seconds per epoch); the best epoch's weights score the coarse subset and then the test split once |
| Regimes and seeds | 1 %, 10 %, 100 %; seed 42; seeds 43 and 44 added at 10 % for the headline comparison |
| Reported quantity | the three rows per regime; the difference `full` minus `scratch` is the value of pretraining, `probe` is representation quality |

```bash
$PY -B -m validation.finetune --fm-module my_fm.py --name myfm --dataset mjd \
    --label-fraction 0.1 --mode full --seed 42 [--pretraining-ids pretraining_ids.npz] [--max-hours 2]
# repeat for --mode probe and --mode scratch, every dataset, every regime; then
$PY -B -m validation.report --runs runs --output results
```

Each run writes `runs/<name>__<mode>/<dataset>/L<fraction>/s<seed>/` with `finetune.json`
(recipe stamp, module hash, history, selected epoch, hours), `training_ids.npz` (labeled regime
ids, coarse ids used for selection, pretraining ids), coarse and full predictions and
`metrics.json` from the same gates as everything else. Labeled inputs that fit the memory budget
(`NLDBD_FINETUNE_RAM_GB`, default 8) are read once; larger regimes stream through the reader each
pass with buffer shuffling. Joint fine-tuning (one encoder, four heads) is not part of 1.2.

---

# Part B: registrations (version 1.1 artefacts)

## 6. Dataset cards

Every card follows the same template; the reader module's doc-string cites the paper code it
reproduces. Row counts come from the frozen manifests.

### 6.1 MJD (MAJORANA DEMONSTRATOR), `readers/mjd.py`

| | |
|---|---|
| Files | 16 `MJD_Train_*.hdf5`, 6 `MJD_Test_*.hdf5` (65,000 rows each). `MJD_NPML_*` excluded. |
| Signal-like (label 1) | all four PSD flags true ("clean"); background: any flag false |
| Energy | `energy_label` keV, float64 |
| Split | test = the 6 Test files (390,000); `default_rng(42).permutation` of the 1,040,000 Train rows (files sorted lexicographically): first 104,000 validation, rest train (936,000) |
| Regimes | 1 % = 9,360, 10 % = 93,600, 100 % = 936,000 labeled events |
| Coarse | 10 % of validation = 10,400 events (42 valid bins) |
| `event_id` | `MJD_Test_0.hdf5::row=0::id=2395098` |
| Model input | float32 `[B, 1, 3800]`: raw waveform minus mean of first 200 samples, divided by max abs |
| Forbidden as input | `energy_label`, PSD flags, `id`, `run_number`, `detector`, `tp0` |

### 6.2 EXO-200, `readers/exo200.py`

| | |
|---|---|
| Files | 29 `<run>-<part>.h5` shards; `Waveforms/<i>` is a (226, 300) int16 array |
| Signal-like (label 1) | `Charge_cluster_number == 1` (the stored exobench label has the opposite orientation; flipped) |
| Energy | `Rotated_energy` keV |
| Split | run-grouped, seed 42 (`manifests/external/exo200_v1_split.json`): train runs 8956 8957 8958 8968 8972 8999 9000 9001 9008 9009 (336,632), validation 8969 (37,987), test 8967 8970 8971 9010 (140,383) |
| Regimes | 1 % = 3,366, 10 % = 33,663, 100 % = 336,632 |
| Coarse | 100 % of validation = 37,987 events (202 valid bins; no smaller fraction keeps >= 20 events per class per bin) |
| `event_id` | `EXO200::8967-000.h5::4` |
| Model input | float32 `[B, 226, 300]`: per-channel baseline (first 200 samples) subtracted, event divided by its max abs; channel blocks U+ 0:38, V+ 38:76, U− 76:114, V− 114:152, APD+ 152:189, APD− 189:226 |
| Forbidden as input | `Rotated_energy`, `Total_*_Energy`, `Charge_cluster_number`, `Charge_Clusters_Pos` (its shape encodes the label), `event_number` |

### 6.3 NEXT, `readers/next.py`

| | |
|---|---|
| Files | `0nubb_part_*/*.h5`, `Bi_part_*/*.h5` (14,973 files, 1,165,489 events), table `MC/hits/table`; or the combined file (below) |
| Signal-like (label 1) | 0nubb; background: Bi214 |
| Energy | float64 sum of the event's hit energies (MeV) converted to keV by the evaluator's exact rule |
| Split | event-count 80/10/10, seed 42, per-class file shuffle, only boundary files sliced (`manifests/external/next_event_split.json`); train 932,391, validation 116,549, test 116,549 (64,879 / 51,670) |
| Regimes | 1 % = 9,324, 10 % = 93,239, 100 % = 932,391 |
| Coarse | 10 % of validation = 11,655 events (9 valid bins; declared floor 0.1: 0.5 % was estimable but noisy) |
| `event_id` | `NEXT::<relative path>::<source event id>` |
| Model input | list of float32 `(n_hits, 4)` = x, y, z [mm], hit energy [MeV]; divide energies by the event total and center coordinates inside the model |
| Forbidden as input | absolute event energy, category, file path, event id |
| Combined file | `validation/build_next_combined.py` writes `/home/klz/Data/NEXT_combined/next_combined.h5` (override `NLDBD_NEXT_COMBINED`): the same hits, ids, labels and energies stored contiguously in canonical split order (56.6 GB, uncompressed on purpose). The reader uses it when its inventory digest matches the manifest, else the files. Train split read: 20 s instead of ~7 min. |

### 6.4 SuperNEMO, `readers/supernemo.py`

| | |
|---|---|
| Files | `data_2nubb_merged.h5`, `data_Bi214_merged.h5` (one row per tracker hit); 0nubb and Tl208 files unused |
| Signal-like (label 1) | 2nubb (group `2nu`); background: Bi214 |
| Energy | `E1 + E2` keV (first row of the event) |
| Split | 4096-event blocks per source permuted with `SeedSequence([42, source position])`, 80/10/10 (`manifests/external/supernemo_split_manifest.json`); train 4,735,250, validation 593,044, test 589,824 (327,680 / 262,144) |
| Regimes | 1 % = 47,352, 10 % = 473,525, 100 % = 4,735,250 |
| Coarse | 5 % of validation = 29,652 events (270 valid bins) |
| `event_id` | `SuperNEMO::2nubb::12345` |
| Model input | list of float32 `(n_hits, 4)` = tX, tY, tZ [mm], tR [mm, may be NaN] |
| Forbidden as input | `E1`, `E2`, `dY dZ theta phiS phiR` (simulation truth), label, source name, `ev_no` |

## 7. Metric cards

### 7.1 `matched_auc` (primary for SuperNEMO, NEXT and MJD; secondary for EXO-200), `validation/metrics/__init__.py`

On EXO-200 it is the SS/MS energy-matched AUC: signal-like means single-site, `Charge_cluster_number == 1` (card 6.2).

Energy-matched AUC under strict600 as defined in section 1, computed by the frozen
`validation/energybench/metrics.py` (SHA-256 in `energybench/SHA256SUMS`). Status `ok` or a
`not_estimable_*` reason; detail: valid bins, common support, per-class coverage and effective
sample sizes. Paper values (Table 1, full labels, seed 42) and this harness's replay of them:

| Model | MJD | EXO-200 | NEXT | SuperNEMO |
|---|---|---|---|---|
| Region + MLP Transformer (paper's recommended baseline) | 0.973 | 0.970 | 0.988 | 0.591 |
| Best native model | BiGRU 0.975 | Static GINE 0.949 | Static GINE 0.995 | Static GINE 0.590 |
| Best Transformer | Region + Fourier 0.981 | Region + Fourier 0.972 | Region + MLP 0.988 | Region + RoPE 0.602 |

MJD and EXO-200 are strict600 values, reproduced to the last digit from the paper's saved
predictions (0.97345 and 0.96956 for Region + MLP). The NEXT and SuperNEMO paper rows used other
energy-binning profiles, so they are approximate references.

Energy-only floors on the frozen test splits (score = +energy):

| | MJD | EXO-200 | NEXT | SuperNEMO |
|---|---|---|---|---|
| AUC_w of energy alone | 0.500 | 0.500 | 0.506 | 0.500 |
| valid 5 keV bins / common support (keV) | 493 / 18–2614 | 426 / 538–2663 | 11 / 2435–2480 | 374 / 350–2218 |

Paper training recipe for the Region + MLP baseline (for a like-for-like comparison, not a rule):
backbone 64-d, 2 encoder layers, 4 heads, feed-forward 256, masked mean pooling; AdamW, lr 5e-4,
weight decay 1e-4, batch 64, cosine schedule, gradient clipping 1, early stopping patience 5 on
validation AUC, at most 50 epochs, seed 42.

### 7.2 `inclusive_auc` (secondary for MJD; diagnostic elsewhere)

Plain rank AUC on all finite scores, independent of energy. Shown only to quantify how much of a
model's separation is energy (compare with `matched_auc`); energy alone scores 0.41 (MJD), 0.28
(EXO-200), 0.71 (NEXT), 0.35 (SuperNEMO) on the test splits. Never reported as a result. On
EXO-200, SuperNEMO and NEXT it is a diagnostic only, not a secondary metric.

### 7.3 Official baseline: Region + MLP Transformer re-run under this protocol (2026-10-25)

**Every model is compared with this table.** Do not compare with the paper's numbers (other energy profiles,
full labels only) or with a physics reconstruction. The table shows the paper's Region + MLP Transformer, trained
from scratch by us on the frozen splits and label regimes of this protocol, scored on the test split by the frozen
evaluator. All 12 runs pass every gate (`benchmark_valid: true`). It is not a reproduction of the paper: what matters is that
the evaluation is self-consistent (the same rules as every foundation-model fine-tune).

Matched AUC (strict600), test split, seed 42:

| Labels | MAJORANA DEMONSTRATOR | EXO-200 | NEXT | SuperNEMO |
|---|---|---|---|---|
| 1 % | 0.869 | 0.776 | 0.913 | 0.573 |
| 10 % | 0.901 | 0.935 | 0.978 | 0.586 |
| 100 % | 0.959 | 0.973 | 0.989 | 0.590 |

**Model.** The paper's own code is imported read-only (`research/paper_baseline/region_mlp.py`). It uses the paper
runs' configurations (bundle `provenance/transformer_model_mapping.csv`, Region + MLP rows), and the parameter counts
equal the paper's:

| Detector | Code | Tokens + positional encoding | Parameters |
|---|---|---|---|
| MJD | `mjd_transformer.MJDTransformer` | raw_patches, patch 20 = 190 tokens of 3800 samples; coordinate_mlp | 112,257 |
| EXO-200 | `exo_transformer.EXOTransformerClassifier` | raw_patches, 6 sensor blocks x 7 channel x 12 time regions; coordinate_mlp | 120,769 |
| NEXT | `next_transformer.NEXTTransformerClassifier` | voxel 15 mm, <= 512 tokens; coordinate_mlp | 111,105 |
| SuperNEMO | published_supernemo `TrackerHitTransformer` | voxel 60 mm, <= 128 tokens; coordinate_mlp | 111,233 |

All four use d_model 64, 4 heads, 2 pre-norm GELU layers, feed-forward 256, dropout 0.1, masked mean pooling and the
paper's own head. Inputs are exactly this protocol's reader inputs; NEXT and SuperNEMO hits go through the paper
tokenisers.

**Training (runner `research/paper_baseline/train_supervised.py`).** The paper recipe:
- AdamW, lr 5e-4, weight decay 1e-4, batch 64.
- CosineAnnealingLR(T_max 50) stepped per epoch, no warm-up.
- Gradient clipping 1, BCE without class weights, at most 50 epochs, seed 42.
- Per detector as in the paper runs: bf16 autocast for NEXT and SuperNEMO, fp32 for MJD and EXO-200; SuperNEMO early
  stopping with patience 12 on the energy-matched AUC, the others patience 5 on plain AUC.

Two rules of the frozen fine-tuning recipe (section 5b) apply exactly as for foundation models:
1. Epoch selection and early stopping on the frozen coarse subset of the validation split, not the whole validation
   split.
2. Small label sets are cycled so that early stopping is armed only after 3000 optimizer steps.

Runs, logs and per-run `train.json` are in `runs/paper_region_mlp/<dataset>/L<fraction>/s42/`. Wall time on one RTX
5090 (shared): 1 % and 10 % take 2-60 min; 100 % takes 1.3 h (MJD), 9.6 h (EXO-200), 3.1 h (NEXT) and 4.5 h
(SuperNEMO).

**Energy-regression baseline.** The same Region + MLP backbone trained from scratch under the regression benchmark
is in `validation_regression/README.md` (section "Baseline").

### 7.4 EXO-200 energy: RMSE (primary 1) and sigma/E at 2615 keV (primary 2)

The model predicts each event's energy (regression extension, `validation_regression/`).

* **Energy RMSE**: `sqrt(mean((E_pred - E_ref)^2))` in keV over all test events, across the full
  energy range, with `E_ref` the reference `Rotated_energy`. Lower is better. It measures agreement
  with the reconstruction, not physical resolution.
* **sigma/E at 2615 keV** (2614.5 keV, 208Tl): the peak fit and reportability gates of
  `validation_regression/README.md` (section "Metric"), with the width divided by the local gain.
  Lower is better. The reference reconstruction gives 1.27 % on the test split; it is the physics
  ceiling, not a competitor.

**Decision rule for a comparison.** RMSE decides first. If the two models' RMSEs differ by less
than 2 keV, they are comparable on RMSE, and sigma/E at 2615 keV decides (lower wins).
A model whose sigma/E at 2615 keV is not reportable (any reportability gate fails) has failed the
EXO-200 evaluation, whatever its RMSE.

### 7.5 SuperNEMO opening-angle RMSE (secondary)

`sqrt(mean((pred_deg - phiS)^2))` in degrees, with `phiS` the true escape angle between the two
tracks (simulation truth). Events with `phiS == 0` are excluded and counted; non-finite
predictions are refused. `phiS` stays a forbidden input (card 6.4). RMSE is dominated by tails, so
report it beside the median absolute error.

---

# Part C: extending, costs, agents, files

## 8. Extending the benchmark

### 8.1 Add a dataset

1. Write `validation/readers/<name>.py` implementing the reader contract (4.1), with a doc-string
   that cites the owner's split, label and preprocessing code. Match the owner's semantics exactly:
   the evaluation-positive class, the physical energy field and unit, the split rule and its seed,
   a stable `event_id`, and the list of forbidden fields. Copy any external split definition into
   `validation/manifests/external/`.
2. Register it in `validation/benchmark.json["datasets"]`: `title`, `reader` (`module:Class`),
   `folder` (under `NLDBD_DATA_ROOT`), `task`, `positive_class`, `energy`, `split_source`,
   `coarse_min_fraction` (0.0 unless a reason is recorded), `metrics` (must include the primary).
3. Bump the version (8.3), then `python -m validation.freeze_splits <name>` to create its manifest.
4. Prove it: `python validation/tests/test_readers_match_manifest.py <name> --all-splits` must
   pass, and if the owner has a reference prediction file, add a check like
   `tests/test_readers_match_paper.py` that the ids, labels and energies agree exactly.
5. Add a dataset card (section 6) and, if the files are slow to read, a consolidated copy like the
   NEXT combined file.

### 8.2 Add a metric

1. Implement `compute(ctx) -> {"value", "status", "detail"}` in `validation/metrics/__init__.py`
   (or a module imported there) and register it with `@metric("name", description=...)`. `ctx`
   gives the aligned arrays and the shared EnergyBench record (`ctx.energybench`), so metrics derived
   from that record cost nothing extra.
2. List it in `benchmark.json`: as `primary_metric` (then it decides gate G5 and the report) or in
   `secondary_metrics`, and in each dataset's `metrics` list where it applies.
3. Bump the version (8.3), regenerate the manifests (their energy-only floors are computed with the
   primary metric), add a metric card (section 7).

### 8.3 Versions

`benchmark.json` carries two numbers: `version` (the protocol, bumped on any change) and
`artefact_version` (the frozen evaluator, profiles and manifests). Manifests and `metrics.json`
record the artefact version; `evaluate_predictions.py` refuses a manifest from another artefact
version and `test_frozen.py` checks all manifests carry the declared one. Any change to the evaluator, a profile, a manifest, a registered
dataset's semantics or the primary metric is a new version; results from different versions are
never mixed in one table (the report records the version per run). Changes that leave all frozen
artefacts byte-identical (new secondary metric, new dataset, documentation) are minor versions.

## 9. Costs and the coarse-versus-full agreement (tiny models, lab workstation, RTX 5090)

Scoring time is reading inputs plus model inference, then the evaluator (NumPy; up to ~7 s for
390,000 events, it runs the metric twice for the label-shuffle null, plus a few seconds of id-set
checks on SuperNEMO's 4.7 M training ids). Coarse mode reads its inputs from a one-time local cache
(`validation/coarse_cache.py`, `NLDBD_CACHE_DIR`, default `<project>/cache`, keyed by the manifest's
coarse identity hash; never shipped; ~5.5 GB for the current four subsets).

| Dataset | coarse events | coarse, first pass (builds cache) | coarse, cached | full events | full scoring |
|---|---|---|---|---|---|
| MJD | 10,400 | 8 s | 2 s | 390,000 | 6 to 18 s |
| NEXT | 11,655 | 29 s (combined file) | 4 s | 116,549 | 19 s (combined file; 76 s through the 14,973 files) |
| SuperNEMO | 29,652 | 5 s | 3 to 4 s | 589,824 | 16 s |
| EXO-200 | 37,987 | 28 to 34 s | 7 to 11 s | 140,383 | 90 to 102 s |

Measured agreement of coarse with full for the same weights: MJD within 0.002 (0.5646 vs 0.5658;
0.8836 vs 0.8848), NEXT within 0.001 (0.9015 vs 0.9007), SuperNEMO within 0.005 (0.5581 vs 0.5624),
EXO-200 within 0.025 (0.8131 vs 0.7896; its validation split is one detector run and the test
split four others: a population difference, not subset noise).

Worked example (`examples/train_tiny.py`, 1 % labels unless noted, `runs/tiny/`, `results/results.md`):

| Dataset | Labels | Epochs | Validation AUC | AUC_w coarse | **AUC_w full (test)** | Gates |
|---|---|---|---|---|---|---|
| MJD | 1 % (9,360) | 15 | 0.605 | 0.565 | 0.566 | all PASS |
| MJD | 10 % (93,600) | 10 | 0.899 | 0.884 | 0.885 | all PASS |
| EXO-200 | 1 % (3,366) | 8 | 0.846 | 0.813 | 0.790 | all PASS |
| NEXT | 1 % (9,324) | 8 | 0.907 | 0.902 | 0.901 | all PASS |
| SuperNEMO | 1 % (47,352) | 6 | 0.567 | 0.558 | 0.562 | all PASS |

These numbers only show that the pipeline runs; the models are far below the paper's.

### Metric robustness tests (2026-10-22, `results/metric_robustness.md`, `results/metric_robustness_models.md`)

Properties of AUC_w measured on the tiny models' full-mode predictions: monotone transforms of the
score (affine, sigmoid, rank, cube) change it by < 1e-8; inverting the score gives exactly 1 minus
the value; the standard deviation over random halves of the test split is 0.0006 (MJD) to 0.0013
(NEXT), so differences below ~0.003 are noise even on the full test split; adding standardised
energy to the scores never raises AUC_w (it falls monotonically with the energy weight, because the
pooled pairwise AUC keeps cross-bin pairs and energy-dependent shifts scramble them), while the
inclusive AUC moves wherever the energy correlation points.

Model variants (full mode, AUC_w / inclusive AUC; 10 % labels on MJD, 1 % elsewhere):

| Model | MJD | EXO-200 | NEXT | SuperNEMO |
|---|---|---|---|---|
| tiny network, trained | 0.890 / 0.899 | 0.790 / 0.836 | 0.901 / 0.902 | 0.562 / 0.564 |
| logistic regression on shape features | 0.822 / 0.833 | 0.818 / 0.860 | 0.892 / 0.893 | 0.563 / 0.565 |
| tiny network, untrained | 0.508 / 0.491 | 0.616 / 0.597 | 0.708 / 0.708 | 0.492 / 0.491 |
| tiny network fed the true energy (rule 5 violated on purpose) | 0.849 / 0.861 | 0.599 / 0.566 | 0.873 / 0.911 | 0.533 / 0.665 |
| energy-proxy statistic, no training (not benchmark-valid: no regime) | 0.500 / 0.427 | 0.678 / 0.483 | 0.493 / 0.493 | 0.479 / 0.476 |

Reading: untrained networks sit at chance or at the weak structure of random features; a model
handed the energy gains inclusive AUC where energy separates the classes (SuperNEMO 0.665) but
never gains AUC_w and usually loses it; the pure energy proxy scores 0.500 on MJD. (The EXO-200
amplitude statistic turned out to be a weak topology feature, 0.68 matched above 0.48 inclusive,
so that row is not an energy test there.) A second seed of the tiny MJD 10 % run gave 0.890
against 0.885, reported as 0.8875 ± 0.0038.

### Fine-tuning protocol test (reference foundation model `examples/tiny_fm.py`, 1 % labels, full mode)

| Row | MJD | EXO-200 | NEXT | SuperNEMO |
|---|---|---|---|---|
| `tinyfm__probe` (encoder frozen) | 0.566 | 0.795 | 0.907 | 0.565 |
| `tinyfm__full` (fine-tuned) | 0.827 | 0.868 | 0.914 | 0.566 |
| `tinyfm__scratch` (same architecture, random init) | 0.888 | 0.863 | 0.918 | 0.564 |

The "pretrained" encoder here is the trunk of the tiny classifier trained on the same 1 % regime,
so the probe reproduces that classifier, full fine-tuning improves on it, and the scratch control
matches or beats it: the protocol reports honestly when pretraining buys nothing. Coarse-mode values
of the same runs track the full-mode ones within 0.01 (MJD 0.829 vs 0.827; EXO-200 0.876 vs 0.868).

## 10. Using this skill with an agent (or by hand)

* **Get a writable copy.** Workstation source: `/home/tidmad/foundation_model/`. NERSC (read-only
  for everyone): `/global/cfs/cdirs/gen0032/foundation_model/`, with a ready environment
  `/global/cfs/cdirs/gen0032/envs/fm/bin/python` (NumPy, h5py). Copy the folder into your
  own project; `runs/`, `results/` and `cache/` are written there. Optionally install this file as
  an agent skill (Claude Code: `~/.claude/skills/nldbd-validation/SKILL.md`) so it triggers itself.
* **Give the agent one sentence**, e.g. "Evaluate `my_model.py` with validation_skill.md on all
  datasets, full mode, three label regimes, seed 42" or "Add dataset X following section 8.1".
  The agent must start with the self-check and stop if it does not pass.
* **What the agent may decide**: model architecture, training recipe, how to implement the
  two-function scoring module, how many seeds. **What it may not decide**: anything in Part A; the
  files with hashes settle those questions.
* **What to hand back**: the reporting checklist in section 5, nothing less.
* The data: workstation `/home/klz/Data/zeronu_benchmark/` (plus `/home/klz/Data/NEXT_combined/`);
  NERSC `/global/cfs/cdirs/gen0032/data/zeronu_benchmark/` (plus `/global/cfs/cdirs/gen0032/data/NEXT_combined/`),
  same layout, read-only for everyone. On NERSC set `NLDBD_DATA_ROOT=/global/cfs/cdirs/gen0032/data/zeronu_benchmark`
  and `NLDBD_NEXT_COMBINED=/global/cfs/cdirs/gen0032/data/NEXT_combined/next_combined.h5`. Anything
  heavier than the self-check on NERSC runs in a Slurm allocation (account `gen0044_g` since 2026-10-25; earlier `gen0032_g`),
  not on a login node.
* `USAGE.md` beside this file is the step-by-step instruction for a first-time user.

## 11. File map

```
validation_skill.md                 this protocol
USAGE.md                            step-by-step instruction for a new user (human + agent)
validation/
  benchmark.json                    THE declaration: version, profile, metrics, regimes, coarse rule, datasets
  energybench/                      frozen paper evaluator (metrics.py, evaluate.py, configs/, tests/, SHA256SUMS, PROVENANCE.md)
  metrics/__init__.py               metric registry (matched_auc primary, inclusive_auc secondary)
  readers/{common,mjd,exo200,next,supernemo}.py   dataset readers (contract in common.py; registry via benchmark.json)
  manifests/<ds>.json, <ds>_ids.npz frozen splits, regimes, coarse subsets, floors; external/ = the owners' split definitions
  finetune_recipe.json              THE frozen fine-tuning recipe (section 5b)
  fm_contract.md                    the foundation-model interface (EMBED_DIM, to_tensors, load_encoder)
  finetune.py                       fine-tunes an encoder (probe / full / scratch) and evaluates it in both modes
  freeze_splits.py                  creates a manifest (new dataset or new version only)
  build_next_combined.py            one-time consolidation of the NEXT release
  make_predictions.py               model module -> predictions.npz (test split, or coarse subset via the cache)
  coarse_cache.py                   one-time local cache of coarse-subset inputs
  evaluate_predictions.py           predictions + training ids -> metrics.json with gates (both modes)
  report.py                         runs/ -> results/results.md, results.csv (primary or any registered metric)
  baselines.py                      optional, not run in this release: re-score the paper's saved predictions
  selfcheck.sh, setup_env.sh, requirements.txt, tests/
examples/tiny_models.py, examples/train_tiny.py   executable reference classifier implementation
examples/tiny_fm.py                 reference foundation model for the fine-tuning protocol
examples/energy_proxy_models.py, examples/metric_robustness.py, examples/train_small_variants.py   metric robustness tests (section 9)
runs/, results/                     outputs (layout in 4.4)
README.md, nersc_upload.sh          entry note; mirror to NERSC gen0032 (read-only for everyone, zpli may write)
```
