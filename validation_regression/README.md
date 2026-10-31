# nldbd-energy-resolution (v1.0.0-dev): energy-regression extension

A separate benchmark beside the frozen classification packet `validation/` (nldbd-energy-matched-auc v1.2). It imports
the classification readers, splits and labeled subsets **read-only** and never modifies them. One foundation model
(the same `fm_contract.md` module) serves both tasks.

## Task
Predict each event's energy from the detector inputs. Score the **physics resolution** of the predicted spectrum on
the frozen test split, not MSE.

| Dataset | Target (reference) | Inputs | Headline |
|---|---|---|---|
| MJD | `energy_label` | classification waveform view + log(scale) + detector index (26 detectors) | FWHM at Q_bb = 2039 keV from FWHM(E) = sqrt(a + bE + cE^2) through the Th-228 lines (>= 4 reportable, incl. 2614.5); also FWHM at 2614.5 |
| EXO-200 | `Rotated_energy` | classification waveform view + log(scale) | sigma/E at 2614.5 keV |
| SuperNEMO | E1 + E2 (calorimeter) | tracker hits only (tX tY tZ tR) | FWHM of E_pred - E_ref over the 0nubb test events, beside the trivial floor (constant prediction) |

NEXT is excluded: its simulation is a narrow window around Q_bb, and its energy is the sum of the input hit energies.

**Scale.** It is the per-event normalisation factor (max |x| after baseline subtraction). With it, the
amplitude-normalised view is lossless: `exp(log_scale) * input` = the raw baseline-subtracted waveform (tested). No
reconstructed-energy field is ever an input.

**SuperNEMO sample.** Training and selection use 0nubb only (user 2026-10-25); the test table holds 0nubb + 2nubb + Bi214 in manifest blocks with the
same 80/10/10 split as classification. 0nubb never enters classification or its pretraining (tested), so its test
blocks (196,608 events) are a clean test set with a peak.

## Metric (resolution.py, score.py)
- **Peak fit.** Binned Poisson-likelihood fit of the predicted spectrum around each line. The window iterates to
  ±8 sigma (MJD; at least ±6 keV) or ±5 sigma (EXO-200, which keeps the 2382 keV Compton edge out).
- **Shape.** Gaussian, plus low- and high-energy exponential tails, plus an erfc step, on a linear background. Known
  Th-228 lines inside the window are fitted with the same shape. FWHM is the half-maximum width of the full shape.
- **Local gain.** The FWHM is divided by the slope of E_pred vs E_ref, measured on events whose *reference* energy
  lies in sidebands of the line. A regressor cannot narrow a peak by compressing the energy scale around it.
- **Reportability gates** (benchmark.json). A line is "not reportable", never silently filled, if any of these fail:
  - signal >= 300 events
  - fit converged
  - chi2/ndf < 3
  - |centroid - line| < 3 FWHM
  - local gain within 1 ± 0.2 (widened from 0.1 on 2026-10-25: the FWHM is already divided by the gain)

## Reference values (test split, metric applied to the reference energies)
| | value |
|---|---|
| MJD FWHM at Q_bb | 2.40 keV (curve through 7 lines; 2614.5: 2.84 keV) |
| EXO-200 sigma/E at 2614.5 | 1.27 % |
| SuperNEMO 0nubb residual | 0 by construction; trivial floor 381 keV; reference half-max width of the 0nubb peak 288 keV |

## Protocol (recipe.json, finetune.py)
- **Modes.** probe (frozen encoder, cached embeddings) / full / scratch. Label regimes 1 / 10 / 100 % use the frozen
  nested labeled subsets of the regression train table.
- **Head.** log E = c[det] + a·log_scale + MLP([LayerNorm(pooled embedding), log_scale, detector embedding]).
  c and a start at the least-squares amplitude calibration on the labeled events; the MLP's last layer starts at zero.
- **Training.** Huber loss in units of 1 % relative error, AdamW, cosine schedule. Early stopping uses the loss on a
  fixed validation selection subset.
- **Test split.** Touched once, after training. The evaluator refuses predictions that are not the full test table
  in canonical order, and refuses training ids that intersect the test split.

## Simple reference fits (baselines.py; calibrations fitted on the 10 % labeled subset; context, not the baseline)
- reference (ceiling)
- MJD amplitude × per-detector gain
- EXO-200 sum of channel maxima
- EXO-200 per-channel linear calibration
- SuperNEMO track-feature ridge regression
- constant (floor)

## Baseline (official, 2026-10-25): Region + MLP Transformer trained from scratch

**Every model is compared with this table.** Do not compare with the reference reconstruction (it is the physics
ceiling, not a competitor).
- Model: the paper's Region + MLP backbone (`research/paper_baseline/region_mlp_encoder.py`, the paper code with its
  scalar head removed), trained with `finetune.py --mode scratch`. The frozen regression recipe and head apply, exactly
  as for a foundation model.
- Split: test; seed 42.
- SuperNEMO: trained and selected on 0nubb only.

| Labels | MJD FWHM at Q_bb (keV) | EXO-200 sigma/E at 2614.5 keV | SuperNEMO 0nubb residual FWHM (keV) |
|---|---|---|---|
| 1 % | 7.8 ± 1.4 | 5.88 ± 0.24 % | 351.6 ± 1.1 |
| 10 % | 8.3 ± 3.0 | 3.96 ± 0.14 % | 351.3 ± 1.0 |
| 100 % | 4.5 ± 0.9 | 2.72 ± 0.03 % | pending (NERSC fm100 queue) |

Context only, never the comparison:
- Reference reconstruction: MJD 2.40 keV, EXO-200 1.27 %.
- Trivial floor (constant prediction): SuperNEMO 381 keV.
- Simple non-AI fits at 10 %: MJD amplitude × gain 10.6 keV; EXO-200 per-channel linear 5.26 %; SuperNEMO track
  features 350 keV.

Training only on events near the line (EXO-200 ±3 sigma / ±5 sigma windows, research option `--train-window-keV`) does
not work: the regressor predicts about 2.6 MeV for every event, the local gain drops to 0 and the line is not
reportable (runs `paper_region_mlp_win3s`, `paper_region_mlp_win5s`).

## Usage
```
python -m validation_regression.tests.test_metric        # metric self-tests (reference, smearing, compression, shuffle)
python -m validation_regression.tests.test_views         # view equivalence / losslessness / sample isolation
python -m validation_regression.baselines --dataset mjd --label-fraction 0.1
python -m validation_regression.finetune --fm-module my_fm.py --name myfm --dataset mjd --label-fraction 0.1 --mode probe
python -m validation_regression.evaluate --dataset mjd --split test --predictions run/predictions.npz --training-ids run/training_ids.npz
```
Environment: `NLDBD_DATA_ROOT` (as for the classification packet) and `NLDBD_REGRESSION_RUNS` (output root, default
`runs_regression/`).
