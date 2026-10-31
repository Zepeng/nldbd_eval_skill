#!/usr/bin/env python3
"""Simple physics baselines for the energy-regression task (no foundation model).

* reference: the reference reconstruction itself (the ceiling; residual 0 by construction)
* mjd amplitude_gain: max |waveform| after baseline subtraction (the view's scale) x per-detector linear gain + offset
* exo200 sum_of_maxima: sum over channels of the per-channel maximum, linear calibration
* exo200 channel_linear: ridge regression on the 226 per-channel maxima (a per-channel gain calibration)
* supernemo track_features: ridge regression on simple tracker features (hit count, extents, path length)
* constant: median train energy (the trivial floor)
Calibrations are fitted on the labeled subset of the regression train table (--label-fraction, frozen subsets).

Usage: python -m validation_regression.baselines --dataset mjd [--label-fraction 0.1] [--split test]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from validation.freeze_splits import labeled_subset_positions  # noqa: E402
from validation.readers.common import BENCHMARK as CLS_BENCHMARK  # noqa: E402
from validation_regression.evaluate import evaluate, headline_line  # noqa: E402
from validation_regression.views import get_view  # noqa: E402

RUNS = Path(os.environ.get("NLDBD_REGRESSION_RUNS", ROOT / "runs_regression"))


def features(dataset, view, split, table, positions=None):
    """Per-event baseline features in table position order (positions=None: whole split)."""
    pos_all = np.arange(len(table["event_id"])) if positions is None else np.asarray(positions)
    where = np.full(len(table["event_id"]), -1, dtype=np.int64); where[pos_all] = np.arange(len(pos_all))
    F = None
    for pos, x, aux in view.iter_regression(split, indices=pos_all, table=table):
        if dataset == "mjd":
            f = np.stack([np.exp(aux["log_scale"]), aux["detector"].astype(np.float32)], 1)
        elif dataset == "exo200":
            f = np.asarray(x).max(axis=2) * np.exp(aux["log_scale"])[:, None]          # per-channel maxima, ADC
        else:
            f = np.stack([track_features(h) for h in x])
        if F is None:
            F = np.full((len(pos_all), f.shape[1]), np.nan, dtype=np.float64)
        F[where[pos]] = f
    return F


def track_features(h):
    h = np.asarray(h, dtype=np.float64); xyz = h[:, :3]; n = len(h)
    if n == 0:
        return np.zeros(8)
    ext = xyz.max(0) - xyz.min(0)
    order = np.argsort(xyz[:, 0]); path = np.linalg.norm(np.diff(xyz[order], axis=0), axis=1).sum() if n > 1 else 0.0
    r = h[:, 3]; r = r[np.isfinite(r)]
    return np.array([n, *ext, path, np.linalg.norm(ext), r.mean() if len(r) else 0.0, np.isfinite(h[:, 3]).mean()])


def ridge(X, y, lam=1e-3):
    mu, sd = X.mean(0), X.std(0) + 1e-12; Z = (X - mu) / sd; Z = np.c_[Z, np.ones(len(Z))]
    w = np.linalg.solve(Z.T @ Z + lam * len(Z) * np.diag(np.r_[np.ones(Z.shape[1] - 1), 0.0]), Z.T @ y)
    return lambda Xn: np.c_[(Xn - mu) / sd, np.ones(len(Xn))] @ w


def fit_predict(name, dataset, Ftr, ytr, Fte):
    if name == "constant":
        return np.full(len(Fte), np.median(ytr))
    if name == "amplitude_gain":
        out = np.full(len(Fte), np.nan); det_tr, det_te = Ftr[:, 1].astype(int), Fte[:, 1].astype(int)
        glob = np.polyfit(Ftr[:, 0], ytr, 1)
        for d in np.unique(det_te):
            m = det_tr == d; p = np.polyfit(Ftr[m, 0], ytr[m], 1) if m.sum() >= 20 else glob
            out[det_te == d] = np.polyval(p, Fte[det_te == d, 0])
        return out
    if name == "sum_of_maxima":
        p = np.polyfit(Ftr.sum(1), ytr, 1); return np.polyval(p, Fte.sum(1))
    if name in ("channel_linear", "track_features"):
        return ridge(Ftr, ytr)(Fte)
    raise ValueError(name)


NAMES = {"mjd": ("amplitude_gain", "constant"), "exo200": ("sum_of_maxima", "channel_linear", "constant"),
         "supernemo": ("track_features", "constant")}


def main(a):
    view = get_view(a.dataset); tr = view.table("train"); te = view.table(a.split)
    pos = labeled_subset_positions(len(tr["event_id"]), a.label_fraction, CLS_BENCHMARK["label_regimes"]["seed"])
    t0 = time.time(); Ftr = features(a.dataset, view, "train", tr, pos); Fte = features(a.dataset, view, a.split, te)
    print(f"[{a.dataset}] features: {len(pos)} calibration + {len(Fte)} {a.split} events in {time.time() - t0:.0f} s", flush=True)
    ytr = tr["energy_keV"][pos]
    for name in ["reference", *NAMES[a.dataset]]:
        run = RUNS / "baselines" / name / a.dataset / f"L{a.label_fraction:g}"; run.mkdir(parents=True, exist_ok=True)
        e = te["energy_keV"].copy() if name == "reference" else fit_predict(name, a.dataset, Ftr, ytr, Fte)
        np.savez_compressed(run / f"predictions_{a.split}.npz", event_id=te["event_id"], energy_keV=e)
        np.savez_compressed(run / "training_ids.npz", event_id=tr["event_id"][pos])
        m = evaluate(a.dataset, a.split, run / f"predictions_{a.split}.npz", run / "training_ids.npz", f"baseline:{name}", table=te)
        (run / f"metrics_{a.split}.json").write_text(json.dumps(m, indent=2, default=float) + "\n")
        print(f"  {name:15s} {headline_line(m)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=tuple(NAMES)); ap.add_argument("--label-fraction", type=float, default=0.1)
    ap.add_argument("--split", default="test", choices=("validation", "test"))
    main(ap.parse_args())
