#!/usr/bin/env python3
"""Evaluate predicted energies of one split (nldbd-energy-resolution).

Input: predictions.npz with ``event_id`` (the split's canonical order, all events) and ``energy_keV`` (predicted);
optionally training_ids.npz (``event_id``: every event whose inputs or energies were used in training / selection),
which must not intersect the evaluated split. Output: metrics.json beside the predictions (or --output).

Usage: python -m validation_regression.evaluate --dataset mjd --split test --predictions run/predictions.npz \
           [--training-ids run/training_ids.npz] [--model NAME] [--output DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from validation_regression.score import score  # noqa: E402
from validation_regression.views import BENCHMARK, DATASETS, get_view  # noqa: E402


def evaluate(dataset, split, pred_file, training_ids=None, model=None, table=None):
    t = get_view(dataset).table(split) if table is None else table
    z = np.load(pred_file, allow_pickle=False)
    ids, e = z["event_id"].astype(str), z["energy_keV"].astype(np.float64)
    if not np.array_equal(ids, t["event_id"]):
        raise SystemExit(f"{pred_file}: event_id must be the {dataset} {split} table in canonical order ({len(t['event_id'])} events)")
    integrity = {"ids_match_split": True}
    if training_ids is not None:
        used = np.load(training_ids, allow_pickle=False)["event_id"].astype(str)
        overlap = int(np.isin(t["event_id"], used).sum())
        integrity.update({"training_ids": str(training_ids), "n_training_ids": int(len(used)), "overlap_with_split": overlap})
        if overlap:
            raise SystemExit(f"{overlap} evaluated {split} events appear in the training ids: not a valid evaluation")
    m = score(dataset, e, t)
    m.update({"benchmark": f"{BENCHMARK['name']} {BENCHMARK['version']}", "split": split, "model": model,
              "predictions": str(pred_file), "predictions_sha256": hashlib.sha256(Path(pred_file).read_bytes()).hexdigest(),
              "integrity": integrity, "evaluated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    return m


def headline_line(m):
    h = m["headline"]; v = h.get("value")
    if not h.get("reportable") or v is None:
        return f"{m['dataset']}: {h['name']} not reportable {h.get('reasons', '')}"
    if m["dataset"] == "exo200":
        return f"{m['dataset']}: {h['name']} = {100 * v:.3f} +- {100 * h['err']:.3f} %"
    extra = f" (trivial floor {h['trivial_floor_keV']:.0f} keV)" if "trivial_floor_keV" in h else ""
    return f"{m['dataset']}: {h['name']} = {v:.3f} +- {h['err']:.3f}{extra}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=DATASETS); ap.add_argument("--split", default="test", choices=("validation", "test"))
    ap.add_argument("--predictions", required=True, type=Path); ap.add_argument("--training-ids", type=Path)
    ap.add_argument("--model"); ap.add_argument("--output", type=Path)
    a = ap.parse_args()
    m = evaluate(a.dataset, a.split, a.predictions, a.training_ids, a.model)
    out = (a.output or a.predictions.parent) / "metrics.json"
    out.write_text(json.dumps(m, indent=2, default=float) + "\n")
    print(headline_line(m)); print(f"wrote {out}")
