#!/usr/bin/env python3
"""Build manifests/external/exo200_split.json from the EXO-200 open-data files and the event blacklist.

The data are the EXO-200 open-data release v3 (15 files <run>.h5, 600-sample waveforms). The run-grouped train / validation / test
assignment below is the data owner's seed-42 selection, carried over unchanged from the split definition used with the earlier release (v1),
so the three splits contain the same runs as before; only the events differ (v3 files hold more events per run, and the blacklisted events are
removed). Every count in the output is read from the data, none is typed in.

Usage: NLDBD_EXO200_DATA=<folder with the 15 <run>.h5 files> python -m validation.make_exo200_split [--check]
With --check the file is not written; the script exits 1 if the existing file's content (not its layout) differs from what the data give.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from validation.readers.common import MANIFESTS, data_root  # noqa: E402

EXTERNAL = MANIFESTS / "external"
BLACKLIST = EXTERNAL / "exo200_event_blacklist.csv"
OUT = EXTERNAL / "exo200_split.json"
SEED = 42
RUNS = {
    "train": [8956, 8957, 8958, 8968, 8972, 8999, 9000, 9001, 9008, 9009],
    "validation": [8969],
    "test": [8967, 8970, 8971, 9010],
}


def build(root: Path) -> dict:
    all_runs = sorted(r for s in RUNS.values() for r in s)
    raw = BLACKLIST.read_bytes()
    rows = list(csv.DictReader(raw.decode().splitlines()))
    bl = {}
    for r in rows:
        bl.setdefault(int(r["run"]), set()).add(int(r["event_number"]))
    if sum(len(v) for v in bl.values()) != len(rows):
        raise SystemExit("blacklist has duplicate (run, event_number) keys")
    if set(bl) - set(all_runs):
        raise SystemExit(f"blacklist lists runs not in the release: {sorted(set(bl) - set(all_runs))}")
    files = [f"{r}.h5" for r in all_runs]
    run_info, class_counts = {}, {s: {"signal": 0, "background": 0} for s in RUNS}
    counts = {s: 0 for s in RUNS}
    total_in_files = total_bl = 0
    for run in all_runs:
        split = next(s for s, rr in RUNS.items() if run in rr)
        with h5py.File(root / f"{run}.h5", "r") as f:
            ev = f["Event_Number"][()].astype(np.int64)
            ncl = f["Charge_Cluster_Number"][()].astype(np.int64)
            shape = f["Waveforms"].shape
        if len(np.unique(ev)) != len(ev):
            raise SystemExit(f"run {run}: Event_Number is not unique")
        if shape[1:] != (226, 600) or shape[0] != len(ev):
            raise SystemExit(f"run {run}: waveforms are {shape}, expected ({len(ev)}, 226, 600)")
        listed = bl.get(run, set())
        bad = np.isin(ev, np.fromiter(listed, dtype=np.int64, count=len(listed)))
        if int(bad.sum()) != len(listed):
            raise SystemExit(f"run {run}: {len(listed)} blacklist rows but {int(bad.sum())} matching events")
        keep = ~bad
        sig = int((ncl[keep] == 1).sum())
        run_info[str(run)] = int(keep.sum())
        total_in_files += int(len(ev)); total_bl += int(bad.sum())
        counts[split] += int(keep.sum()); class_counts[split]["signal"] += sig; class_counts[split]["background"] += int(keep.sum()) - sig
    return {
        "schema_version": 1,
        "dataset": "EXO-200 Open Source Data Release for AI/ML Applications",
        "dataset_version": "v3",
        "dataset_doi": "10.5281/zenodo.20419164",
        "supersedes": "release v1 (29 shards <run>-<part>.h5, 300-sample windows that cut signals off, 515,002 events), which this benchmark no longer reads",
        "waveform": {"channels": 226, "samples": 600, "window_us": 600, "pre_trigger_us": 200, "dtype": "int16",
                     "layout": "one file per run: Waveforms (N, 226, 600), plus Event_Number, Charge_Cluster_Number, Rotated_Energy, ..."},
        "expected_hdf5_files": len(files),
        "expected_files": files,
        "expected_events_in_files": total_in_files,
        "expected_events": total_in_files - total_bl,
        "blacklist": {
            "file": BLACKLIST.name, "sha256": hashlib.sha256(raw).hexdigest(), "rows": len(rows), "key": ["run", "event_number"],
            "action": "events listed are removed from every split (train, validation and test)",
            "reason": "events with incorrect labels (data-owner addendum); the CSV 'reason' column reads multiple_reconstructed_scintillation_clusters",
            "note": "the 'split' column of the CSV belongs to an earlier project's split and is not used",
        },
        "seed": SEED,
        "split_method": "run-grouped; run assignment carried over unchanged from the split definition of release v1",
        "preprocessing": {"baseline_samples": 200, "classification_amplitude_normalization": True},
        "label_definition": {"signal": "Charge_Cluster_Number == 1", "background": "Charge_Cluster_Number > 1", "positive_roc_label": 1},
        "counts": counts,
        "class_counts": class_counts,
        "runs": RUNS,
        "overlap_counts": {"train_validation": 0, "train_test": 0, "validation_test": 0},
        "run_event_counts": run_info,
    }


def dumps(x, depth=0) -> str:
    """JSON in the layout of the earlier split file: lists of scalars and small dicts on one line, long string lists two per line."""
    pad = "  " * (depth + 1)
    if isinstance(x, dict):
        if depth >= 2 and not any(isinstance(v, (dict, list)) for v in x.values()):
            return json.dumps(x)
        return "{\n" + ",\n".join(f"{pad}{json.dumps(k)}: {dumps(v, depth + 1)}" for k, v in x.items()) + "\n" + "  " * depth + "}"
    if isinstance(x, list):
        one = json.dumps(x)
        if len(one) <= 80:
            return one
        rows = [", ".join(json.dumps(v) for v in x[i:i + 2]) for i in range(0, len(x), 2)]
        return "[\n" + ",\n".join(pad + r for r in rows) + "\n" + "  " * depth + "]"
    return json.dumps(x)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    data = build(data_root("exo200"))
    new = dumps(data) + "\n"
    if a.check:
        same = OUT.exists() and json.loads(OUT.read_text()) == data
        print("OK exo200_split.json matches the data" if same else "DIFFERS from the data")
        sys.exit(0 if same else 1)
    OUT.write_text(new)
    print(f"wrote {OUT}: {data['expected_events_in_files']} events in files, {data['blacklist']['rows']} blacklisted, counts {data['counts']}")


if __name__ == "__main__":
    main()
