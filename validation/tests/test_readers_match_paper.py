"""Check that a reader reproduces the paper's test population exactly.

Usage: python -m tests.test_readers_match_paper <dataset> [path to the paper's event_data folder]
Compares event ids (set and canonical order), labels and energies with the paper's saved
prediction file for the Region+MLP model (or the best classic model if no transformer file exists).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from validation.readers import get_reader  # noqa: E402

PAPER_FILE = {"mjd": "events/MJD__region_mlp.npz", "exo200": "events/EXO-200__region_mlp.npz",
              "next": "events/NEXT__gine.npz", "supernemo": "events/SuperNEMO__gine.npz"}


def main(dataset: str, event_data: str | None = None):
    event_data = Path(event_data or os.environ.get("NLDBD_PAPER_EVENT_DATA",
                      "/home/wenyu/iclr final paper/EnergyBench_reproducibility_20260923/event_data"))
    paper = np.load(event_data / PAPER_FILE[dataset], allow_pickle=False)
    t0 = time.time()
    t = get_reader(dataset).table("test")
    print(f"{dataset}: read test table in {time.time() - t0:.1f} s, {len(t['event_id'])} events")
    pid = paper["event_id"].astype(str)
    assert len(pid) == len(t["event_id"]), (len(pid), len(t["event_id"]))
    same_order = np.array_equal(pid, t["event_id"])
    print("  canonical order identical:", same_order)
    if not same_order:
        assert set(pid) == set(t["event_id"]), "event id SETS differ"
        order = {e: i for i, e in enumerate(t["event_id"])}
        idx = np.array([order[e] for e in pid])
    else:
        idx = np.arange(len(pid))
    lab_ok = np.array_equal(paper["label"].astype(np.int8), t["label"][idx])
    de = np.abs(paper["energy_keV"] - t["energy_keV"][idx])
    print("  labels identical:", lab_ok, "| label counts paper", np.bincount(paper["label"].astype(int)),
          "reader", np.bincount(t["label"]))
    print("  energy max |diff| keV:", float(np.nanmax(de)), "| exact:", bool(np.array_equal(paper["energy_keV"], t["energy_keV"][idx])))
    if "group" in paper.files:
        print("  groups identical:", np.array_equal(paper["group"].astype(str), t["group"][idx]))
    ok = lab_ok and np.nanmax(de) == 0.0
    print("RESULT", dataset, "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main(*sys.argv[1:]) else 1)
