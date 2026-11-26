"""Unit tests of the EXO-200 reader that need no raw data (small synthetic run files), plus consistency checks of the shipped definition.

Usage: python validation/tests/test_exo200_reader.py
Checks: blacklisted events are removed and the kept rows stay ascending; a blacklist row that is not in the file is an error; the label is
Charge_Cluster_Number == 1; raw reads agree for contiguous and non-contiguous rows; the reader refuses release v1 files, a wrong waveform layout,
a wrong waveform length, missing files, and a blacklist copy next to the data that differs; the repository's blacklist is the one the split
definition was built with; the split definition is internally consistent.
"""
import csv
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from validation.readers import exo200  # noqa: E402


def make_run(path, n=40, channels=4, samples=30, seed=0):
    rng = np.random.default_rng(seed)
    ev = np.arange(2, 2 + 3 * n, 3)[:n].astype(np.int32)            # stored event numbers (not row numbers)
    ncl = rng.integers(1, 4, n).astype(np.int8)
    with h5py.File(path, "w") as f:
        f["Event_Number"] = ev
        f["Charge_Cluster_Number"] = ncl
        f["Rotated_Energy"] = rng.uniform(500, 3500, n).astype(np.float16)
        f["Waveforms"] = rng.integers(800, 4000, (n, channels, samples)).astype(np.int16)
    return ev, ncl


def folder(path, first, blacklist_copy=None):
    """A data folder with the 15 expected file names; only the first file has content (the reader inspects that one)."""
    split = json.loads(exo200.SPLIT_FILE.read_text())
    for i, name in enumerate(split["expected_files"]):
        with h5py.File(Path(path) / name, "w") as f:
            if i == 0:
                first(f)
    if blacklist_copy is not None:
        (Path(path) / exo200.DATA_BLACKLIST_NAME).write_bytes(blacklist_copy)


def refuses(root, needle):
    try:
        exo200.EXO200Reader(root)
    except (FileNotFoundError, RuntimeError) as e:
        return needle in str(e)
    return False


def check(name, cond):
    print(f"{'OK ' if cond else 'BAD'} {name}")
    return bool(cond)


def main():
    ok = True
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "8999.h5"
        ev, ncl = make_run(p)
        listed = {int(ev[3]), int(ev[10]), int(ev[11])}
        r = exo200.read_run_table(p, listed)
        ok &= check("blacklisted events removed", not set(r["event_number"].tolist()) & listed and len(r["event_number"]) == len(ev) - 3)
        ok &= check("n_removed", r["n_removed"] == 3)
        ok &= check("kept rows ascending and index the original file", np.all(np.diff(r["row"]) > 0) and np.array_equal(ev[r["row"]], r["event_number"]))
        ok &= check("label source is the cluster count", np.array_equal(r["n_clusters"], ncl[r["row"]]))
        ok &= check("energies read as float64", r["energy_keV"].dtype == np.float64)
        try:
            exo200.read_run_table(p, {10 ** 9}); ok &= check("unknown blacklist event is an error", False)
        except RuntimeError:
            ok &= check("unknown blacklist event is an error", True)
        rd = exo200.EXO200Reader.__new__(exo200.EXO200Reader); rd.files = [p]
        with h5py.File(p, "r") as f:
            wf = f["Waveforms"][()]
        ok &= check("contiguous raw read", np.array_equal(rd.read_raw(0, np.arange(5, 12)), wf[5:12]))
        rows = np.array([1, 4, 5, 9, 30])
        ok &= check("non-contiguous raw read", np.array_equal(rd.read_raw(0, rows), wf[rows]))
        x = exo200.EXO200Reader.preprocess(wf[:6])
        ok &= check("preprocess: baseline-subtracted, max |x| = 1 per event", np.allclose(np.abs(x).reshape(6, -1).max(1), 1.0))

    # the reader must refuse anything that is not release v3, with a message that says what to do
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "8956-000.h5").write_bytes(b"")
        ok &= check("release v1 file names are refused", refuses(d, "release v1"))
    with tempfile.TemporaryDirectory() as d:
        folder(d, lambda f: f.create_group("Waveforms"))
        ok &= check("per-event Waveforms group (v1 layout) is refused", refuses(d, "group of per-event arrays"))
    with tempfile.TemporaryDirectory() as d:
        folder(d, lambda f: f.create_dataset("Waveforms", data=np.zeros((2, 226, 300), np.int16)))
        ok &= check("300-sample waveforms are refused", refuses(d, "found shape (2, 226, 300)"))
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "8956.h5").write_bytes(b"")
        ok &= check("missing files are refused", refuses(d, "release v3"))
    with tempfile.TemporaryDirectory() as d:
        folder(d, lambda f: f.create_dataset("Waveforms", data=np.zeros((2, 226, 600), np.int16)), blacklist_copy=b"run,event_number\n8956,1\n")
        ok &= check("a different blacklist copy next to the data is refused", refuses(d, "differs from the repository's blacklist"))
    with tempfile.TemporaryDirectory() as d:
        folder(d, lambda f: f.create_dataset("Waveforms", data=np.zeros((2, 226, 600), np.int16)), blacklist_copy=exo200.BLACKLIST_FILE.read_bytes())
        try:
            exo200.EXO200Reader(d); ok &= check("correct layout and identical blacklist copy are accepted", True)
        except Exception as e:
            ok &= check(f"correct layout and identical blacklist copy are accepted ({e})", False)

    split = json.loads(exo200.SPLIT_FILE.read_text())
    raw = exo200.BLACKLIST_FILE.read_bytes()
    ok &= check("repository blacklist matches the split definition (sha256)", hashlib.sha256(raw).hexdigest() == split["blacklist"]["sha256"])
    rows = list(csv.DictReader(raw.decode().splitlines()))
    keys = {(int(r["run"]), int(r["event_number"])) for r in rows}
    ok &= check("blacklist keys unique and counted", len(keys) == len(rows) == split["blacklist"]["rows"])
    runs = [r for s in split["runs"].values() for r in s]
    ok &= check("runs are disjoint across splits", len(runs) == len(set(runs)) and split["expected_files"] == [f"{r}.h5" for r in sorted(runs)])
    ok &= check("blacklisted runs are all in the release", {r for r, _ in keys} <= set(runs))
    info = split["run_event_counts"]
    ok &= check("per-run counts add up", sum(info.values()) == split["expected_events"]
                and {int(r) for r in info} == set(runs) and all(sum(info[str(r)] for r in rr) == split["counts"][s] for s, rr in split["runs"].items()))
    ok &= check("split counts add up", sum(split["counts"].values()) == split["expected_events"] == split["expected_events_in_files"] - len(rows))
    ok &= check("class counts add up", all(sum(split["class_counts"][s].values()) == split["counts"][s] for s in split["counts"]))
    print("RESULT", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
