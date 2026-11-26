"""EXO-200 open data (release v3) sensor waveforms.

Data: "EXO-200 Open Source Data Release for AI/ML Applications", release v3 (DOI 10.5281/zenodo.20419164): 600 us = 600-sample windows
(200 us before and 400 us after the trigger). This release replaces v1, whose 300-sample windows cut signals off; the reader refuses anything else
and says so (see ``_find_files``).

* 15 files ``<run>.h5`` (one per run); ``Waveforms`` is one (N, 226, 600) int16 array, plus ``Event_Number`` (unique within a run),
  ``Charge_Cluster_Number``, ``Rotated_Energy`` (float16 in the file, read as float64) and further reconstructed quantities;
* blacklist: ``manifests/external/exo200_event_blacklist.csv`` lists (run, event_number) pairs of events with incorrect labels
  (data-owner addendum). They are removed from EVERY split before anything else happens (counts, ids, labels, energies, inputs).
  The copy in the repository is authoritative; a copy found next to the data (``event_blacklist_v1.csv``) must be byte-identical;
* label: ``Charge_Cluster_Number == 1`` is signal-like (evaluation-positive, label 1), more clusters is background (label 0);
* energy: ``Rotated_Energy`` (keV). It is stored as float16, so values above 2,048 keV have a resolution of 2 keV;
* split (run-grouped, ``manifests/external/exo200_split.json``): train runs 8956 8957 8958 8968 8972 8999 9000 9001 9008 9009,
  validation 8969, test 8967 8970 8971 9010; within a split, files in sorted order and rows ascending;
* event_id: ``"EXO200::8967.h5::4"`` (file name, stored ``Event_Number``);
* forbidden inputs: every energy field, ``Charge_Cluster_Number``, ``Charge_Cluster_Energy`` / ``Charge_Cluster_Position`` (their shape or
  content encodes the label), ``Event_Number``, the run number;
* input: waveforms as float32, per-channel mean of the first 200 samples subtracted, the whole event divided by its maximum absolute
  amplitude (skipped when 0) -> ``[B, 226, 600]``.
"""
from __future__ import annotations

import csv
import hashlib
import re

import h5py
import numpy as np

from .common import MANIFESTS, Reader, load_json

BASELINE_SAMPLES = 200
SAMPLES = 600
CHANNELS = 226
SPLIT_FILE = MANIFESTS / "external" / "exo200_split.json"
BLACKLIST_FILE = MANIFESTS / "external" / "exo200_event_blacklist.csv"
DATA_BLACKLIST_NAME = "event_blacklist_v1.csv"  # name of the owner's copy shipped with the data

NEEDS_RELEASE = ("This benchmark uses the EXO-200 open-data release v3 (15 files <run>.h5, waveforms 226 x 600; DOI 10.5281/zenodo.20419164). "
                 "Release v1 (shards <run>-<part>.h5, 300 samples) is superseded. Download the v3 release and point NLDBD_EXO200_DATA "
                 "(or NLDBD_DATA_ROOT/EXO-200) at it.")


def load_blacklist(path=BLACKLIST_FILE) -> dict:
    """{run: set(event_number)} from the blacklist CSV (columns run, event_number, ...)."""
    out: dict = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            out.setdefault(int(row["run"]), set()).add(int(row["event_number"]))
    return out


def read_run_table(path, listed) -> dict:
    """Per-event arrays of one run file with the blacklisted events removed.

    Returns event_number, n_clusters, energy_keV (float64) and row (position in the file) of the kept events, and the number removed.
    Raises if a listed event is not in the file, so a blacklist that does not belong to the data cannot pass silently."""
    with h5py.File(path, "r") as f:
        ev = np.asarray(f["Event_Number"][()], dtype=np.int64)
        ncl = np.asarray(f["Charge_Cluster_Number"][()], dtype=np.int64)
        energy = np.asarray(f["Rotated_Energy"][()], dtype=np.float64)
    listed = np.fromiter(listed, dtype=np.int64, count=len(listed))
    bad = np.isin(ev, listed)
    if int(bad.sum()) != len(listed):
        raise RuntimeError(f"{path}: {len(listed)} blacklisted events listed, {int(bad.sum())} found in the file")
    keep = np.flatnonzero(~bad)
    return {"event_number": ev[keep], "n_clusters": ncl[keep], "energy_keV": energy[keep], "row": keep, "n_removed": int(bad.sum())}


class EXO200Reader(Reader):
    name = "exo200"
    signal_group = "signal"
    background_group = "background"

    def __init__(self, root=None):
        super().__init__(root)
        self.manifest = load_json(SPLIT_FILE)
        digest = hashlib.sha256(BLACKLIST_FILE.read_bytes()).hexdigest()
        if digest != self.manifest["blacklist"]["sha256"]:
            raise RuntimeError("manifests/external/exo200_event_blacklist.csv differs from the one the split definition was built with")
        owner_copy = self.root / DATA_BLACKLIST_NAME
        if owner_copy.exists() and hashlib.sha256(owner_copy.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"{owner_copy} differs from the repository's blacklist; the data and the benchmark disagree on which events to drop")
        self.blacklist = load_blacklist()
        self.files = self._find_files()

    def _find_files(self):
        """The 15 run files of release v3; stops with an explanation for release v1 or any other content."""
        v1 = sorted(p.name for p in self.root.glob("*.h5") if re.fullmatch(r"\d{4}-\d{3}\.h5", p.name))
        if v1:
            raise FileNotFoundError(f"{self.root} holds EXO-200 release v1 files ({v1[0]} ...). {NEEDS_RELEASE}")
        files = sorted(p for p in self.root.glob("*.h5") if p.stem.isdigit())
        names = [f.name for f in files]
        expected = self.manifest["expected_files"]
        if names != expected:
            raise FileNotFoundError(f"{self.root}: expected the {len(expected)} files {expected[0]} ... {expected[-1]}, found {names}. {NEEDS_RELEASE}")
        with h5py.File(files[0], "r") as f:
            wf = f.get("Waveforms")
            if not isinstance(wf, h5py.Dataset) or wf.ndim != 3 or wf.shape[1:] != (CHANNELS, SAMPLES):
                found = f"a group of per-event arrays" if isinstance(wf, h5py.Group) else (f"shape {wf.shape}" if wf is not None else "no Waveforms")
                raise FileNotFoundError(f"{files[0]}: Waveforms must be one dataset of shape (N, {CHANNELS}, {SAMPLES}); found {found}. {NEEDS_RELEASE}")
        return files

    def split_files(self, split):
        runs = set(self.manifest["runs"][split])
        return [f for f in self.files if int(f.stem) in runs]

    def table(self, split: str) -> dict:
        ids, labels, energies, file_idx, rows = [], [], [], [], []
        for path in self.split_files(split):
            k = self.files.index(path)
            r = read_run_table(path, self.blacklist.get(int(path.stem), set()))
            n = len(r["row"])
            labels.append((r["n_clusters"] == 1).astype(np.int8))
            energies.append(r["energy_keV"])
            ids.append(np.array([f"EXO200::{path.name}::{e}" for e in r["event_number"]], dtype=str))
            file_idx.append(np.full(n, k)); rows.append(r["row"])
        t = {"event_id": np.concatenate(ids), "label": np.concatenate(labels), "energy_keV": np.concatenate(energies),
             "file_index": np.concatenate(file_idx), "row": np.concatenate(rows)}
        t["group"] = np.where(t["label"] == 1, self.signal_group, self.background_group)
        t["file"] = np.array([f.name for f in self.files])[t["file_index"]]
        expected = self.manifest["counts"][split]
        if len(t["event_id"]) != expected:
            raise RuntimeError(f"EXO-200 {split}: {len(t['event_id'])} events, manifest says {expected}")
        return self.check_table(t)

    @staticmethod
    def preprocess(raw: np.ndarray) -> np.ndarray:
        x = np.asarray(raw, dtype=np.float32)
        x = x - x[:, :, :BASELINE_SAMPLES].mean(axis=2, keepdims=True)
        scale = np.abs(x).reshape(len(x), -1).max(axis=1)[:, None, None]
        x = np.divide(x, scale, out=x, where=scale > 0)
        return x

    def read_raw(self, file_index: int, rows) -> np.ndarray:
        """Raw int16 waveforms (len(rows), 226, 600) of strictly increasing ``rows`` of file ``file_index``."""
        rows = np.asarray(rows)
        with h5py.File(self.files[file_index], "r") as f:
            wf = f["Waveforms"]
            if len(rows) and rows[-1] - rows[0] + 1 == len(rows):
                return wf[int(rows[0]):int(rows[-1]) + 1]
            return wf[rows]

    def iter_inputs(self, split: str, batch_size: int = 64, indices=None, table=None):
        t = self.table(split) if table is None else table
        pos = np.arange(len(t["event_id"])) if indices is None else np.asarray(indices)
        order = np.lexsort((t["row"][pos], t["file_index"][pos]))
        pos = pos[order]
        for k in np.unique(t["file_index"][pos]):
            here = pos[t["file_index"][pos] == k]
            for start in range(0, len(here), batch_size):
                chunk = here[start:start + batch_size]
                yield chunk, self.preprocess(self.read_raw(int(k), t["row"][chunk]))
