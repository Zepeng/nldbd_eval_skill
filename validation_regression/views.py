"""Energy-bearing views of the official readers (regression task).

The classification view is unchanged: each event yields exactly the inputs the classification readers yield
(amplitude-normalised waveforms / raw tracker hits). The regression view adds per event
* ``log_scale``: log of the normalisation scale that preprocessing divided by (the per-event max |x| after baseline
  subtraction; 0 for SuperNEMO, whose hit inputs are not normalised), so that view + scale is lossless;
* ``detector``: the MJD detector index (position in benchmark.json["datasets"]["mjd"]["detectors"]; 0 elsewhere).
The regression target is the reference reconstructed energy ``energy_keV`` of the table (MJD ``energy_label``,
EXO-200 ``Rotated_energy``, SuperNEMO ``E1 + E2``); no energy field is ever an input.

SuperNEMO regression uses its own table (manifest blocks, same 80/10/10 split as classification): train and validation
contain 0nubb only (user 2026-10-25), test contains 0nubb (the headline peak), 2nubb and Bi214 (diagnostics). 0nubb never
enters the classification task or its pretraining, so its blocks are a clean regression sample with a peak.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import h5py

from validation.readers.mjd import MJDReader
from validation.readers.exo200 import EXO200Reader
from validation.readers.supernemo import SuperNEMOReader

HERE = Path(__file__).resolve().parent
BENCHMARK = json.loads((HERE / "benchmark.json").read_text())
DATASETS = tuple(BENCHMARK["datasets"])


def _log_scale(scale):
    s = np.asarray(scale, dtype=np.float64).reshape(-1)
    return np.log(np.where(s > 0, s, 1.0)).astype(np.float32)


class MJDRegression(MJDReader):
    def __init__(self, root=None):
        super().__init__(root)
        self.detectors = np.asarray(BENCHMARK["datasets"]["mjd"]["detectors"], dtype=np.int64)

    def _read_meta(self, files):
        meta = super()._read_meta(files)
        det = []
        for path in files:
            with h5py.File(path, "r") as f:
                det.append(np.asarray(f["detector"][()], dtype=np.int64))
        det = np.concatenate(det)
        idx = np.searchsorted(self.detectors, det)
        ok = (idx < len(self.detectors)) & (self.detectors[np.minimum(idx, len(self.detectors) - 1)] == det)
        if not ok.all():
            raise ValueError(f"MJD: detector ids not in benchmark.json: {np.unique(det[~ok])}")
        meta["detector"] = idx.astype(np.int64)
        return meta

    def preprocess(self, raw):
        x = np.asarray(raw, dtype=np.float32)
        x = x - x[:, :200].mean(axis=1, keepdims=True)
        self._scale = np.abs(x).max(axis=1)
        return MJDReader.preprocess(raw)

    def iter_regression(self, split, batch_size=256, indices=None, table=None):
        t = self.table(split) if table is None else table
        for pos, x in self.iter_inputs(split, batch_size=batch_size, indices=indices, table=t):
            yield pos, x, {"log_scale": _log_scale(self._scale), "detector": t["detector"][pos]}


class EXO200Regression(EXO200Reader):
    def preprocess(self, raw):
        x = np.asarray(raw, dtype=np.float32)
        x = x - x[:, :, :200].mean(axis=2, keepdims=True)
        self._scale = np.abs(x).reshape(len(x), -1).max(axis=1)
        return EXO200Reader.preprocess(raw)

    def iter_regression(self, split, batch_size=64, indices=None, table=None):
        t = self.table(split) if table is None else table
        for pos, x in self.iter_inputs(split, batch_size=batch_size, indices=indices, table=t):
            yield pos, x, {"log_scale": _log_scale(self._scale), "detector": np.zeros(len(pos), dtype=np.int64)}


REGRESSION_SOURCES = {"0nubb": ("0nu", 1), "2nubb": ("2nu", 1), "Bi214": ("Bi214", 0)}
# Training and selection use 0nubb only (user 2026-10-25): the test metric is the 0nubb peak, and training on the
# 2nubb/Bi214 continua taught regressors the mixed-source energy distribution (residuals grew with more labels).
# The test table keeps all three sources for the secondary diagnostics.
SPLIT_SOURCES = {"train": ("0nubb",), "validation": ("0nubb",), "test": ("0nubb", "2nubb", "Bi214")}


class SuperNEMORegression(SuperNEMOReader):
    """Regression table over sources 0nubb, 2nubb, Bi214 (label = classification orientation, 0nubb = 1; unused)."""

    def slices(self, split):
        return [s for s in self.manifest["splits"][split] if s["source_key"] in SPLIT_SOURCES[split]]

    def table(self, split):
        """Same construction as SuperNEMOReader.table, over the regression sources (count checked against the
        manifest's per-category energy counts)."""
        ids, labels, energies, groups, src, evno = [], [], [], [], [], []
        for slc in self.slices(split):
            key = slc["source_key"]; off = self.offsets(key)
            a, b = slc["event_start"], slc["event_stop"]
            with h5py.File(self.root / self.file_of[key], "r") as f:
                r0, r1 = off[a], off[b]
                first = off[a:b] - r0
                e = np.asarray(f["E1"][r0:r1], dtype=np.float64)[first] + np.asarray(f["E2"][r0:r1], dtype=np.float64)[first]
            group, label = REGRESSION_SOURCES[key]; n = b - a
            ids.append(np.array([f"SuperNEMO::{key}::{i}" for i in range(a, b)], dtype=str)); energies.append(e)
            labels.append(np.full(n, label, dtype=np.int8)); groups.append(np.full(n, group, dtype=object))
            src.append(np.full(n, key, dtype=object)); evno.append(np.arange(a, b))
        t = {"event_id": np.concatenate(ids), "label": np.concatenate(labels), "energy_keV": np.concatenate(energies),
             "group": np.concatenate(groups).astype(str), "source_key": np.concatenate(src).astype(str),
             "ev_no": np.concatenate(evno)}
        by = self.manifest["counts"]["energy"][split]["by_category"]
        category = {f["source_key"]: f["category"] for f in self.manifest["inventory"]}
        expected = sum(by[category[k]] for k in SPLIT_SOURCES[split])
        if len(t["event_id"]) != expected:
            raise RuntimeError(f"SuperNEMO regression {split}: {len(t['event_id'])} events, manifest says {expected}")
        return self.check_table(t)

    def iter_regression(self, split, batch_size=64, indices=None, table=None):
        t = self.table(split) if table is None else table
        for pos, x in self.iter_inputs(split, batch_size=batch_size, indices=indices, table=t):
            yield pos, x, {"log_scale": np.zeros(len(pos), dtype=np.float32), "detector": np.zeros(len(pos), dtype=np.int64)}


READERS = {"mjd": MJDRegression, "exo200": EXO200Regression, "supernemo": SuperNEMORegression}


def get_view(dataset, root=None):
    if dataset not in READERS:
        raise ValueError(f"no regression task for {dataset!r}; registered: {tuple(READERS)}")
    return READERS[dataset](root)
