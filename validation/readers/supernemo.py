"""SuperNEMO simulated tracker events.

Paper protocol (supernemobench; manifest ``manifests/external/supernemo_split_manifest.json``):
* four flat HDF5 files ``data_{0nubb,2nubb,Bi214,Tl208}_merged.h5``, one row per tracker hit:
  ``ev_no`` (contiguous per file), ``E1``/``E2`` calorimeter energies in keV (repeated on every row
  of an event), ``tX tY tZ`` hit coordinates in mm, ``tR`` drift radius (may be NaN), plus
  simulation-truth fields ``dY dZ theta phiS phiR`` and ``label``;
* classification task: 2nubb = 1 (signal-like, evaluation-positive, group ``2nu``), Bi214 = 0;
  0nubb and Tl208 are excluded from this task;
* energy: ``E1 + E2`` (keV, float64 of the first row of the event);
* split: each source cut into 4096-event blocks, blocks permuted with ``SeedSequence([42, source
  position])``, 80/10/10 by largest remainder; test = 327,680 2nubb + 262,144 Bi214 = 589,824;
  canonical order = manifest slice order restricted to the two task sources;
* event_id: ``"SuperNEMO::<source_key>::<ev_no>"``;
* inputs: tracker hits ``(tX, tY, tZ, tR)`` float32 per event (variable length). ``E1``, ``E2``,
  ``dY dZ theta phiS phiR``, labels and source names must never be model inputs.
"""
from __future__ import annotations

import numpy as np
import h5py

from .common import MANIFESTS, Reader, load_json

SOURCES = {"2nubb": ("2nu", 1), "Bi214": ("Bi214", 0)}
HIT_FIELDS = ("tX", "tY", "tZ", "tR")


class SuperNEMOReader(Reader):
    name = "supernemo"
    signal_group = "2nu"
    background_group = "Bi214"

    def __init__(self, root=None):
        super().__init__(root)
        self.manifest = load_json(MANIFESTS / "external" / "supernemo_split_manifest.json")
        self.file_of = {f["source_key"]: f["file_name"] for f in self.manifest["inventory"]}
        self._offsets = {}

    def offsets(self, source_key: str) -> np.ndarray:
        """Row offset of every event (length n_events + 1), cached under manifests/."""
        if source_key in self._offsets:
            return self._offsets[source_key]
        cache = MANIFESTS / f"supernemo_{source_key}_hit_counts.npz"
        if cache.exists():
            counts = np.load(cache)["hit_counts"].astype(np.int64)
        else:
            with h5py.File(self.root / self.file_of[source_key], "r") as f:
                ev = f["ev_no"][()]
            starts = np.r_[0, np.flatnonzero(ev[1:] != ev[:-1]) + 1]
            if not np.array_equal(ev[starts], np.arange(len(starts))):
                raise ValueError(f"{source_key}: ev_no is not contiguous from 0")
            counts = np.diff(np.r_[starts, len(ev)])
            np.savez_compressed(cache, hit_counts=counts.astype(np.uint16) if counts.max() < 65536 else counts)
        off = np.r_[0, np.cumsum(counts)]
        self._offsets[source_key] = off
        return off

    def slices(self, split: str):
        return [s for s in self.manifest["splits"][split] if s["source_key"] in SOURCES]

    def table(self, split: str) -> dict:
        ids, labels, energies, groups, src, evno = [], [], [], [], [], []
        handles = {}
        try:
            for slc in self.slices(split):
                key = slc["source_key"]
                if key not in handles:
                    handles[key] = h5py.File(self.root / self.file_of[key], "r")
                f = handles[key]
                off = self.offsets(key)
                a, b = slc["event_start"], slc["event_stop"]
                first_rows = off[a:b]
                r0, r1 = off[a], off[b]
                e1 = np.asarray(f["E1"][r0:r1], dtype=np.float64)[first_rows - r0]
                e2 = np.asarray(f["E2"][r0:r1], dtype=np.float64)[first_rows - r0]
                n = b - a
                ev = np.arange(a, b)
                ids.append(np.array([f"SuperNEMO::{key}::{i}" for i in ev], dtype=str))
                energies.append(e1 + e2)
                group, label = SOURCES[key]
                labels.append(np.full(n, label, dtype=np.int8)); groups.append(np.full(n, group, dtype=object))
                src.append(np.full(n, key, dtype=object)); evno.append(ev)
        finally:
            for h in handles.values():
                h.close()
        t = {"event_id": np.concatenate(ids), "label": np.concatenate(labels), "energy_keV": np.concatenate(energies),
             "group": np.concatenate(groups).astype(str), "source_key": np.concatenate(src).astype(str),
             "ev_no": np.concatenate(evno)}
        expected = self.manifest["counts"]["classification"][split]["total"]
        if len(t["event_id"]) != expected:
            raise RuntimeError(f"SuperNEMO {split}: {len(t['event_id'])} events, manifest says {expected}")
        return self.check_table(t)

    def iter_inputs(self, split: str, batch_size: int = 64, indices=None, table=None):
        """Yield (positions, list of float32 (n_hits, 4) arrays [tX, tY, tZ, tR]); tR may be NaN."""
        t = self.table(split) if table is None else table
        pos = np.arange(len(t["event_id"])) if indices is None else np.asarray(indices)
        order = np.lexsort((t["ev_no"][pos], t["source_key"][pos]))
        pos = pos[order]
        for key in np.unique(t["source_key"][pos]):
            here = pos[t["source_key"][pos] == key]
            off = self.offsets(key)
            with h5py.File(self.root / self.file_of[key], "r") as f:
                for start in range(0, len(here), batch_size):
                    chunk = here[start:start + batch_size]
                    ev = t["ev_no"][chunk]
                    # read one contiguous row span covering the chunk (events are nearly contiguous)
                    r0, r1 = off[ev.min()], off[ev.max() + 1]
                    cols = np.stack([np.asarray(f[name][r0:r1], dtype=np.float32) for name in HIT_FIELDS], axis=1)
                    batch = [cols[off[e] - r0: off[e + 1] - r0].copy() for e in ev]
                    yield chunk, batch
