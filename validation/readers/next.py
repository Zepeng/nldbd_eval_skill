"""NEXT simulated 0nubb / Bi214 hit clouds.

Paper protocol (evalutaions_workflow/energybench/data.py; manifest
``manifests/external/next_event_split.json``):
* 14,973 files ``0nubb_part_*/*.h5`` and ``Bi_part_*/*.h5``; table ``MC/hits/table`` with one row per
  hit: ``values_block_0`` = event id, ``values_block_1`` = (x, y, z, hit energy [MeV]),
  ``values_block_2`` = 'Signal'/'Bkg'; rows of one event are contiguous;
* label: 0nubb = 1 (signal-like, evaluation-positive), Bi214 = 0;
* energy: float64 sum of the event's hit energies (MeV) converted to keV with the evaluator's exact
  canonical-edge rule (``metrics.to_keV``);
* split: event-count 80/10/10, seed 42; files shuffled per class, only boundary files sliced
  (half-open event ordinals ``event_start:event_stop``); test = 116,549 events (64,879 / 51,670);
  canonical order = manifest slice order;
* event_id: ``"NEXT::<relative_path>::<source event id>"``;
* inputs: hits ``(x, y, z, e)`` float32 per event (variable length). Tokenizers divide hit energies
  by the event total and center coordinates; absolute energy must not be a model input.

Combined file: if ``NLDBD_NEXT_COMBINED`` (default ``/home/klz/Data/NEXT_combined/next_combined.h5``)
exists and its inventory digest matches the manifest, tables and inputs are read from it (one
sequential file instead of 14,973 small ones; built by ``validation/build_next_combined.py``).
Results are identical either way; the frozen manifest hashes are the proof.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import h5py

from .common import MANIFESTS, Reader, load_json
from ..energybench.metrics import to_keV

TABLE = "MC/hits/table"
CATEGORY_LABEL = {"0nubb": 1, "Bi214": 0}
ROW_LABEL = {"0nubb": b"Signal", "Bi214": b"Bkg"}


class NEXTReader(Reader):
    name = "next"
    signal_group = "0nubb"
    background_group = "Bi214"

    DEFAULT_COMBINED = "/home/klz/Data/NEXT_combined/next_combined.h5"

    def __init__(self, root=None):
        super().__init__(root)
        self.manifest = load_json(MANIFESTS / "external" / "next_event_split.json")
        self.combined = None
        candidate = Path(os.environ.get("NLDBD_NEXT_COMBINED", self.DEFAULT_COMBINED))
        if candidate.exists():
            with h5py.File(candidate, "r") as f:
                if f.attrs.get("inventory_digest") == self.manifest["inventory"]["digest"]:
                    self.combined = candidate
                    self._ranges = json.loads(f.attrs["split_ranges"])

    def _table_combined(self, split):
        a, b = self._ranges[split]
        with h5py.File(self.combined, "r") as f:
            t = {"event_id": f["event_id"][a:b].astype(str), "label": f["label"][a:b].astype(np.int8),
                 "energy_MeV": f["energy_MeV"][a:b].astype(np.float64), "file": f["file"][a:b].astype(str),
                 "ordinal": f["ordinal"][a:b].astype(np.int64), "hit_offset": f["hit_offset"][a:b + 1]}
        t["group"] = np.where(t["label"] == 1, self.signal_group, self.background_group)
        t["energy_keV"] = to_keV(t["energy_MeV"], "MeV")
        t["combined_start"] = a
        return self.check_table(t)

    def _iter_inputs_combined(self, split, batch_size, indices, table):
        t = table
        pos = np.sort(np.arange(len(t["event_id"])) if indices is None else np.asarray(indices))
        off = t["hit_offset"]
        with h5py.File(self.combined, "r") as f:
            hits = f["hits"]
            for s in range(0, len(pos), batch_size):
                chunk = pos[s:s + batch_size]
                lo, hi = off[chunk[0]], off[chunk[-1] + 1]
                block = hits[lo:hi]
                yield chunk, [block[off[p] - lo: off[p + 1] - lo].copy() for p in chunk]

    @staticmethod
    def _events(table):
        """Return (event ids, start row, stop row, hit xyz+e) for contiguous event runs."""
        ev = table["values_block_0"][:, 0]
        starts = np.r_[0, np.flatnonzero(ev[1:] != ev[:-1]) + 1]
        stops = np.r_[starts[1:], len(ev)]
        ids = ev[starts]
        if len(np.unique(ids)) != len(ids):
            raise ValueError("non-contiguous event ids")
        return ids, starts, stops

    def table(self, split: str) -> dict:
        if self.combined is not None:
            return self._table_combined(split)
        ids, labels, energies, groups, files, ordinal = [], [], [], [], [], []
        for slc in self.manifest["splits"][split]:
            rel = slc["relative_path"]
            with h5py.File(self.root / rel, "r") as f:
                t = f[TABLE][()]
            ev, starts, stops = self._events(t)
            expected = ROW_LABEL[slc["category"]]
            if not np.all(t["values_block_2"][:, 0] == expected):
                raise ValueError(f"{rel}: row labels are not all {expected!r}")
            e_hit = np.asarray(t["values_block_1"][:, 3], dtype=np.float64)
            sums = np.add.reduceat(e_hit, starts)
            sel = np.arange(slc["event_start"], slc["event_stop"])
            if slc["event_stop"] > len(ev):
                raise ValueError(f"{rel}: manifest slice exceeds {len(ev)} events")
            ids.append(np.array([f"NEXT::{rel}::{int(i)}" for i in ev[sel]], dtype=str))
            energies.append(sums[sel])
            labels.append(np.full(len(sel), CATEGORY_LABEL[slc["category"]], dtype=np.int8))
            groups.append(np.full(len(sel), slc["category"], dtype=object))
            files.append(np.full(len(sel), rel, dtype=object)); ordinal.append(sel)
        t = {"event_id": np.concatenate(ids), "label": np.concatenate(labels),
             "energy_MeV": np.concatenate(energies), "group": np.concatenate(groups).astype(str),
             "file": np.concatenate(files).astype(str), "ordinal": np.concatenate(ordinal)}
        t["energy_keV"] = to_keV(t["energy_MeV"], "MeV")
        expected = self.manifest["counts"][split]
        if len(t["event_id"]) != expected:
            raise RuntimeError(f"NEXT {split}: {len(t['event_id'])} events, manifest says {expected}")
        return self.check_table(t)

    def iter_inputs(self, split: str, batch_size: int = 64, indices=None, table=None):
        """Yield (positions, list of float32 (n_hits, 4) arrays [x, y, z, hit energy MeV])."""
        t = self.table(split) if table is None else table
        if self.combined is not None and "hit_offset" in t:
            yield from self._iter_inputs_combined(split, batch_size, indices, t)
            return
        pos = np.arange(len(t["event_id"])) if indices is None else np.asarray(indices)
        order = np.lexsort((t["ordinal"][pos], t["file"][pos]))
        pos = pos[order]
        batch_pos, batch = [], []
        for rel in np.unique(t["file"][pos]):
            here = pos[t["file"][pos] == rel]
            with h5py.File(self.root / rel, "r") as f:
                tab = f[TABLE][()]
            ev, starts, stops = self._events(tab)
            xyz_e = np.asarray(tab["values_block_1"], dtype=np.float32)
            for p, o in zip(here, t["ordinal"][here]):
                batch_pos.append(p); batch.append(xyz_e[starts[o]:stops[o]].copy())  # copy: a view would pin the whole file in memory
                if len(batch) == batch_size:
                    yield np.array(batch_pos), batch
                    batch_pos, batch = [], []
        if batch:
            yield np.array(batch_pos), batch
