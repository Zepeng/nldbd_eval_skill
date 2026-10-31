"""EXO-200 open data (v1) sensor waveforms.

Paper protocol (exobench, Transformer_Approach/exo200_detector; manifest
``manifests/external/exo200_v1_split.json``):
* 29 ``<run>-<part>.h5`` shards, 515,002 events; ``Waveforms/<i>`` is a (226, 300) int16 array;
* label: ``Charge_cluster_number == 1`` is signal-like (evaluation-positive, label 1); the stored
  exobench label is the opposite orientation (background = 1) and is flipped here;
* energy: ``Rotated_energy`` (keV, float32 in the file) read as float64;
* split (run-grouped, seed 42): train runs 8956 8957 8958 8968 8972 8999 9000 9001 9008 9009,
  validation 8969, test 8967 8970 8971 9010 (140,383 events); within a split, files in sorted
  order and rows ascending;
* event_id: ``"EXO200::8967-000.h5::4"`` (file name, stored ``event_number``);
* forbidden inputs: ``Charge_Clusters_Pos`` (its shape encodes the label), ``Charge_cluster_number``,
  and every energy field;
* input: waveforms as float32, per-channel mean of the first 200 samples subtracted, the whole event
  divided by its maximum absolute amplitude (skipped when 0) -> ``[B, 226, 300]``.
"""
from __future__ import annotations

import numpy as np
import h5py

from .common import MANIFESTS, Reader, load_json

BASELINE_SAMPLES = 200


class EXO200Reader(Reader):
    name = "exo200"
    signal_group = "signal"
    background_group = "background"

    def __init__(self, root=None):
        super().__init__(root)
        self.manifest = load_json(MANIFESTS / "external" / "exo200_v1_split.json")
        files = sorted(self.root.glob("*.h5"))
        names = [f.name for f in files]
        if names != self.manifest["expected_files"]:
            raise FileNotFoundError(f"EXO-200 files differ from the manifest: {names}")
        self.files = files

    def split_files(self, split):
        runs = set(self.manifest["runs"][split])
        return [f for f in self.files if int(f.name.split("-")[0]) in runs]

    def table(self, split: str) -> dict:
        ids, labels, energies, file_idx, rows = [], [], [], [], []
        for path in self.split_files(split):
            k = self.files.index(path)
            with h5py.File(path, "r") as f:
                nccl = np.asarray(f["Charge_cluster_number"][()], dtype=np.int64)
                ev = np.asarray(f["event_number"][()], dtype=np.int64)
                energies.append(np.asarray(f["Rotated_energy"][()], dtype=np.float64))
            n = len(nccl)
            labels.append((nccl == 1).astype(np.int8))
            ids.append(np.array([f"EXO200::{path.name}::{e}" for e in ev], dtype=str))
            file_idx.append(np.full(n, k)); rows.append(np.arange(n))
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

    def iter_inputs(self, split: str, batch_size: int = 64, indices=None, table=None):
        t = self.table(split) if table is None else table
        pos = np.arange(len(t["event_id"])) if indices is None else np.asarray(indices)
        order = np.lexsort((t["row"][pos], t["file_index"][pos]))
        pos = pos[order]
        for k in np.unique(t["file_index"][pos]):
            here = pos[t["file_index"][pos] == k]
            rows = t["row"][here]
            with h5py.File(self.files[k], "r") as f:
                wf = f["Waveforms"]
                for start in range(0, len(here), batch_size):
                    raw = np.stack([wf[str(r)][()] for r in rows[start:start + batch_size]])
                    yield here[start:start + batch_size], self.preprocess(raw)
