"""MAJORANA DEMONSTRATOR (MJD) waveforms.

Paper protocol (mjdbench, Transformer_Approach/mjd_detector):
* files: 16 ``MJD_Train_*.hdf5`` (65,000 rows each) and 6 ``MJD_Test_*.hdf5``. The 3 ``MJD_NPML_*.hdf5``
  shards are EXCLUDED from this benchmark entirely (user decision 2026-10-22): never read them, not
  even for pretraining;
* label: event is *clean* (signal-like, evaluation-positive, label 1) when all four PSD flags
  ``psd_label_low_avse, psd_label_high_avse, psd_label_dcr, psd_label_lq`` are True;
* energy: ``energy_label`` in keV read as float64 (never the float32 cast);
* split: Train files sorted lexicographically (Train_0, Train_1, Train_10, ..., Train_9) and
  concatenated; ``np.random.default_rng(42).permutation(N)``; the first round(0.10 N) = 104,000
  indices are validation, the remaining 936,000 train; test = all Test files in order (390,000);
* event_id: ``"MJD_Test_0.hdf5::row=0::id=2395098"`` (file name, row within file, stored id);
* input: ``raw_waveform`` (3800 samples), minus the mean of the first 200 samples, divided by the
  maximum absolute amplitude (skipped when it is 0) -> float32 ``[B, 1, 3800]``.
"""
from __future__ import annotations

import numpy as np
import h5py

from .common import Reader

PSD_NAMES = ("psd_label_low_avse", "psd_label_high_avse", "psd_label_dcr", "psd_label_lq")
SEED = 42
VALIDATION_FRACTION = 0.10
BASELINE_SAMPLES = 200
WAVEFORM_LENGTH = 3800


class MJDReader(Reader):
    name = "mjd"
    signal_group = "clean"
    background_group = "nonclean"

    def train_files(self):
        return sorted(self.root.glob("MJD_Train_*.hdf5"))

    def test_files(self):
        return sorted(self.root.glob("MJD_Test_*.hdf5"))

    # ------------------------------------------------------------------------------------
    def _read_meta(self, files):
        ids, labels, energies, file_idx, rows = [], [], [], [], []
        for k, path in enumerate(files):
            with h5py.File(path, "r") as f:
                n = f["energy_label"].shape[0]
                flags = np.stack([np.asarray(f[name][()], dtype=bool) for name in PSD_NAMES], axis=1)
                energies.append(np.asarray(f["energy_label"][()], dtype=np.float64))
                stored = np.asarray(f["id"][()], dtype=np.int64)
            labels.append(np.all(flags, axis=1).astype(np.int8))
            rows.append(np.arange(n, dtype=np.int64))
            file_idx.append(np.full(n, k, dtype=np.int64))
            ids.append(np.array([f"{path.name}::row={r}::id={i}" for r, i in zip(range(n), stored)], dtype=str))
        return {
            "event_id": np.concatenate(ids),
            "label": np.concatenate(labels),
            "energy_keV": np.concatenate(energies),
            "file_index": np.concatenate(file_idx),
            "row": np.concatenate(rows),
        }

    def _trainval_order(self):
        files = self.train_files()
        meta = self._read_meta(files)
        n = len(meta["event_id"])
        perm = np.random.default_rng(SEED).permutation(n)
        n_val = int(round(VALIDATION_FRACTION * n))
        return files, meta, perm[:n_val], perm[n_val:]

    def table(self, split: str) -> dict:
        if split == "test":
            files = self.test_files()
            meta = self._read_meta(files)
            sel = np.arange(len(meta["event_id"]))
        elif split in ("train", "validation"):
            files, meta, val_idx, train_idx = self._trainval_order()
            sel = val_idx if split == "validation" else train_idx
        else:
            raise ValueError(split)
        t = {k: v[sel] for k, v in meta.items()}
        t["group"] = np.where(t["label"] == 1, self.signal_group, self.background_group)
        t["file"] = np.array([f.name for f in files], dtype=str)[t["file_index"]]
        return self.check_table(t)

    # ------------------------------------------------------------------------------------
    @staticmethod
    def preprocess(raw: np.ndarray) -> np.ndarray:
        x = np.asarray(raw, dtype=np.float32)
        x = x - x[:, :BASELINE_SAMPLES].mean(axis=1, keepdims=True)
        scale = np.abs(x).max(axis=1, keepdims=True)
        x = np.divide(x, scale, out=x, where=scale > 0)
        return x[:, None, :]

    def iter_inputs(self, split: str, batch_size: int = 256, indices=None, table=None):
        t = self.table(split) if table is None else table
        pos = np.arange(len(t["event_id"])) if indices is None else np.asarray(indices)
        files = self.test_files() if split == "test" else self.train_files()
        # read each file once, in row order, then emit in the requested position order
        order = np.lexsort((t["row"][pos], t["file_index"][pos]))
        pos = pos[order]
        for k in np.unique(t["file_index"][pos]):
            here = pos[t["file_index"][pos] == k]
            rows = t["row"][here]
            with h5py.File(files[k], "r") as f:
                ds = f["raw_waveform"]
                for start in range(0, len(here), batch_size):
                    r = rows[start:start + batch_size]
                    raw = ds[np.sort(r)]  # h5py needs increasing indices
                    inv = np.argsort(np.argsort(r))
                    yield here[start:start + batch_size], self.preprocess(raw[inv])
