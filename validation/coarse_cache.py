"""One-time local cache of the coarse-subset inputs, so coarse evaluations take seconds.

The cache stores exactly what ``Reader.iter_inputs`` yields for the frozen coarse positions, in
one HDF5 file per dataset under ``NLDBD_CACHE_DIR`` (default ``<project>/cache``). It is keyed by
the manifest's coarse identity hash, so a regenerated manifest invalidates it automatically. It is
a performance device only: never ship it, never read labels or energies from it (it has none).

Formats: MJD float32 [n, 1, 3800] (preprocessed); EXO-200 int16 [n, 226, 300] raw waveforms
(preprocessing is applied on load, exactly as the reader does); NEXT and SuperNEMO ragged hits as
float32 [total_hits, 4] with int64 offsets.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import h5py
import numpy as np

from .readers.common import MANIFESTS
from .freeze_splits import coarse_positions

ROOT = Path(__file__).resolve().parents[1]


def cache_dir() -> Path:
    return Path(os.environ.get("NLDBD_CACHE_DIR", ROOT / "cache"))


def cache_path(dataset: str) -> Path:
    return cache_dir() / f"{dataset}_coarse_inputs.h5"


def _coarse_spec(dataset: str):
    manifest = json.loads((MANIFESTS / f"{dataset}.json").read_text())
    return manifest["coarse"]


def is_valid(dataset: str) -> bool:
    path = cache_path(dataset)
    if not path.exists():
        return False
    with h5py.File(path, "r") as f:
        return f.attrs.get("ids_sha256_sorted") == _coarse_spec(dataset)["ids_sha256_sorted"]


def build(dataset: str, reader, table, positions, batch_size=256) -> Path:
    """Read the coarse inputs once through the reader and store them."""
    path = cache_path(dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".building.h5")
    spec = _coarse_spec(dataset)
    t0 = time.time()
    ids = table["event_id"][positions]
    with h5py.File(tmp, "w") as f:
        f.attrs["dataset"] = dataset
        f.attrs["ids_sha256_sorted"] = spec["ids_sha256_sorted"]
        f.create_dataset("event_id", data=ids.astype("S"))
        where = {p: i for i, p in enumerate(positions)}
        if dataset == "mjd":
            ds = f.create_dataset("inputs", shape=(len(positions), 1, 3800), dtype=np.float32, chunks=(64, 1, 3800))
            for pos, x in reader.iter_inputs("validation", batch_size=batch_size, indices=positions, table=table):
                idx = np.array([where[p] for p in pos]); order = np.argsort(idx)
                ds[idx[order]] = x[order]
        elif dataset == "exo200":
            ds = f.create_dataset("inputs_int16", shape=(len(positions), 226, 300), dtype=np.int16, chunks=(16, 226, 300))
            # raw int16, read through the reader's own file/row locators to stay exact
            files = reader.files
            order = np.lexsort((table["row"][positions], table["file_index"][positions]))
            pos_sorted = positions[order]
            for k in np.unique(table["file_index"][pos_sorted]):
                here = pos_sorted[table["file_index"][pos_sorted] == k]
                with h5py.File(files[k], "r") as g:
                    wf = g["Waveforms"]
                    for start in range(0, len(here), batch_size):
                        chunk = here[start:start + batch_size]
                        raw = np.stack([wf[str(r)][()] for r in table["row"][chunk]]).astype(np.int16)
                        idx = np.array([where[p] for p in chunk]); o = np.argsort(idx)
                        ds[idx[o]] = raw[o]
        else:
            hits, offsets, sizes = [], [0], np.zeros(len(positions), dtype=np.int64)
            per_event = [None] * len(positions)
            for pos, batch in reader.iter_inputs("validation", batch_size=batch_size, indices=positions, table=table):
                for p, h in zip(pos, batch):
                    per_event[where[p]] = np.asarray(h, dtype=np.float32)
            for i, h in enumerate(per_event):
                hits.append(h); sizes[i] = len(h)
            f.create_dataset("hits", data=np.concatenate(hits), dtype=np.float32)
            f.create_dataset("offsets", data=np.r_[0, np.cumsum(sizes)], dtype=np.int64)
        f.attrs["seconds_to_build"] = round(time.time() - t0, 1)
    tmp.replace(path)
    return path


def iter_cached(dataset: str, batch_size=256):
    """Yield (positions into the coarse table order, inputs) from the cache."""
    from .readers.exo200 import EXO200Reader
    with h5py.File(cache_path(dataset), "r") as f:
        n = len(f["event_id"])
        if dataset == "mjd":
            ds = f["inputs"]
            for s in range(0, n, batch_size):
                yield np.arange(s, min(s + batch_size, n)), ds[s:s + batch_size]
        elif dataset == "exo200":
            ds = f["inputs_int16"]
            for s in range(0, n, batch_size):
                yield np.arange(s, min(s + batch_size, n)), EXO200Reader.preprocess(ds[s:s + batch_size])
        else:
            off = f["offsets"][()]
            hits = f["hits"]
            for s in range(0, n, batch_size):
                e = min(s + batch_size, n)
                block = hits[off[s]:off[e]]
                yield np.arange(s, e), [block[off[i] - off[s]: off[i + 1] - off[s]] for i in range(s, e)]


def cached_event_ids(dataset: str):
    with h5py.File(cache_path(dataset), "r") as f:
        return f["event_id"][()].astype(str)
