"""Generic one-time local cache of reader inputs for a fixed set of events (NLDBD_CACHE_DIR).

Used for the labeled-regime inputs of the fine-tuning protocol (one file per dataset x label
fraction) so every fine-tune of every candidate reads them from fast local/scratch storage instead
of the raw data. Keyed by the SHA-256 of the sorted event ids; the harness never ships a cache.
Formats as in coarse_cache: mjd float32 [n,1,3800]; exo200 int16 [n,226,300] raw (preprocessed on
load); next/supernemo ragged hits float32 [total,4] + offsets.
"""
from __future__ import annotations

import time
from pathlib import Path

import h5py
import numpy as np

from .coarse_cache import cache_dir
from .readers.common import sha256_lines


def path_for(dataset: str, name: str) -> Path:
    return cache_dir() / f"{dataset}_{name}_inputs.h5"


def is_valid(dataset: str, name: str, event_ids) -> bool:
    p = path_for(dataset, name)
    if not p.exists():
        return False
    with h5py.File(p, "r") as f:
        return f.attrs.get("ids_sha256_sorted") == sha256_lines(np.sort(np.asarray(event_ids).astype(str)))


def build(dataset: str, name: str, reader, split: str, table, positions, batch_size=256) -> Path:
    """Read the inputs of `positions` (into `table`) once through the reader and store them in position order."""
    p = path_for(dataset, name); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".building.h5"); t0 = time.time()
    positions = np.asarray(positions); ids = table["event_id"][positions]
    where = {int(q): i for i, q in enumerate(positions)}
    with h5py.File(tmp, "w") as f:
        f.attrs["dataset"] = dataset; f.attrs["name"] = name; f.attrs["split"] = split
        f.attrs["ids_sha256_sorted"] = sha256_lines(np.sort(ids))
        f.create_dataset("event_id", data=ids.astype("S"))
        if dataset == "mjd":
            ds = f.create_dataset("inputs", shape=(len(positions), 1, 3800), dtype=np.float32, chunks=(64, 1, 3800))
            for pos, x in reader.iter_inputs(split, batch_size=batch_size, indices=positions, table=table):
                idx = np.array([where[int(q)] for q in pos]); o = np.argsort(idx); ds[idx[o]] = x[o]
        elif dataset == "exo200":
            ds = f.create_dataset("inputs_int16", shape=(len(positions), 226, 300), dtype=np.int16, chunks=(16, 226, 300))
            order = np.lexsort((table["row"][positions], table["file_index"][positions])); ps = positions[order]
            for k in np.unique(table["file_index"][ps]):
                here = ps[table["file_index"][ps] == k]
                with h5py.File(reader.files[k], "r") as g:
                    wf = g["Waveforms"]
                    for s in range(0, len(here), batch_size):
                        chunk = here[s:s + batch_size]
                        raw = np.stack([wf[str(int(r))][()] for r in table["row"][chunk]]).astype(np.int16)
                        idx = np.array([where[int(q)] for q in chunk]); o = np.argsort(idx); ds[idx[o]] = raw[o]
        else:
            per_event = [None] * len(positions)
            for pos, batch in reader.iter_inputs(split, batch_size=batch_size, indices=positions, table=table):
                for q, h in zip(pos, batch):
                    per_event[where[int(q)]] = np.asarray(h, dtype=np.float32)
            sizes = np.array([len(h) for h in per_event], dtype=np.int64)
            f.create_dataset("hits", data=np.concatenate(per_event), dtype=np.float32)
            f.create_dataset("offsets", data=np.r_[0, np.cumsum(sizes)], dtype=np.int64)
        f.attrs["seconds_to_build"] = round(time.time() - t0, 1)
    tmp.replace(p)
    return p


def load_all(dataset: str, name: str):
    """Return (items, event_ids): the whole cache in memory as a list of per-event inputs."""
    from .readers.exo200 import EXO200Reader
    with h5py.File(path_for(dataset, name), "r") as f:
        ids = f["event_id"][()].astype(str)
        if dataset == "mjd":
            items = list(f["inputs"][()])
        elif dataset == "exo200":
            raw = f["inputs_int16"][()]
            items = []
            for s in range(0, len(raw), 256):   # preprocess each chunk ONCE and keep per-event copies, never views into the chunk
                chunk = EXO200Reader.preprocess(raw[s:s + 256])
                items.extend(np.ascontiguousarray(chunk[i]) for i in range(len(chunk)))
            del raw
        else:
            off = f["offsets"][()]; hits = f["hits"][()]
            items = [hits[off[i]:off[i + 1]] for i in range(len(ids))]
    return items, ids


def iter_batches(dataset: str, name: str, batch_size=256):
    """Stream the cache in storage order: (positions into the cache order, inputs)."""
    from .readers.exo200 import EXO200Reader
    with h5py.File(path_for(dataset, name), "r") as f:
        n = len(f["event_id"])
        for s in range(0, n, batch_size):
            e = min(s + batch_size, n)
            if dataset == "mjd":
                yield np.arange(s, e), f["inputs"][s:e]
            elif dataset == "exo200":
                yield np.arange(s, e), EXO200Reader.preprocess(f["inputs_int16"][s:e])
            else:
                off = f["offsets"][()]; block = f["hits"][off[s]:off[e]]
                yield np.arange(s, e), [block[off[i] - off[s]: off[i + 1] - off[s]] for i in range(s, e)]
