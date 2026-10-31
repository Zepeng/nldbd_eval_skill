#!/usr/bin/env python3
"""Consolidate the 14,973-file NEXT release into one HDF5 file in canonical split order.

Lossless: the same float32 hit values (x, y, z, energy MeV), the same event ids, labels and the
same float64 event energies as the per-file reader. Events are stored contiguously, split by
split (train, validation, test, each a contiguous range), so reading a split is a sequential scan.
The NEXT reader uses this file automatically when it exists and its inventory digest matches
the frozen split manifest.

Usage: python -m validation.build_next_combined [--output /home/klz/Data/NEXT_combined/next_combined.h5]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from validation.readers.next import NEXTReader, TABLE, CATEGORY_LABEL, ROW_LABEL  # noqa: E402
from validation.readers.common import SPLITS  # noqa: E402

DEFAULT_OUTPUT = "/home/klz/Data/NEXT_combined/next_combined.h5"
HIT_CHUNK = 1 << 18


def main(output: Path):
    reader = NEXTReader()
    reader.combined = None  # force the per-file path while building
    manifest = reader.manifest
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(".building.h5")
    t0 = time.time()
    n_total = sum(manifest["counts"][s] for s in SPLITS)
    with h5py.File(tmp, "w") as out:
        hits = out.create_dataset("hits", shape=(0, 4), maxshape=(None, 4), dtype=np.float32, chunks=(HIT_CHUNK, 4))
        meta = {"hit_offset": np.zeros(n_total + 1, dtype=np.int64), "label": np.zeros(n_total, np.int8),
                "energy_MeV": np.zeros(n_total, np.float64), "ordinal": np.zeros(n_total, np.int32),
                "source_event_id": np.zeros(n_total, np.int64), "event_id": [], "file": [], "split": []}
        i, written = 0, 0
        ranges = {}
        buf, buf_rows = [], 0
        for split in SPLITS:
            start = i
            for slc in manifest["splits"][split]:
                rel = slc["relative_path"]
                with h5py.File(reader.root / rel, "r") as f:
                    t = f[TABLE][()]
                ev, starts, stops = reader._events(t)
                if not np.all(t["values_block_2"][:, 0] == ROW_LABEL[slc["category"]]):
                    raise ValueError(f"{rel}: unexpected row labels")
                e_hit = np.asarray(t["values_block_1"][:, 3], dtype=np.float64)
                sums = np.add.reduceat(e_hit, starts)
                xyz_e = np.asarray(t["values_block_1"], dtype=np.float32)
                for o in range(slc["event_start"], slc["event_stop"]):
                    h = xyz_e[starts[o]:stops[o]]
                    buf.append(h); buf_rows += len(h)
                    meta["hit_offset"][i + 1] = meta["hit_offset"][i] + len(h)
                    meta["label"][i] = CATEGORY_LABEL[slc["category"]]
                    meta["energy_MeV"][i] = sums[o]
                    meta["ordinal"][i] = o
                    meta["source_event_id"][i] = int(ev[o])
                    meta["event_id"].append(f"NEXT::{rel}::{int(ev[o])}")
                    meta["file"].append(rel); meta["split"].append(split)
                    i += 1
                if buf_rows >= 8 * HIT_CHUNK:
                    block = np.concatenate(buf); hits.resize(written + len(block), axis=0); hits[written:] = block
                    written += len(block); buf, buf_rows = [], 0
                    print(f"  {i}/{n_total} events, {written / 1e9:.2f} G hits, {time.time() - t0:.0f} s", flush=True)
            ranges[split] = (start, i)
            if manifest["counts"][split] != i - start:
                raise RuntimeError(f"{split}: {i - start} events, manifest says {manifest['counts'][split]}")
        if buf:
            block = np.concatenate(buf); hits.resize(written + len(block), axis=0); hits[written:] = block; written += len(block)
        assert written == meta["hit_offset"][-1]
        for k in ("hit_offset", "label", "energy_MeV", "ordinal", "source_event_id"):
            out.create_dataset(k, data=meta[k])
        out.create_dataset("event_id", data=np.array(meta["event_id"], dtype="S"))
        out.create_dataset("file", data=np.array(meta["file"], dtype="S"))
        out.create_dataset("split", data=np.array(meta["split"], dtype="S"))
        out.attrs["inventory_digest"] = manifest["inventory"]["digest"]
        out.attrs["split_ranges"] = json.dumps(ranges)
        out.attrs["source_root"] = str(reader.root)
        out.attrs["seconds_to_build"] = round(time.time() - t0, 1)
    os.replace(tmp, output)
    print(f"wrote {output} ({output.stat().st_size / 1e9:.1f} GB, {written} hits, {n_total} events) in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=Path(os.environ.get("NLDBD_NEXT_COMBINED", DEFAULT_OUTPUT)))
    main(ap.parse_args().output)
