#!/usr/bin/env python3
"""Run any model over the frozen test split and write the standardized predictions.npz.

The model is a Python module (file path or import name) exposing two functions:

    load_model(dataset: str) -> Any                 # build / load weights for this dataset
    score(model, inputs) -> np.ndarray              # one float per event, higher = signal-like

``inputs`` is exactly what ``Reader.iter_inputs`` yields for the dataset (see readers/*.py):
MJD float32 [B, 1, 3800]; EXO-200 float32 [B, 226, 300]; NEXT list of float32 (n_hits, 4)
[x, y, z, hit energy MeV]; SuperNEMO list of float32 (n_hits, 4) [tX, tY, tZ, tR].
No labels, energies or categories ever reach the model.

Usage:
  python -m validation.make_predictions --dataset mjd --model-module examples/tiny_models.py \
      --output runs/tiny/mjd/L0.01/s42/predictions.npz [--split test] [--batch-size 256] [--limit N]
  --subset coarse  scores the frozen coarse subset of the validation split (for --mode coarse).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from validation.readers import get_reader  # noqa: E402
from validation.readers.common import DATASETS, MANIFESTS  # noqa: E402
from validation.freeze_splits import coarse_positions  # noqa: E402
from validation import coarse_cache  # noqa: E402


def load_module(spec: str):
    path = Path(spec)
    if path.suffix == ".py" and path.exists():
        mod_spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(mod_spec)
        sys.modules[path.stem] = module
        mod_spec.loader.exec_module(module)
        return module, hashlib.sha256(path.read_bytes()).hexdigest()
    module = importlib.import_module(spec)
    src = Path(module.__file__).read_bytes() if getattr(module, "__file__", None) else b""
    return module, hashlib.sha256(src).hexdigest()


def main(args):
    module, module_sha = load_module(args.model_module)
    reader = get_reader(args.dataset)
    if args.subset == "coarse":
        args.split = "validation"
    cache_ready = args.subset == "coarse" and args.limit is None and not args.no_cache and coarse_cache.is_valid(args.dataset)
    if cache_ready:
        # cached coarse run: never touch the raw data; the frozen manifest supplies the event ids
        frozen = np.load(MANIFESTS / f"{args.dataset}_ids.npz", allow_pickle=False)
        table = {"event_id": frozen["coarse_event_id"].astype(str)}
        n = len(table["event_id"])
        indices = np.arange(n)
    else:
        table = reader.table(args.split)
        n = len(table["event_id"])
        indices = None if args.limit is None else np.arange(min(args.limit, n))
        if args.subset == "coarse":
            c = json.loads((MANIFESTS / f"{args.dataset}.json").read_text())["coarse"]
            indices = coarse_positions(n, c["fraction"], c["seed"])
            if args.limit is not None:
                indices = indices[:args.limit]
    model = module.load_model(args.dataset)
    scores = np.full(n, np.nan)
    t0 = time.time()
    done, next_report = 0, 20000
    source = None if cache_ready else reader.iter_inputs(args.split, batch_size=args.batch_size, indices=indices, table=table)
    cache_used = False
    if args.subset == "coarse" and args.limit is None and not args.no_cache:
        if not cache_ready:
            print(f"  building the coarse input cache at {coarse_cache.cache_path(args.dataset)} (one time)", flush=True)
            coarse_cache.build(args.dataset, reader, table, indices, batch_size=args.batch_size)
        if not np.array_equal(coarse_cache.cached_event_ids(args.dataset), table["event_id"][indices]):
            raise RuntimeError("coarse cache event ids do not match the manifest; delete the cache file")
        source = ((indices[p], x) for p, x in coarse_cache.iter_cached(args.dataset, batch_size=args.batch_size))
        cache_used = True
    for positions, inputs in source:
        s = np.asarray(module.score(model, inputs), dtype=np.float64).reshape(-1)
        if s.shape[0] != len(positions):
            raise RuntimeError(f"score() returned {s.shape[0]} values for {len(positions)} events")
        scores[positions] = s
        done += len(positions)
        if done >= next_report:
            print(f"  {done}/{n if indices is None else len(indices)} events, {time.time() - t0:.0f} s", flush=True)
            next_report += 20000
    keep = np.arange(n) if indices is None else indices
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, event_id=table["event_id"][keep], score=scores[keep])
    prov = {"dataset": args.dataset, "split": args.split, "subset": args.subset, "coarse_cache_used": cache_used, "n_events": int(len(keep)), "model_module": str(args.model_module),
            "model_module_sha256": module_sha, "batch_size": args.batch_size, "seconds": round(time.time() - t0, 1),
            "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model_info": getattr(module, "MODEL_INFO", None)}
    out.with_name(out.stem + "_provenance.json").write_text(json.dumps(prov, indent=2, default=str) + "\n")
    print(json.dumps({"output": str(out), "n": int(len(keep)), "nonfinite": int((~np.isfinite(scores[keep])).sum()), "seconds": prov["seconds"]}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=DATASETS)
    ap.add_argument("--model-module", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--split", default="test", choices=("train", "validation", "test"))
    ap.add_argument("--subset", default=None, choices=(None, "coarse"), help="'coarse' = frozen validation subset")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None, help="first N events only (smoke tests; result is not benchmark-valid)")
    ap.add_argument("--no-cache", action="store_true", help="coarse mode: read through the dataset reader instead of the local cache")
    main(ap.parse_args())
