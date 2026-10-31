#!/usr/bin/env python3
"""Score one candidate foundation model for an external optimiser (SIDERIUS), under the frozen
profile in siderius_eval_profile.json. The optimiser never touches the harness: it only provides
the module that implements fm_contract.md and asks for 'trial' or 'formal'.

A trial round never touches the test split (coarse predictions only); a formal round scores both.
Writes <output>/score.json with the aggregate (mean primary metric over datasets), every
per-dataset / per-mode / per-regime value, and the paths of the underlying runs; prints the JSON.
Exit code 0 even when the aggregate is null (the JSON says why); exit 2 on a harness error.

Usage:
  python -m validation.score_fm --fm-module cand.py --name cand_007 --round trial  --output scores/cand_007_trial
  python -m validation.score_fm --fm-module cand.py --name cand_007 --round formal --output scores/cand_007_formal
Options: --runs-root <dir> (default runs/), --pretraining-ids ids.npz, --datasets mjd,exo200 (subset for smoke tests only),
--parallel N (concurrent fine-tunes; env NLDBD_SCORE_PARALLEL), --gpus 0,1,2,3 (round-robin device assignment; env NLDBD_SCORE_GPUS).
A formal round is 12 independent fine-tunes (probe, full, scratch x 4 datasets): with --parallel 12 --gpus 0,1,2,3 it takes one
fine-tune's wall time instead of twelve.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from validation.readers.common import BENCHMARK  # noqa: E402

PROFILE = json.loads((ROOT / "validation" / "siderius_eval_profile.json").read_text())


def run_finetune(args, dataset, mode, frac, max_hours, runs_root, gpu=None):
    cmd = [sys.executable, "-B", "-m", "validation.finetune", "--fm-module", str(args.fm_module), "--name", args.name, "--dataset", dataset,
           "--label-fraction", str(frac), "--mode", mode, "--seed", str(args.seed), "--max-hours", str(max_hours),
           "--eval-modes", "coarse" if args.round == "trial" else "coarse,full"]
    if args.pretraining_ids:
        cmd += ["--pretraining-ids", str(args.pretraining_ids)]
    env = dict(__import__("os").environ, NLDBD_RUNS_ROOT=str(runs_root))
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    run = Path(runs_root) / f"{args.name}__{mode}" / dataset / f"L{frac}" / f"s{args.seed}"
    out = {"dataset": dataset, "mode": mode, "label_fraction": frac, "run_dir": str(run), "seconds": round(time.time() - t0, 1),
           "exit_code": r.returncode}
    if r.returncode != 0:
        out["error"] = (r.stderr or r.stdout)[-2000:]
        return out
    for which, path in (("coarse", run / "coarse" / "metrics.json"), ("full", run / "metrics.json")):
        if path.exists():
            m = json.loads(path.read_text())
            out[which] = {"primary_value": m["primary_value"], "benchmark_valid": m["benchmark_valid"],
                          "failed_gates": m["failed_gates"], "skipped_gates": m["skipped_gates"],
                          "metrics": {k: v["value"] for k, v in m["metrics"].items()}}
    return out


def main(args):
    spec = PROFILE[args.round]
    datasets = args.datasets.split(",") if args.datasets else PROFILE["datasets"]
    runs_root = Path(args.runs_root) if args.runs_root else ROOT / "runs"
    tasks = []   # (dataset, mode, fraction): every fine-tune of the round is independent, so they run --parallel at a time
    for ds in datasets:
        for mode in spec["finetune_modes"]:
            for frac in spec["label_fractions"]:
                tasks.append((ds, mode, frac))
        if args.round == "formal" and spec.get("scratch_control"):
            tasks.append((ds, "scratch", spec["label_fractions"][-1]))
    gpus = [g.strip() for g in args.gpus.split(",")] if args.gpus else [None]
    t0 = time.time()
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
        futures = [pool.submit(run_finetune, args, ds, mode, frac, spec["max_hours_per_finetune"], runs_root, gpus[i % len(gpus)])
                   for i, (ds, mode, frac) in enumerate(tasks)]
        results = []
        for f in futures:
            results.append(f.result())
            print(json.dumps({k: results[-1].get(k) for k in ("dataset", "mode", "label_fraction", "seconds", "exit_code")}), flush=True)
    # aggregate: mean over datasets of the primary metric in the round's evaluation mode, using the
    # strongest fine-tune mode listed (last in finetune_modes) and the largest regime listed
    key_mode, key_frac, eval_mode = spec["finetune_modes"][-1], spec["label_fractions"][-1], spec["evaluation_mode"]
    per_dataset = {}
    for r in results:
        if r["mode"] == key_mode and r["label_fraction"] == key_frac and eval_mode in r:
            v = r[eval_mode]["primary_value"] if r[eval_mode]["benchmark_valid"] or eval_mode == "coarse" else None
            per_dataset[r["dataset"]] = v
    values = [per_dataset.get(ds) for ds in datasets]
    aggregate = None if any(v is None for v in values) else float(sum(values) / len(values))
    out = {"benchmark": BENCHMARK["name"], "benchmark_version": BENCHMARK["version"], "profile_version": PROFILE["version"],
           "round": args.round, "name": args.name, "fm_module": str(args.fm_module), "datasets": datasets,
           "score": aggregate, "score_definition": f"mean over datasets of {BENCHMARK['primary_metric']} ({eval_mode} mode, {key_mode} fine-tune, {key_frac:g} labels)",
           "per_dataset": per_dataset, "runs": results, "hours": round((time.time() - t0) / 3600, 3),
           "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    Path(args.output).mkdir(parents=True, exist_ok=True)
    (Path(args.output) / "score.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"round": args.round, "score": aggregate, "per_dataset": per_dataset, "hours": out["hours"], "output": str(Path(args.output) / "score.json")}))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fm-module", required=True); ap.add_argument("--name", required=True)
    ap.add_argument("--round", required=True, choices=("trial", "formal"))
    ap.add_argument("--output", required=True)
    ap.add_argument("--runs-root", default=None); ap.add_argument("--pretraining-ids", default=None)
    ap.add_argument("--datasets", default=None); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--parallel", type=int, default=int(__import__("os").environ.get("NLDBD_SCORE_PARALLEL", "1")),
                    help="fine-tunes run concurrently (each is one process; a probe needs ~2 GB RAM, a full fine-tune of EXO-200 ~6 GB)")
    ap.add_argument("--gpus", default=__import__("os").environ.get("NLDBD_SCORE_GPUS", ""),
                    help="comma-separated CUDA device ids assigned round-robin to the fine-tunes (default: inherit CUDA_VISIBLE_DEVICES)")
    try:
        sys.exit(main(ap.parse_args()))
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"error": repr(exc)}), file=sys.stderr); sys.exit(2)
