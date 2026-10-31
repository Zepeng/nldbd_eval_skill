#!/usr/bin/env python3
"""Collect every runs/<model>/<dataset>/L<fraction>/s<seed>/metrics.json into results.md / results.csv.

Only benchmark-valid results (all gates passed) enter the main table; invalid ones are listed
separately with their failed gates. The paper's Region + MLP value is shown as a reference row.

Usage: python -m validation.report [--runs runs] [--output results]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from validation.readers.common import BENCHMARK, DATASETS  # noqa: E402

PRETTY = {name: spec["title"] for name, spec in BENCHMARK["datasets"].items()}
PRIMARY = BENCHMARK["primary_metric"]
# Table 1 of the paper, Region + MLP, energy-matched AUC. MJD/EXO-200 are strict600 values;
# NEXT/SuperNEMO were reported under their own profiles (see validation_skill.md, section 6).
PAPER_REGION_MLP = {"mjd": 0.973, "exo200": 0.970, "next": 0.988, "supernemo": 0.591}


def collect(runs: Path, mode="full", metric=None):
    metric = metric or PRIMARY
    rows = []
    pattern = "*/*/L*/s*/metrics.json" if mode == "full" else "*/*/L*/s*/coarse/metrics.json"
    for path in sorted(runs.glob(pattern)):
        m = json.loads(path.read_text())
        if m.get("mode", "full") != mode:
            continue
        parts = path.parts[:-1] if mode == "full" else path.parts[:-2]
        model, dataset, regime, seed = parts[-4], parts[-3], parts[-2], parts[-1]
        value = (m.get("metrics") or {}).get(metric, {}).get("value") if "metrics" in m else m.get(metric)
        rows.append({"model": model, "dataset": dataset, "label_fraction": float(regime[1:]), "seed": int(seed[1:]),
                     "metric": metric, "matched_auc": value, "benchmark_valid": m["benchmark_valid"],
                     "benchmark_version": m.get("benchmark_version", "1.0"),
                     "failed_gates": ";".join(m["failed_gates"] + m["skipped_gates"]), "path": str(path)})
    return rows


def fmt(values):
    if not values:
        return "–"
    if len(values) == 1:
        return f"{values[0]:.4f}"
    import statistics
    return f"{statistics.mean(values):.4f} ± {statistics.stdev(values):.4f} (n={len(values)})"


def table(rows, title, note):
    valid = [r for r in rows if r["benchmark_valid"] and r["matched_auc"] is not None]
    lines = [title, "", note, "",
             "| Model | Labels | " + " | ".join(PRETTY[d] for d in DATASETS) + " |", "|---|---|" + "---|" * len(DATASETS)]
    keys = sorted({(r["model"], r["label_fraction"]) for r in valid}, key=lambda k: (k[0], -k[1]))
    for model, frac in keys:
        cells = [fmt([r["matched_auc"] for r in valid if r["model"] == model and r["label_fraction"] == frac and r["dataset"] == d]) for d in DATASETS]
        lines.append(f"| {model} | {frac:g} | " + " | ".join(cells) + " |")
    return lines


def main(runs: Path, output: Path, metric=None):
    metric = metric or PRIMARY
    rows = collect(runs, "full", metric)
    coarse = collect(runs, "coarse", metric)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "results.csv").open("w", newline="") as f:
        allrows = [dict(r, mode="full") for r in rows] + [dict(r, mode="coarse") for r in coarse]
        w = csv.DictWriter(f, fieldnames=list(allrows[0].keys()) if allrows else ["model"]); w.writeheader(); w.writerows(allrows)
    valid = [r for r in rows if r["benchmark_valid"] and r["matched_auc"] is not None]
    lines = [f"# {metric} on the frozen test splits ({BENCHMARK['name']} v{BENCHMARK['version']}, profile {BENCHMARK['profile']})", "",
             "Mean ± std over seeds when more than one seed exists. Only benchmark-valid runs (all gates passed) are shown."
             + ("" if metric == PRIMARY else f" NOTE: {metric} is a secondary metric; the benchmark metric is {PRIMARY}."), "",
             "| Model | Labels | " + " | ".join(PRETTY[d] for d in DATASETS) + " |", "|---|---|" + "---|" * len(DATASETS)]
    keys = sorted({(r["model"], r["label_fraction"]) for r in valid}, key=lambda k: (k[0], -k[1]))
    for model, frac in keys:
        cells = [fmt([r["matched_auc"] for r in valid if r["model"] == model and r["label_fraction"] == frac and r["dataset"] == d]) for d in DATASETS]
        lines.append(f"| {model} | {frac:g} | " + " | ".join(cells) + " |")
    if metric == PRIMARY:
        lines.append("| *paper Region + MLP (Table 1, full labels)* | 1 | " + " | ".join(f"{PAPER_REGION_MLP.get(d, float('nan')):.3f}" for d in DATASETS) + " |")
    if coarse:
        lines += [""] + table(coarse, "## Coarse evaluation (frozen validation subset; development numbers, NOT the benchmark)",
                              "Same metric and gates on a seeded fraction of the validation split; use it to compare candidates quickly.")
    bad = [r for r in rows if not r["benchmark_valid"]]
    if bad:
        lines += ["", "## Runs that are NOT benchmark-valid", "", "| Model | Dataset | Labels | Seed | AUC_w | Failed or skipped gates |", "|---|---|---|---|---|---|"]
        lines += [f"| {r['model']} | {PRETTY[r['dataset']]} | {r['label_fraction']:g} | {r['seed']} | "
                  f"{'–' if r['matched_auc'] is None else f'{r['matched_auc']:.4f}'} | {r['failed_gates']} |" for r in bad]
    (output / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--output", type=Path, default=Path("results"))
    ap.add_argument("--metric", default=None, help=f"registered metric to tabulate (default: the primary one, {PRIMARY})")
    a = ap.parse_args()
    main(a.runs, a.output, a.metric)
