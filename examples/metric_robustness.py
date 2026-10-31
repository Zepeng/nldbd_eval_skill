#!/usr/bin/env python3
"""Metric-property tests on existing full-mode predictions (no training).

For each run given (runs/<model>/<ds>/L<f>/s<seed>/predictions.npz) and its frozen test labels
and energies, compute matched_auc (and inclusive_auc as a diagnostic) under:
  * monotone transforms of the score (affine, sigmoid, rank, cube)   -> must be identical
  * score inversion                                                    -> 1 - value
  * energy contamination: score + lambda * z(energy), lambda in a grid -> matched should move much
    less than inclusive
  * additive Gaussian noise of sigma = k * std(score)                   -> smooth degradation
  * random half-splits of the test population (10 draws)               -> spread = resolution
Writes results/metric_robustness.md and .json.
Usage: python examples/metric_robustness.py runs/tiny/mjd/L0.1/s42 runs/tiny/exo200/L0.01/s42 ...
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from validation.readers.common import BENCHMARK, MANIFESTS  # noqa: E402
from validation.metrics import MetricContext, compute_all  # noqa: E402

PRIMARY = BENCHMARK["primary_metric"]


def evaluate(label, score, energy, group):
    m = compute_all([PRIMARY, "inclusive_auc"], MetricContext(label=label, score=score, energy_keV=energy, group=group))
    return m[PRIMARY]["value"], m["inclusive_auc"]["value"]


def main(run_dirs):
    rows, report = [], ["# Metric robustness tests (version %s)" % BENCHMARK["version"], ""]
    for run in run_dirs:
        run = Path(run); ds = run.parts[-3]
        ids = np.load(MANIFESTS / f"{ds}_ids.npz", allow_pickle=False)
        pred = np.load(run / "predictions.npz", allow_pickle=False)
        te = ids["test_event_id"]; order = np.argsort(te); pos = order[np.searchsorted(te[order], pred["event_id"].astype(str))]
        y, e, g = ids["test_label"][pos].astype(np.int8), ids["test_energy_keV"][pos], ids["test_group"][pos].astype(str)
        s = pred["score"].astype(np.float64)
        base_m, base_i = evaluate(y, s, e, g)
        out = {"run": str(run), "dataset": ds, "matched": base_m, "inclusive": base_i}
        rank = np.argsort(np.argsort(s)).astype(np.float64)
        out["monotone"] = {name: evaluate(y, f(s), e, g)[0] for name, f in
                           (("affine", lambda v: 3.0 * v - 7.0), ("sigmoid", lambda v: 1 / (1 + np.exp(-v))), ("rank", lambda v: rank), ("cube", lambda v: v ** 3))}
        out["inverted"] = evaluate(y, -s, e, g)[0]
        ok = np.isfinite(e)
        z = np.zeros_like(s); z[ok] = (e[ok] - e[ok].mean()) / e[ok].std()
        sz = (s - s.mean()) / s.std()
        out["energy_mix"] = {str(lam): evaluate(y, sz + lam * z, e, g) for lam in (0.0, 0.3, 1.0, 3.0, 10.0)}
        rng = np.random.default_rng(0)
        out["noise"] = {str(k): evaluate(y, sz + k * rng.standard_normal(len(sz)), e, g)[0] for k in (0.1, 0.5, 1.0, 2.0)}
        halves = []
        for _ in range(10):
            sel = rng.random(len(s)) < 0.5
            halves.append(evaluate(y[sel], s[sel], e[sel], g[sel])[0])
        halves = [h for h in halves if h is not None]
        out["half_split"] = {"mean": float(np.mean(halves)), "std": float(np.std(halves)), "n": len(halves)}
        rows.append(out)
        report += [f"## {ds}: `{run}`", "",
                   f"baseline matched_auc {base_m:.4f}, inclusive_auc {base_i:.4f}", "",
                   "| test | matched_auc | note |", "|---|---|---|"]
        for k, v in out["monotone"].items():
            report.append(f"| monotone: {k} | {v:.6f} | identical to baseline: {abs(v - base_m) < 1e-9} |")
        report.append(f"| inverted score | {out['inverted']:.6f} | 1 - baseline = {1 - base_m:.6f} |")
        for lam, (m, i) in out["energy_mix"].items():
            report.append(f"| + {lam} x z(energy) | {m:.4f} | inclusive {i:.4f} |")
        for k, v in out["noise"].items():
            report.append(f"| + noise {k} x std | {v:.4f} | |")
        report.append(f"| half-split resolution | {out['half_split']['mean']:.4f} ± {out['half_split']['std']:.4f} | std over 10 random halves |")
        report.append("")
        print(f"{ds}: baseline {base_m:.4f}; monotone max dev {max(abs(v - base_m) for v in out['monotone'].values()):.1e}; "
              f"energy-mix matched {[round(v[0],3) for v in out['energy_mix'].values()]} inclusive {[round(v[1],3) for v in out['energy_mix'].values()]}; "
              f"half-split std {out['half_split']['std']:.4f}", flush=True)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "metric_robustness.json").write_text(json.dumps(rows, indent=2) + "\n")
    (ROOT / "results" / "metric_robustness.md").write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
