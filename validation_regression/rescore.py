#!/usr/bin/env python3
"""Re-evaluate every saved regression prediction under runs_regression/ with the current metric definitions
(benchmark.json); rewrites the metrics JSON beside each predictions file. Usage: python -m validation_regression.rescore"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from validation_regression.evaluate import evaluate, headline_line  # noqa: E402
from validation_regression.views import get_view  # noqa: E402

RUNS = ROOT / "runs_regression"
tables = {}
for pred in sorted(p for p in RUNS.rglob("predictions*.npz") if "_obsolete" not in str(p)):
    ds = next(p for p in pred.parts if p in ("mjd", "exo200", "supernemo"))
    split = "validation" if pred.stem.endswith("_validation") else "test"
    old = pred.with_name("metrics.json" if pred.stem == "predictions" else pred.stem.replace("predictions", "metrics") + ".json")
    model = json.loads(old.read_text()).get("model") if old.exists() else None
    ids = pred.parent / "training_ids.npz"
    if (ds, split) not in tables:
        tables[(ds, split)] = get_view(ds).table(split)
    m = evaluate(ds, split, pred, ids if ids.exists() else None, model, table=tables[(ds, split)])
    old.write_text(json.dumps(m, indent=2, default=float) + "\n")
    print(f"{str(pred.relative_to(RUNS)):70s} {headline_line(m)}", flush=True)
