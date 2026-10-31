#!/usr/bin/env python3
"""Re-score the paper's saved per-event predictions under the strict600 profile.

Needs the paper's event-data companion (folder with ``events/`` and ``historical_transformers/``;
default: ``NLDBD_PAPER_EVENT_DATA``). Writes ``baselines/strict600_baselines.json`` and
``baselines/strict600_baselines.md``. Only the energy-matched AUC is reported.

Usage: python -m validation.baselines [--event-data DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from validation.energybench.metrics import evaluate, fingerprint, load_config, to_keV  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_EVENT_DATA = "/home/wenyu/iclr final paper/EnergyBench_reproducibility_20260923/event_data"
DATASET_KEY = {"MJD": "mjd", "EXO-200": "exo200", "NEXT": "next", "SuperNEMO": "supernemo"}
APPROACH = {"mvcnn": "native", "gine": "native", "bigru": "native", "mamba": "native"}
PRETTY = {"mvcnn": "Multi-view CNN", "gine": "Static GINE", "bigru": "BiGRU", "mamba": "PointMamba-lite",
          "entity_mlp": "Entity + MLP", "entity_fourier": "Entity + Fourier", "entity_rope": "Entity + RoPE",
          "region_mlp": "Region + MLP", "region_fourier": "Region + Fourier", "region_rope": "Region + RoPE",
          "summary_mlp": "Summary + MLP", "summary_fourier": "Summary + Fourier", "summary_rope": "Summary + RoPE"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def standardize(path: Path, dataset: str):
    """Return (label, score, energy_keV, group, event_id) in the evaluation convention."""
    d = np.load(path, allow_pickle=False)
    label = d["label"].astype(np.int8)
    score = d["score"].astype(np.float64)
    if "energy_keV" in d.files:
        e = d["energy_keV"].astype(np.float64)
    elif "energy" in d.files:  # historical NEXT transformer files: MeV
        e = to_keV(d["energy"].astype(np.float64), "MeV")
    elif "energy_condition" in d.files:  # historical SuperNEMO transformer files: keV
        e = d["energy_condition"].astype(np.float64)
    else:
        raise ValueError(f"{path}: no energy field")
    group = d["group"].astype(str) if "group" in d.files else (d["category"].astype(str) if "category" in d.files else label.astype(str))
    return label, score, e, group, d["event_id"].astype(str)


def main(event_data: Path):
    config = load_config("strict600")
    rows = []
    files = sorted((event_data / "events").glob("*.npz")) + sorted((event_data / "historical_transformers").glob("*.npz"))
    for path in files:
        stem = path.stem
        ds, model = stem.split("__", 1)
        if ds not in DATASET_KEY:
            continue  # NEXT-exploratory batch and the SuperNEMO energy-only illustration are not main-table rows
        label, score, e, group, ids = standardize(path, ds)
        r = evaluate(label, score, e, group=group, energy_unit="keV", config=config)
        rows.append({"dataset": DATASET_KEY[ds], "model": model, "model_name": PRETTY.get(model, model),
                     "approach": APPROACH.get(model, "transformer" if model in PRETTY else "next_extended"),
                     "main_table": model in PRETTY, "n_events": int(len(score)),
                     "matched_auc": r["matching"]["matched_auc"], "matching_status": r["matching"]["status"],
                     "inclusive_auc_diagnostic": r["inclusive_auc"],
                     "source": str(path.relative_to(event_data)), "source_sha256": sha256(path),
                     "label_counts": {"0": int((label == 0).sum()), "1": int((label == 1).sum())}})
        print(f"{ds:10s} {model:45s} AUC_w={r['matching']['matched_auc']:.4f} ({r['matching']['status']})", flush=True)
    out = {"profile": "strict600", "protocol_sha256": fingerprint(config), "event_data": str(event_data),
           "note": "Paper models re-scored under strict600 for every dataset. For MJD and EXO-200 these equal "
                   "Table 1; for NEXT and SuperNEMO Table 1 used other profiles, so small differences are expected.",
           "records": rows}
    (HERE / "baselines").mkdir(exist_ok=True)
    (HERE / "baselines" / "strict600_baselines.json").write_text(json.dumps(out, indent=2) + "\n")
    lines = ["# Paper baselines re-scored under strict600 (energy-matched AUC)", "",
             "| Model | MJD | EXO-200 | NEXT | SuperNEMO |", "|---|---|---|---|---|"]
    by = {(r["dataset"], r["model"]): r for r in rows}
    for model in PRETTY:
        cells = []
        for ds in ("mjd", "exo200", "next", "supernemo"):
            r = by.get((ds, model))
            cells.append("–" if r is None else ("n/a" if r["matched_auc"] is None else f"{r['matched_auc']:.4f}"))
        lines.append(f"| {PRETTY[model]} | " + " | ".join(cells) + " |")
    extra = [r for r in rows if not r["main_table"]]
    if extra:
        lines += ["", "## Extended NEXT classification candidates (Table 10)", "", "| Model | AUC_w |", "|---|---|"]
        lines += [f"| {r['model']} | {r['matched_auc']:.4f} |" for r in sorted(extra, key=lambda r: -(r["matched_auc"] or 0))]
    (HERE / "baselines" / "strict600_baselines.md").write_text("\n".join(lines) + "\n")
    print("wrote", HERE / "baselines" / "strict600_baselines.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--event-data", type=Path, default=Path(os.environ.get("NLDBD_PAPER_EVENT_DATA", DEFAULT_EVENT_DATA)))
    main(ap.parse_args().event_data)
