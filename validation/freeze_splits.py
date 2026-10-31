#!/usr/bin/env python3
"""Freeze the official splits of one dataset into manifests/<dataset>.json and <dataset>_ids.npz.

The JSON records, per split, event counts, label counts and SHA-256 identity hashes; the nested
label-regime subsets (1 %, 10 %, 100 % of the training split, seed 42); the COARSE evaluation
subset (a seeded fraction of the validation split: the smallest candidate fraction that is at least
COARSE_MIN_FRACTION[dataset], on which the strict600 matched AUC is estimable AND each class keeps at
least COARSE_COVERAGE_RATIO of the coverage that the full validation split keeps); and the energy-only shortcut floor (score = energy) on the
test split and on the coarse subset. The NPZ stores the event ids of every split and,
for the test split, labels and energies, so predictions can be checked on a machine without the
raw data.

Usage: python -m validation.freeze_splits <dataset> [--force]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from validation.readers import get_reader  # noqa: E402
from validation.readers.common import MANIFESTS, SPLITS, sha256_lines  # noqa: E402
from validation.energybench.metrics import evaluate, load_config, fingerprint  # noqa: E402

from validation.readers.common import BENCHMARK, dataset_spec  # noqa: E402

LABEL_FRACTIONS = tuple(BENCHMARK["label_regimes"]["fractions"])
SUBSET_SEED = BENCHMARK["label_regimes"]["seed"]
COARSE_SEED = BENCHMARK["coarse"]["seed"]
COARSE_CANDIDATE_FRACTIONS = tuple(BENCHMARK["coarse"]["candidate_fractions"])
COARSE_COVERAGE_RATIO = BENCHMARK["coarse"]["coverage_ratio"]
COARSE_MIN_FRACTION = {name: spec.get("coarse_min_fraction", 0.0) for name, spec in BENCHMARK["datasets"].items()}


def labeled_subset_positions(n_train: int, fraction: float, seed: int = SUBSET_SEED) -> np.ndarray:
    """Positions (into the canonical train order) of the nested labeled subset."""
    perm = np.random.default_rng(seed).permutation(n_train)
    k = n_train if fraction >= 1.0 else int(round(fraction * n_train))
    return np.sort(perm[:k])


def coarse_positions(n_validation: int, fraction: float, seed: int = COARSE_SEED) -> np.ndarray:
    """Positions (into the canonical validation order) of the coarse evaluation subset."""
    perm = np.random.default_rng(seed).permutation(n_validation)
    k = n_validation if fraction >= 1.0 else int(round(fraction * n_validation))
    return np.sort(perm[:k])


def floor_record(t, pos, config):
    out = {}
    for name, sign in (("plus_energy", 1.0), ("minus_energy", -1.0)):
        r = evaluate(t["label"][pos], sign * t["energy_keV"][pos], t["energy_keV"][pos], group=t["group"][pos],
                     energy_unit="keV", config=config)
        out[name] = {"inclusive_auc": r["inclusive_auc"], "matched_auc": r["matching"]["matched_auc"],
                     "matching_status": r["matching"]["status"], "common_support_keV": r["matching"]["common_support_keV"],
                     "valid_bins": r["matching"]["valid_bin_count"],
                     "coverage": [r["matching"]["classes"][c]["matched_fraction_original_finite"] for c in ("0", "1")]}
    return out


def main(dataset: str, force: bool = False):
    out_json = MANIFESTS / f"{dataset}.json"
    out_npz = MANIFESTS / f"{dataset}_ids.npz"
    if out_json.exists() and not force:
        print(f"{out_json} exists; use --force to regenerate")
        return
    reader = get_reader(dataset)
    config = load_config(BENCHMARK["profile"])
    record = {"dataset": dataset, "benchmark_version": BENCHMARK.get("artefact_version", BENCHMARK["version"]),
              "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "data_root": str(reader.root), "reader": reader.__class__.__name__,
              "label_convention": "1 = signal-like = evaluation-positive class",
              "signal_group": reader.signal_group, "background_group": reader.background_group,
              "profile": BENCHMARK["profile"], "protocol_sha256": fingerprint(config), "splits": {}}
    arrays = {}
    tables = {}
    for split in SPLITS:
        t0 = time.time()
        t = reader.table(split)
        tables[split] = t
        record["splits"][split] = reader.summary(split, t)
        record["splits"][split]["seconds_to_read"] = round(time.time() - t0, 1)
        arrays[f"{split}_event_id"] = t["event_id"]
        print(f"{dataset} {split}: {len(t['event_id'])} events in {time.time() - t0:.1f} s", flush=True)
    for split in SPLITS:
        for other in SPLITS:
            if other > split and np.intersect1d(tables[split]["event_id"], tables[other]["event_id"]).size:
                raise RuntimeError(f"{dataset}: {split} and {other} overlap")
    test = tables["test"]
    arrays["test_label"] = test["label"]
    arrays["test_energy_keV"] = test["energy_keV"]
    arrays["test_group"] = test["group"]
    # nested label regimes on the canonical train order
    n_train = len(tables["train"]["event_id"])
    regimes = {"seed": SUBSET_SEED, "rule": "np.random.default_rng(seed).permutation(n_train)[:round(fraction*n_train)], "
                                            "positions into the canonical train order; subsets are nested", "fractions": {}}
    for frac in LABEL_FRACTIONS:
        pos = labeled_subset_positions(n_train, frac)
        ids = tables["train"]["event_id"][pos]
        regimes["fractions"][str(frac)] = {"n": int(len(pos)), "ids_sha256_sorted": sha256_lines(np.sort(ids)),
                                           "label_counts": {"0": int((tables["train"]["label"][pos] == 0).sum()),
                                                            "1": int((tables["train"]["label"][pos] == 1).sum())}}
    record["label_regimes"] = regimes
    # energy-only shortcut floor on the test split
    record["energy_only_floor_test"] = floor_record(test, np.arange(len(test["event_id"])), config)
    # coarse evaluation subset: smallest candidate fraction of validation with an estimable matched AUC
    val = tables["validation"]
    n_val = len(val["event_id"])
    full_val = floor_record(val, np.arange(n_val), config)["plus_energy"]
    candidates = {}
    for frac in COARSE_CANDIDATE_FRACTIONS:
        if frac < COARSE_MIN_FRACTION.get(dataset, 0.0):
            continue
        pos = coarse_positions(n_val, frac)
        floor = floor_record(val, pos, config)
        f = floor["plus_energy"]
        ratio = [None if (c is None or r in (None, 0)) else c / r for c, r in zip(f["coverage"], full_val["coverage"])]
        candidates[str(frac)] = {"n": int(len(pos)), "matching_status": f["matching_status"], "valid_bins": f["valid_bins"],
                                 "coverage": f["coverage"], "coverage_ratio_to_full_validation": ratio}
        if f["matching_status"] == "ok" and all(x is not None and x >= COARSE_COVERAGE_RATIO for x in ratio):
            break
    record["coarse"] = {"split": "validation", "seed": COARSE_SEED, "fraction": frac, "n": int(len(pos)),
                        "rule": "np.random.default_rng(seed).permutation(n_validation)[:round(fraction*n_validation)], positions into the canonical validation order; "
                                f"smallest candidate fraction with an estimable matched AUC and per-class coverage >= {COARSE_COVERAGE_RATIO} x the full validation split's coverage",
                        "candidate_fractions": list(COARSE_CANDIDATE_FRACTIONS), "coverage_ratio_required": COARSE_COVERAGE_RATIO,
                        "minimum_fraction_for_this_dataset": COARSE_MIN_FRACTION.get(dataset, 0.0),
                        "candidates": candidates, "full_validation_floor": full_val,
                        "label_counts": {"0": int((val["label"][pos] == 0).sum()), "1": int((val["label"][pos] == 1).sum())},
                        "ids_sha256_sorted": sha256_lines(np.sort(val["event_id"][pos])),
                        "energy_only_floor": floor}
    arrays["coarse_event_id"] = val["event_id"][pos]
    arrays["coarse_label"] = val["label"][pos]
    arrays["coarse_energy_keV"] = val["energy_keV"][pos]
    arrays["coarse_group"] = val["group"][pos]
    print(f"{dataset} coarse subset: {frac:g} of validation = {len(pos)} events, "
          f"{floor['plus_energy']['valid_bins']} valid bins, status {floor['plus_energy']['matching_status']}", flush=True)
    tmp = out_npz.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(out_npz)
    record["ids_npz"] = out_npz.name
    tmp_json = out_json.with_suffix(".tmp.json")
    tmp_json.write_text(json.dumps(record, indent=2) + "\n")
    tmp_json.replace(out_json)
    print(f"wrote {out_json} and {out_npz} ({out_npz.stat().st_size / 1e6:.1f} MB)")
    print(json.dumps({k: v for k, v in record["energy_only_floor_test"].items()}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    main(a.dataset, a.force)
