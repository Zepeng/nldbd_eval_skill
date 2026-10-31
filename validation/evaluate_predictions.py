#!/usr/bin/env python3
"""Score one prediction file on the frozen evaluation population and run the integrity gates.

Two modes:
  --mode full    (default) the complete frozen TEST split: the benchmark number.
  --mode coarse  the frozen COARSE subset of the VALIDATION split (manifest["coarse"]): a quick
                 development number with the same metric and gates; never a benchmark result.

Inputs
  predictions.npz   event_id (str, one per test event), score (float, higher = signal-like).
                    Optional: label, energy_keV (checked against the frozen manifest, never used).
  training_ids.npz  event_id: every event whose inputs were used in ANY training step on this
                    dataset (supervised or self-supervised, any split); labeled_event_id: every
                    event whose LABEL was used for a gradient step. Required unless --no-training-ids.

Output: <output>/metrics.json with the energy-matched AUC (strict600), gate results and the full
evaluator record. Exit code 1 when a gate fails. Nothing here reads raw detector data: the frozen
manifest (manifests/<dataset>.json + <dataset>_ids.npz) carries the test labels and energies.

Usage:
  python -m validation.evaluate_predictions --dataset mjd --predictions P.npz \
      --training-ids T.npz --label-fraction 0.01 --output runs/<model>/mjd/L0.01/s42
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from validation.energybench.metrics import evaluate, fingerprint, load_config  # noqa: E402
from validation.readers.common import BENCHMARK, DATASETS, MANIFESTS, dataset_spec, sha256_lines  # noqa: E402
from validation.freeze_splits import labeled_subset_positions  # noqa: E402
from validation.metrics import MetricContext, compute_all  # noqa: E402

NULL_SEED = 0


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_frozen(dataset: str):
    manifest = json.loads((MANIFESTS / f"{dataset}.json").read_text())
    ids = np.load(MANIFESTS / manifest["ids_npz"], allow_pickle=False)
    frozen = {k: ids[k] for k in ids.files}
    for split in ("train", "validation", "test"):
        got = sha256_lines(np.sort(frozen[f"{split}_event_id"]))
        if got != manifest["splits"][split]["ids_sha256_sorted"]:
            raise RuntimeError(f"{dataset}: frozen {split} ids do not match their manifest hash")
    return manifest, frozen


def gate(name, passed, detail):
    return {"gate": name, "status": "PASS" if passed else "FAIL", "detail": detail}


def population(frozen, mode):
    key = "test" if mode == "full" else "coarse"
    return frozen[f"{key}_event_id"], frozen[f"{key}_label"].astype(np.int8), frozen[f"{key}_energy_keV"], frozen[f"{key}_group"].astype(str)


def main(args):
    manifest, frozen = load_frozen(args.dataset)
    expected = BENCHMARK.get("artefact_version", BENCHMARK["version"])
    if manifest.get("benchmark_version") != expected:
        raise RuntimeError(f"{args.dataset}: manifest carries artefact version {manifest.get('benchmark_version')}, "
                           f"code declares {expected}; regenerate the manifest or use the matching code")
    if args.mode == "coarse" and "coarse_event_id" not in frozen:
        raise RuntimeError(f"{args.dataset}: manifest has no coarse subset; regenerate it with freeze_splits.py")
    config = load_config(BENCHMARK["profile"])
    if fingerprint(config) != manifest["protocol_sha256"]:
        raise RuntimeError("evaluator profile differs from the one the manifest was frozen with")
    gates = []
    pred = np.load(args.predictions, allow_pickle=False)
    # G1 schema
    missing = {"event_id", "score"} - set(pred.files)
    ids = pred["event_id"].astype(str) if "event_id" in pred.files else np.array([], dtype=str)
    score = pred["score"].astype(np.float64) if "score" in pred.files else np.array([])
    unique = len(np.unique(ids)) == len(ids)
    finite = np.isfinite(score) if score.size else np.array([], dtype=bool)
    n_nonfinite = int((~finite).sum())
    # every score must be a finite number: a NaN/inf prediction makes every rank metric undefined (the model, not the
    # metric, is broken), so G1 fails and the run is reported as not valid instead of crashing the evaluator
    gates.append(gate("G1_schema", not missing and unique and ids.ndim == 1 and score.shape == ids.shape and n_nonfinite == 0,
                      {"missing_keys": sorted(missing), "unique_ids": bool(unique), "n": int(len(ids)),
                       "n_nonfinite_scores": n_nonfinite,
                       "score_min": float(score[finite].min()) if finite.any() else None,
                       "score_max": float(score[finite].max()) if finite.any() else None}))
    if missing or not unique or n_nonfinite:
        return finish(args, manifest, gates, None, None)
    # G2 exact identity of the evaluation population (test split, or the coarse validation subset)
    test_ids, pop_label, pop_energy, pop_group = population(frozen, args.mode)
    same_set = len(ids) == len(test_ids) and np.array_equal(np.sort(ids), np.sort(test_ids))
    gates.append(gate("G2_test_identity", same_set,
                      {"population": "test" if args.mode == "full" else "coarse_validation_subset",
                       "n_predictions": int(len(ids)), "n_frozen": int(len(test_ids)),
                       "n_missing": int(len(np.setdiff1d(test_ids, ids))), "n_unknown": int(len(np.setdiff1d(ids, test_ids))),
                       "canonical_order": bool(len(ids) == len(test_ids) and np.array_equal(ids, test_ids))}))
    if not same_set:
        return finish(args, manifest, gates, None, None)
    order = np.argsort(test_ids)
    pos = order[np.searchsorted(test_ids[order], ids)]  # position of each prediction in the frozen order
    label = pop_label[pos]
    energy = pop_energy[pos]
    group = pop_group[pos]
    # G3 labels/energies supplied by the model are not trusted but must agree
    if "label" in pred.files or "energy_keV" in pred.files:
        ok = True
        detail = {}
        if "label" in pred.files:
            ok &= bool(np.array_equal(pred["label"].astype(np.int8), label)); detail["labels_agree"] = bool(ok)
        if "energy_keV" in pred.files:
            e_ok = bool(np.allclose(pred["energy_keV"].astype(np.float64), energy, rtol=0, atol=1e-6)); detail["energies_agree"] = e_ok; ok &= e_ok
        gates.append(gate("G3_supplied_metadata_agrees", ok, detail))
    # G4 training contamination and label regime
    if args.no_training_ids:
        gates.append({"gate": "G4_no_test_contamination", "status": "SKIPPED", "detail": "no training id file supplied; result is NOT a benchmark result"})
    else:
        tr = np.load(args.training_ids, allow_pickle=False)
        seen = tr["event_id"].astype(str) if "event_id" in tr.files else np.array([], dtype=str)
        labeled = tr["labeled_event_id"].astype(str) if "labeled_event_id" in tr.files else np.array([], dtype=str)
        allowed = np.union1d(frozen["train_event_id"], frozen["validation_event_id"])
        real_test = frozen["test_event_id"]
        leak = np.intersect1d(seen, real_test)
        leak_l = np.intersect1d(labeled, real_test)
        foreign = np.setdiff1d(seen, allowed)
        detail = {"n_training_events": int(len(seen)), "n_labeled_events": int(len(labeled)),
                  "n_test_events_seen_in_training": int(leak.size), "n_unknown_training_ids": int(foreign.size),
                  "examples_leaked": leak[:5].tolist()}
        ok = leak.size == 0 and leak_l.size == 0 and foreign.size == 0
        if args.mode == "coarse":  # coarse events may have been seen unlabeled (pretraining); labels on them are a leak
            coarse_labeled = np.intersect1d(labeled, test_ids)
            detail["n_coarse_events_seen_unlabeled"] = int(np.intersect1d(seen, test_ids).size)
            detail["n_coarse_events_with_labels_used"] = int(coarse_labeled.size)
            ok &= coarse_labeled.size == 0
        gates.append(gate("G4a_no_test_contamination", ok, detail))
        lab_in_train = np.setdiff1d(labeled, frozen["train_event_id"])
        detail = {"n_labeled": int(len(labeled)), "n_labeled_outside_train_split": int(lab_in_train.size)}
        ok = lab_in_train.size == 0
        if args.label_fraction is None:
            gates.append({"gate": "G4b_label_regime", "status": "SKIPPED",
                          "detail": {**detail, "reason": "no --label-fraction declared; a benchmark number must belong to a frozen regime"}})
        else:
            exp = manifest["label_regimes"]["fractions"][str(args.label_fraction)]
            n_train = manifest["splits"]["train"]["n"]
            expected_ids = frozen["train_event_id"][labeled_subset_positions(n_train, args.label_fraction, manifest["label_regimes"]["seed"])]
            exact = bool(np.array_equal(np.sort(labeled), np.sort(expected_ids)))
            detail.update({"label_fraction": args.label_fraction, "n_expected": exp["n"], "labeled_set_equals_frozen_regime": exact,
                           "n_extra": int(np.setdiff1d(labeled, expected_ids).size), "n_missing": int(np.setdiff1d(expected_ids, labeled).size)})
            ok &= exact
            gates.append(gate("G4b_label_regime", ok, detail))
    # metrics: every metric declared for this dataset; the primary one decides reportability (G5)
    ctx = MetricContext(label=label, score=score, energy_keV=energy, group=group, profile=BENCHMARK["profile"])
    names = dataset_spec(args.dataset)["metrics"]
    primary = BENCHMARK["primary_metric"]
    if primary not in names:
        raise RuntimeError(f"{args.dataset}: primary metric {primary!r} is not declared for this dataset")
    metrics = compute_all(names, ctx)
    r = ctx.energybench
    gates.append(gate("G5_primary_metric_reportable", metrics[primary]["value"] is not None,
                      {"metric": primary, "status": metrics[primary]["status"], **metrics[primary]["detail"]}))
    # null and floor diagnostics (never gates on the model; they document the metric's behaviour here)
    rng = np.random.default_rng(NULL_SEED)
    null = compute_all([primary], MetricContext(label=label[rng.permutation(len(label))], score=score, energy_keV=energy,
                                                group=group, profile=BENCHMARK["profile"]))[primary]["value"]
    floor = manifest["energy_only_floor_test"] if args.mode == "full" else manifest["coarse"]["energy_only_floor"]
    diagnostics = {"label_shuffle_null_primary_metric": null, "energy_only_floor": floor}
    return finish(args, manifest, gates, r, diagnostics, metrics)


def _finite(x):
    """JSON-safe copy: non-finite floats become null (strict JSON has no NaN)."""
    if isinstance(x, dict):
        return {k: _finite(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_finite(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    return x


def finish(args, manifest, gates, record, diagnostics, metrics=None):
    failed = [g["gate"] for g in gates if g["status"] == "FAIL"]
    skipped = [g["gate"] for g in gates if g["status"] == "SKIPPED"]
    primary = BENCHMARK["primary_metric"]
    out = {"benchmark": BENCHMARK["name"], "benchmark_version": BENCHMARK["version"],
           "dataset": args.dataset, "mode": args.mode,
           "population": "test split (benchmark)" if args.mode == "full" else
           f"coarse subset of validation ({manifest['coarse']['fraction']:g} = {manifest['coarse']['n']} events); NOT a benchmark number",
           "profile": "strict600", "protocol_sha256": manifest["protocol_sha256"],
           "evaluated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "predictions": str(Path(args.predictions).resolve()), "predictions_sha256": sha256_file(args.predictions),
           "training_ids": None if args.no_training_ids else str(Path(args.training_ids).resolve()),
           "label_fraction": args.label_fraction, "seed": args.seed, "model": args.model,
           "primary_metric": primary,
           "primary_value": None if metrics is None else metrics[primary]["value"],
           "metrics": metrics,
           "matched_auc": None if record is None else record["matching"]["matched_auc"],
           "benchmark_valid": not failed and not skipped and metrics is not None and metrics[primary]["value"] is not None,
           "gates": gates, "failed_gates": failed, "skipped_gates": skipped,
           "diagnostics": diagnostics, "energybench": record}
    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(_finite(out), indent=2, allow_nan=False) + "\n")
    line = {"dataset": args.dataset, "mode": args.mode, primary: out["primary_value"], "benchmark_valid": out["benchmark_valid"],
            "failed_gates": failed, "skipped_gates": skipped, "output": str(out_dir / "metrics.json")}
    print(json.dumps(line))
    return 0 if out["benchmark_valid"] else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=DATASETS)
    ap.add_argument("--mode", default="full", choices=("full", "coarse"))
    ap.add_argument("--predictions", required=True, type=Path)
    ap.add_argument("--training-ids", type=Path)
    ap.add_argument("--no-training-ids", action="store_true", help="skip G4 (result is then NOT benchmark-valid)")
    ap.add_argument("--label-fraction", type=float, choices=tuple(BENCHMARK["label_regimes"]["fractions"]))
    ap.add_argument("--seed", type=int)
    ap.add_argument("--model", default=None, help="model name recorded in metrics.json")
    ap.add_argument("--output", required=True, type=Path)
    a = ap.parse_args()
    if not a.no_training_ids and a.training_ids is None:
        ap.error("--training-ids is required (or pass --no-training-ids for a non-benchmark evaluation)")
    sys.exit(main(a))
