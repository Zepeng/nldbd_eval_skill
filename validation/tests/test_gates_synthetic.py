"""Exercise evaluate_predictions on synthetic scores built from a frozen manifest (no raw data).

Checks: perfect scores -> AUC_w 1; score = energy -> AUC_w ~ 0.5 (the shortcut floor); a test id in
the training ids -> G4a fails; a wrong labeled set -> G4b fails; missing ids -> G2 fails.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from validation.freeze_splits import labeled_subset_positions  # noqa: E402


def run(dataset, pred, train, extra, out):
    cmd = [sys.executable, "-B", "-m", "validation.evaluate_predictions", "--dataset", dataset, "--predictions", str(pred),
           "--output", str(out)] + (["--training-ids", str(train)] if train else ["--no-training-ids"]) + extra
    rc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True).returncode
    m = json.loads((out / "metrics.json").read_text())
    return rc, m


def main():
    manifests = sorted(p.stem for p in (ROOT / "validation" / "manifests").glob("*.json"))
    dataset = "exo200" if "exo200" in manifests else manifests[0]
    ids = np.load(ROOT / "validation" / "manifests" / f"{dataset}_ids.npz", allow_pickle=False)
    te, lab, e = ids["test_event_id"], ids["test_label"], ids["test_energy_keV"]
    tr, va = ids["train_event_id"], ids["validation_event_id"]
    pos = labeled_subset_positions(len(tr), 0.01)
    ok = True
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        np.savez(d / "good_train.npz", event_id=np.concatenate([tr[pos], va]), labeled_event_id=tr[pos])
        np.savez(d / "leak_train.npz", event_id=np.concatenate([tr[pos], te[:3]]), labeled_event_id=tr[pos])
        np.savez(d / "wrong_regime.npz", event_id=tr[pos[:-1]], labeled_event_id=tr[pos[:-1]])
        rng = np.random.default_rng(1)
        np.savez(d / "perfect.npz", event_id=te, score=lab + 0.1 * rng.random(len(te)))
        np.savez(d / "energy.npz", event_id=te, score=e)
        np.savez(d / "missing.npz", event_id=te[:-5], score=lab[:-5].astype(float))
        rc, m = run(dataset, d / "perfect.npz", d / "good_train.npz", ["--label-fraction", "0.01"], d / "o1")
        print("perfect scores:", rc, m["matched_auc"], m["failed_gates"]); ok &= rc == 0 and abs(m["matched_auc"] - 1) < 1e-9
        rc, m = run(dataset, d / "energy.npz", d / "good_train.npz", ["--label-fraction", "0.01"], d / "o2")
        print("score = energy:", rc, round(m["matched_auc"], 4), "inclusive", round(m["metrics"]["inclusive_auc"]["value"], 4))
        ok &= rc == 0 and abs(m["matched_auc"] - 0.5) < 0.01
        rc, m = run(dataset, d / "perfect.npz", d / "leak_train.npz", ["--label-fraction", "0.01"], d / "o3")
        print("leaked test id:", rc, m["failed_gates"]); ok &= rc == 1 and "G4a_no_test_contamination" in m["failed_gates"]
        rc, m = run(dataset, d / "perfect.npz", d / "wrong_regime.npz", ["--label-fraction", "0.01"], d / "o4")
        print("wrong regime:", rc, m["failed_gates"]); ok &= rc == 1 and "G4b_label_regime" in m["failed_gates"]
        rc, m = run(dataset, d / "missing.npz", d / "good_train.npz", [], d / "o5")
        print("missing ids:", rc, m["failed_gates"]); ok &= rc == 1 and "G2_test_identity" in m["failed_gates"]
    print("RESULT", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
