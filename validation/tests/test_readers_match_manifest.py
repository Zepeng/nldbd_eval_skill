"""Check that the reader on THIS machine reproduces the frozen manifest of a dataset.

Usage: python validation/tests/test_readers_match_manifest.py <dataset> [--all-splits]
Without --all-splits only the test split is read (fast); with it, train and validation too.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from validation.readers import get_reader  # noqa: E402
from validation.readers.common import MANIFESTS  # noqa: E402


def main(dataset, all_splits=False):
    manifest = json.loads((MANIFESTS / f"{dataset}.json").read_text())
    reader = get_reader(dataset)
    ok = True
    for split in (("train", "validation", "test") if all_splits else ("test",)):
        t0 = time.time()
        t = reader.table(split)
        s = reader.summary(split, t)
        ref = manifest["splits"][split]
        checks = {k: s[k] == ref[k] for k in ("n", "label_counts", "ids_sha256_sorted", "ids_sha256_canonical_order", "labels_sha256_canonical_order")}
        checks["energy_sha256"] = s["energy_keV"]["sha256_repr"] == ref["energy_keV"]["sha256_repr"]
        ok &= all(checks.values())
        print(f"{dataset} {split}: {s['n']} events in {time.time() - t0:.1f} s ->", {k: ('OK' if v else 'BAD') for k, v in checks.items()})
    print("RESULT", dataset, "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    sys.exit(0 if main(args[0], "--all-splits" in sys.argv) else 1)
