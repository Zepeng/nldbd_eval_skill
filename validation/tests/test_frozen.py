"""Fail if the frozen evaluator, its profiles, or the frozen manifests were modified."""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ok = True
    for line in (HERE / "energybench" / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split()
        got = sha(HERE / "energybench" / name)
        print(f"{'OK ' if got == digest else 'BAD'} energybench/{name}")
        ok &= got == digest
    bench = json.loads((HERE / "benchmark.json").read_text())
    print(f"benchmark {bench['name']} v{bench['version']} (artefacts v{bench.get('artefact_version', bench['version'])}), primary metric {bench['primary_metric']}, datasets {list(bench['datasets'])}")
    recipe = HERE / "finetune_recipe.json"
    print(f"{'OK ' if recipe.exists() else 'BAD'} finetune_recipe.json present"); ok &= recipe.exists()
    for name in bench["datasets"]:
        m = HERE / "manifests" / f"{name}.json"
        if not m.exists():
            print(f"BAD manifests/{name}.json missing"); ok = False; continue
        d = json.loads(m.read_text())
        npz = HERE / "manifests" / d["ids_npz"]
        same = d.get("benchmark_version") == bench.get("artefact_version", bench["version"]) and d["profile"] == bench["profile"]
        print(f"{'OK ' if npz.exists() and same else 'BAD'} manifests/{name}.json: v{d.get('benchmark_version')}, "
              f"{d['splits']['test']['n']} test events, coarse {d['coarse']['n']}, protocol {d['protocol_sha256'][:12]}")
        ok &= npz.exists() and same
    from validation.metrics import REGISTRY  # noqa: E402
    for name in [bench["primary_metric"], *bench["secondary_metrics"]]:
        print(f"{'OK ' if name in REGISTRY else 'BAD'} metric {name} registered")
        ok &= name in REGISTRY
    print("RESULT", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
