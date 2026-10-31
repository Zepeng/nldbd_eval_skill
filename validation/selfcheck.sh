#!/usr/bin/env bash
# Prove that this copy of the harness is intact and (if data is reachable) that the readers
# reproduce the frozen splits. Usage: bash validation/selfcheck.sh [python] [dataset ...]
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"
PY="${1:-${PYTHON:-python3}}"; shift || true
echo "== 1. frozen evaluator and manifests unchanged"
"$PY" -B validation/tests/test_frozen.py || exit 1
echo "== 2. evaluator unit tests (from the paper release)"
( cd validation/energybench && "$PY" -B tests/test_metrics.py 2>&1 | tail -2 ) || exit 1
echo "== 3. gates on synthetic predictions"
"$PY" -B validation/tests/test_gates_synthetic.py || exit 1
for d in "$@"; do
  echo "== 4. reader reproduces the frozen $d splits (needs the data)"
  "$PY" -B validation/tests/test_readers_match_manifest.py "$d" || exit 1
done
echo "SELFCHECK PASS"
