#!/usr/bin/env bash
# Create a minimal environment for the harness next to this folder (./.venv) with uv or venv.
# Usage: bash validation/setup_env.sh [--with-torch]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"
if command -v uv >/dev/null 2>&1; then
  uv venv --python 3.11 .venv >/dev/null
  uv pip install --python .venv/bin/python -r validation/requirements.txt
  if [[ "${1:-}" == "--with-torch" ]]; then uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cu128; fi
else
  python3 -m venv .venv
  .venv/bin/python -m pip install -q -r validation/requirements.txt
  if [[ "${1:-}" == "--with-torch" ]]; then .venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cu128; fi
fi
echo "environment ready: $HERE/.venv/bin/python"
