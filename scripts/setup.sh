#!/usr/bin/env bash
# One-time setup. Creates a venv, installs deps, and proves the pipeline works.
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"

if [ ! -d .venv ]; then
  echo "[setup] creating .venv"
  "$PYTHON" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "[setup] installing requirements"
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

echo "[setup] running the math guardrails"
pytest -q

echo "[setup] running the pipeline on the echo backend"
python -m src.evaluate --config configs/default.yaml --io.run_name smoke \
  --estimate.bootstrap_resamples 200

echo
echo "[setup] done. Results in results/smoke/."
echo "[setup] Next: run a real model, starting small:"
echo "  python -m src.evaluate --model.backend hf \\"
echo "    --model.name Qwen/Qwen2.5-1.5B-Instruct \\"
echo "    --task.horizons '[1,2,4,8]' --task.episodes_per_horizon 16 \\"
echo "    --io.run_name qwen15_pilot"
