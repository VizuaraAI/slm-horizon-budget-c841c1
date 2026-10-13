#!/usr/bin/env bash
# The main sweep: every model, every task family, baseline (no intervention).
#
#   bash scripts/run_sweep.sh                 # full sweep
#   EPISODES=16 HORIZONS='[1,2,4,8]' bash scripts/run_sweep.sh   # quick pilot
#
# Runs are skipped if their summary.json already exists, so this is safe to
# re-run after an interruption. Delete the run directory to force a redo.
set -euo pipefail

cd "$(dirname "$0")/.."

EPISODES="${EPISODES:-200}"
HORIZONS="${HORIZONS:-[1,2,4,8,16,32,64]}"
RESULTS="${RESULTS:-results}"

MODELS=(
  "Qwen/Qwen2.5-0.5B-Instruct"
  "meta-llama/Llama-3.2-1B-Instruct"
  "Qwen/Qwen2.5-1.5B-Instruct"
  "HuggingFaceTB/SmolLM2-1.7B-Instruct"
  "google/gemma-2-2b-it"
  "Qwen/Qwen2.5-3B-Instruct"
  "meta-llama/Llama-3.2-3B-Instruct"
  "microsoft/Phi-3-mini-4k-instruct"
  "Qwen/Qwen2.5-7B-Instruct"
)

FAMILIES=(chain_kv chain_arith fsm)

for model in "${MODELS[@]}"; do
  short="$(basename "$model" | tr '[:upper:]' '[:lower:]' | tr '.' '-')"
  for family in "${FAMILIES[@]}"; do
    run="${short}__${family}"
    if [ -f "${RESULTS}/${run}/summary.json" ]; then
      echo "[sweep] skip ${run} (already done)"
      continue
    fi
    echo "[sweep] ${run}"
    python -m src.evaluate \
      --config configs/default.yaml \
      --model.backend hf \
      --model.name "$model" \
      --task.family "$family" \
      --task.horizons "$HORIZONS" \
      --task.episodes_per_horizon "$EPISODES" \
      --io.results_dir "$RESULTS" \
      --io.run_name "$run"
  done
done

echo
echo "[sweep] rebuilding figures"
python -m src.plotting --summaries "${RESULTS}"/*/summary.json --out figures/
echo "[sweep] done. Copy the numbers into experiments/log_template.md."
