#!/usr/bin/env bash
# Build the intervention ledger for one model: baseline, then each intervention,
# each priced in extra forward passes per step.
#
#   bash scripts/run_interventions.sh Qwen/Qwen2.5-1.5B-Instruct
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL="${1:-Qwen/Qwen2.5-1.5B-Instruct}"
FAMILY="${FAMILY:-chain_kv}"
EPISODES="${EPISODES:-200}"
RESULTS="${RESULTS:-results}"
SHORT="$(basename "$MODEL" | tr '[:upper:]' '[:lower:]' | tr '.' '-')"

common=(
  --config configs/default.yaml
  --model.backend hf
  --model.name "$MODEL"
  --task.family "$FAMILY"
  --task.episodes_per_horizon "$EPISODES"
  --io.results_dir "$RESULTS"
)

echo "[ledger] baseline"
python -m src.evaluate "${common[@]}" \
  --intervention.kind none --io.run_name "${SHORT}__baseline"

echo "[ledger] training the step-level verifier on the baseline traces"
python -m src.train_verifier \
  --traces "${RESULTS}/${SHORT}__baseline/traces.jsonl" \
  --out "${RESULTS}/${SHORT}__baseline/verifier.npz"

echo "[ledger] k-sample voting"
python -m src.evaluate "${common[@]}" \
  --intervention.kind vote --intervention.vote_k 3 \
  --io.run_name "${SHORT}__vote3"

echo "[ledger] verifier gate"
python -m src.evaluate "${common[@]}" \
  --intervention.kind verifier \
  --intervention.verifier_path "${RESULTS}/${SHORT}__baseline/verifier.npz" \
  --io.run_name "${SHORT}__verifier"

echo
echo "[ledger] done. Compare h50_observed across the three summary.json files"
echo "[ledger] and divide the horizon gained by the extra forward passes per"
echo "[ledger] step. That ratio is the ledger column in the paper."
