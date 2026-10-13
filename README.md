# slm-horizon-budget

Inverting the compounding-error arithmetic for small language model agents.

Everybody quotes the forward direction: chain eight components at 90% per-step
competence and you get a 43% end-to-end ceiling. Nobody runs it backwards. This
repo runs it backwards. You state the task horizon `H` you need and the
end-to-end success rate `T` you can live with, and it tells you the per-step
competence `p*` your model has to reach, how far your model actually is from
that, and which intervention closes the gap most cheaply.

The naive inversion is `p* = T^(1/H)`. That answer is wrong in three ways, and
measuring the three is the research:

| term | what it is | direction |
| --- | --- | --- |
| heterogeneity `sigma` | steps are not equally hard, so a mean accuracy overstates the product (AM-GM) | naive is too lenient |
| self-conditioning `delta` | once an error is in context, the next step gets harder | naive is too lenient |
| recovery `r` | some bad intermediate states get repaired and the task still finishes | naive is too harsh |

Which term dominates for your model, your task family and your horizon is an
empirical question. That is the paper.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Torch and transformers are only needed for the `hf` backend. The whole pipeline
runs without a GPU on the `echo` backend.

## The first three things to run

**1. Check the math, in under a second.**

```bash
pytest -q
```

Nine tests. They pin down the properties that everything else depends on: the
model reproduces `p^H` exactly in the independent corner case, the inversion
round-trips, heterogeneity lowers success, and recovery raises it. If you change
`src/horizon_model.py`, these are your guardrail.

**2. Run the full pipeline on the simulator.**

```bash
python -m src.evaluate --config configs/default.yaml --io.run_name smoke
```

The `echo` backend is a step oracle with a known per-step accuracy and a known
contagion strength. It never looks at the prompt. Use it to check that the
harness, the estimator and the inversion all work, and to confirm the estimator
recovers parameters you already know. Takes seconds. Writes
`results/smoke/traces.jsonl` and `results/smoke/summary.json`.

**3. Run a real small model.**

```bash
python -m src.evaluate \
  --config configs/default.yaml \
  --model.backend hf --model.name Qwen/Qwen2.5-1.5B-Instruct \
  --task.family chain_kv --io.run_name qwen15_kv
```

Start with `--task.horizons '[1,2,4,8]'` and
`--task.episodes_per_horizon 16` to confirm the model answers in the expected
format before you spend hours on the full sweep.

## What comes next

```bash
# sweep every model and task family
bash scripts/run_sweep.sh

# train the step-level verifier on traces you already logged
python -m src.train_verifier --traces results/qwen15_kv/traces.jsonl \
  --out results/qwen15_kv/verifier.npz

# re-run with an intervention and compare the reachable horizon
python -m src.evaluate --intervention.kind vote --io.run_name qwen15_kv_vote
python -m src.evaluate --intervention.kind verifier \
  --intervention.verifier_path results/qwen15_kv/verifier.npz \
  --io.run_name qwen15_kv_verifier

# rebuild the paper figures from your own results
python -m src.plotting --summaries results/*/summary.json --out figures/
```

## Layout

```
configs/default.yaml       every knob, with comments
src/tasks.py               ChainKV, ChainArith, FsmWalk: horizon-tunable, per-step ground truth
src/solvers.py             echo simulator and the HuggingFace backend
src/harness.py             runs episodes, feeds errors forward, logs per-step correctness
src/horizon_model.py       the forward law and the inversion
src/estimate.py            fits p, delta, r, sigma from traces, with bootstrap CIs
src/interventions.py       voting, verifier gate, and their compute cost
src/train_verifier.py      trains the step-level verifier
src/evaluate.py            CLI entry point
src/plotting.py            paper figures from summary.json
experiments/log_template.md one row per run, mapping onto the paper's Table 1
tests/                     guardrails for the math
```

No datasets are committed. The task families are generated from a seed, so a
run is reproducible from its config alone, and the external-validity checks
(tau-bench, GAIA) download from their own public sources.

## The one design decision worth knowing

When the model gets a step wrong, the harness does **not** stop and does **not**
repair the state. It feeds the model its own wrong answer and keeps going. That
is the only way to measure `delta` (how much worse the model gets after it has
erred) and `r` (how often it climbs back out). An evaluation that halts at the
first error can measure neither, which is why most reported per-step accuracies
cannot be inverted.
