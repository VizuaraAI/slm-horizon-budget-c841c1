# CLAUDE.md

Instructions for the coding agent working on this repo with Sai.

## What this project is

A research project, not a product. The paper is `../paper/main.tex`. The goal is
to answer one question well enough to publish: **given a target task horizon H
and a target end-to-end success rate T, what per-step competence must a small
language model actually reach?**

The naive answer is `p* = T^(1/H)`. The contribution is showing precisely how
that answer is wrong and by how much, decomposed into three measurable terms:

- `sigma` : step-difficulty heterogeneity. Raises the requirement (AM-GM).
- `delta` : self-conditioning, the accuracy drop after an error enters context.
  Raises the requirement.
- `r` : recovery, the chance a bad state gets repaired. Lowers the requirement.

**The sign of the net correction is not known in advance.** Do not write code,
comments or paper text that assumes the correction is positive. If a run comes
back saying the naive law was right for some model, that is a finding, and it
goes in the paper as-is.

## Layout

| path | what it holds |
| --- | --- |
| `src/horizon_model.py` | the math: forward law `S(H)`, inversion by bisection |
| `src/estimate.py` | fits `p, delta, r, sigma` from logged traces |
| `src/tasks.py` | the three task families |
| `src/harness.py` | the episode loop |
| `src/solvers.py` | `echo` simulator, `hf` real model |
| `src/interventions.py` | voting, verifier gate, and their forward-pass cost |
| `tests/` | guardrails for the math |
| `experiments/log_template.md` | one row per run |

## Conventions

- **Config over flags.** Everything lives in `configs/default.yaml`. Override
  with dotted args (`--model.name X`, `--task.family fsm`). Do not add bare
  argparse flags to `evaluate.py`.
- **Seeded and reproducible.** Every episode is generated from a seed derived
  from `(seed, horizon, episode_index)`. A run must be reproducible from its
  config alone. Never introduce unseeded randomness.
- **Measure, do not assume.** If a quantity can be read off the traces, read it
  off the traces. The reason `estimate.py` debiases the heterogeneity estimate
  for binomial sampling noise is that an earlier version read pure noise as
  heterogeneity and invented an inversion gap out of nothing. That class of bug
  is the one that will sink this paper. Watch for it.
- **Never repair the model's state.** The harness feeds wrong answers forward on
  purpose. If you find yourself writing a fallback that substitutes the
  ground-truth value when the model has drifted, stop: that silently
  manufactures a recovery rate. There is a comment in `harness.py` marking the
  exact spot.
- **Cost is part of every intervention.** `Intervention.run` returns
  `(answer, n_forward_passes)`. An intervention that cannot report its cost
  cannot go in the ledger.
- **Keep `src/` readable in one sitting.** If a module passes roughly 250 lines,
  that is a signal to split it, not to keep appending.
- Style: type hints on public functions, `from __future__ import annotations`,
  no em dashes in comments or docstrings, comments explain *why* not *what*.

## Before you say something works

```bash
pytest -q                                                    # the math
python -m src.evaluate --config configs/default.yaml --io.run_name smoke   # the pipeline
```

The echo backend has a *known* `echo_step_accuracy` and `echo_contagion`. After
any change to the harness or estimator, check that the fitted `p_hat` comes back
close to the configured value. If it does not, the estimator is wrong, not the
simulator.

## Where the project is going

1. Validate the estimator against the simulator, where ground truth is known.
2. Measure the real models: nine open SLMs from 0.5B to 8B, three task families,
   horizons 1 to 64, at least 200 episodes per cell.
3. Build the intervention ledger: voting, verifier gate, decomposition, context
   hygiene, each priced in extra reachable horizon per extra forward pass.
4. Transfer check: does a requirement curve fitted on the controlled tasks
   predict behaviour on horizon-stratified tau-bench and GAIA slices?
5. Write up. Results table in the paper maps row-for-row onto
   `experiments/log_template.md`.

## Things to push back on

If Sai asks for a change that would make a number look better without making it
more correct (dropping a model that behaves badly, picking the horizon grid
after seeing results, quietly switching the success criterion), say so. The
whole value of this paper is that the measurement is trustworthy.
