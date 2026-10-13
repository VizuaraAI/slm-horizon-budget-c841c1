"""Run episodes step by step and log exactly what the inversion needs.

The one design decision that matters: when the model gets a step wrong we do
NOT stop and we do NOT silently repair the state. We feed its own wrong answer
forward. That is the only way to measure two quantities the naive arithmetic
assumes away:

* ``delta``    how much per-step accuracy drops once an error is in context
* ``recovery`` how often a drifted trajectory snaps back to the true chain

A step is scored against the transition from the state the model is *actually*
in, not from the true state, so "correct step on a wrong trajectory" is
distinguishable from "recovered".
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .interventions import Intervention
from .tasks import Episode, make_episode, normalize


@dataclass
class StepRecord:
    index: int
    horizon: int
    family: str
    episode_id: int
    # Was the prefix already off the true chain when this step began?
    degraded_in: bool
    # Did the answer match the transition from the state the model was in?
    step_correct: bool
    # Is the model back on the true chain after this step?
    on_true_chain: bool
    recovered: bool
    n_calls: int
    answer: str
    expected: str


@dataclass
class EpisodeRecord:
    episode_id: int
    family: str
    horizon: int
    success: bool
    n_calls: int
    steps: list[StepRecord] = field(default_factory=list)


def run_episode(
    episode: Episode,
    solver,
    intervention: Intervention,
    episode_id: int = 0,
) -> EpisodeRecord:
    if hasattr(solver, "reset"):
        solver.reset()

    record = EpisodeRecord(episode_id, episode.family, episode.horizon, False, 0)
    state = episode.steps[0].state_in
    degraded = False

    for step in episode.steps:
        # The transition from the state the model is ACTUALLY in. None means
        # the trajectory has drifted off the task's defined state space, so no
        # answer can be correct. Falling back to the true-chain answer here
        # would hand the model a free recovery it never earned.
        expected_here = episode.transition(step.index, state)
        answer, calls = intervention.run(solver, episode, step, state, degraded)
        record.n_calls += calls

        if hasattr(solver, "record"):
            solver.record(step.instruction, answer)

        step_correct = expected_here is not None and normalize(answer) == normalize(expected_here)
        on_true_chain = normalize(answer) == normalize(step.answer)
        recovered = degraded and on_true_chain

        record.steps.append(
            StepRecord(
                index=step.index,
                horizon=episode.horizon,
                family=episode.family,
                episode_id=episode_id,
                degraded_in=degraded,
                step_correct=step_correct,
                on_true_chain=on_true_chain,
                recovered=recovered,
                n_calls=calls,
                answer=answer,
                expected=expected_here if expected_here is not None else "",
            )
        )

        state = answer
        degraded = not on_true_chain

    record.success = normalize(state) == normalize(episode.final_answer)
    return record


def run_sweep(cfg: dict, solver, intervention: Intervention) -> list[EpisodeRecord]:
    """Run every (horizon, episode) cell for one model and one intervention."""
    task_cfg = dict(cfg["task"])
    family = task_cfg.pop("family")
    horizons = task_cfg.pop("horizons")
    episodes_per_horizon = int(task_cfg.pop("episodes_per_horizon"))
    base_seed = int(cfg.get("seed", 0))

    records: list[EpisodeRecord] = []
    episode_id = 0
    for horizon in horizons:
        for n in range(episodes_per_horizon):
            rng = random.Random(base_seed * 1_000_003 + horizon * 9973 + n)
            episode = make_episode(family, rng, int(horizon), **task_cfg)
            records.append(run_episode(episode, solver, intervention, episode_id))
            episode_id += 1
    return records


def write_records(records: list[EpisodeRecord], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(asdict(record)) + "\n")
    return path


def read_records(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
