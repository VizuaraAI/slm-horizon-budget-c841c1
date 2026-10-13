"""Interventions that raise effective per-step competence at a compute cost.

Each intervention wraps a single step and returns (answer, n_forward_passes).
The forward-pass count is the price: the intervention ledger in the paper is
"extra reachable horizon bought per extra forward pass per step", so an
intervention that cannot report its cost cannot be scored.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .tasks import Episode, Step, normalize


class Intervention(Protocol):
    name: str

    def run(self, solver, episode: Episode, step: Step, state_in: str,
            degraded: bool) -> tuple[str, int]:
        ...


@dataclass
class NoIntervention:
    temperature: float = 0.0
    name: str = "none"

    def run(self, solver, episode: Episode, step: Step, state_in: str,
            degraded: bool) -> tuple[str, int]:
        answer = solver.solve(episode, step, state_in, degraded, 1, self.temperature)[0]
        return answer, 1


@dataclass
class VotingIntervention:
    """First-to-ahead-by-k voting over independent samples of the same step.

    Sample until one answer leads the runner-up by k votes, or until the sample
    budget runs out, then take the plurality. This is the per-step error
    correction used by massively decomposed agentic processes; it only helps to
    the extent that the samples are genuinely independent, which is why the
    temperature matters and why correlated failure modes defeat it.
    """

    k: int = 2
    max_samples: int = 9
    temperature: float = 0.7
    name: str = "vote"

    def run(self, solver, episode: Episode, step: Step, state_in: str,
            degraded: bool) -> tuple[str, int]:
        tally: Counter[str] = Counter()
        raw: dict[str, str] = {}
        calls = 0
        while calls < self.max_samples:
            answer = solver.solve(episode, step, state_in, degraded, 1, self.temperature)[0]
            calls += 1
            key = normalize(answer)
            raw.setdefault(key, answer)
            tally[key] += 1
            ranked = tally.most_common(2)
            lead = ranked[0][1] - (ranked[1][1] if len(ranked) > 1 else 0)
            if lead >= self.k:
                break
        winner = tally.most_common(1)[0][0]
        return raw[winner], calls


@dataclass
class VerifierGate:
    """A step-level verifier that accepts, retries, or gives up.

    The verifier is a small logistic model over cheap features of the step (see
    train_verifier.py). It is deliberately not the generator judging itself:
    intrinsic self-correction without an external signal is unreliable, so the
    verifier is trained on logged traces with ground-truth step labels.
    """

    weights: np.ndarray | None = None
    threshold: float = 0.5
    max_retries: int = 2
    temperature: float = 0.7
    name: str = "verifier"

    def score(self, features: np.ndarray) -> float:
        if self.weights is None:
            # With no trained verifier, accept everything: the gate degrades to
            # the baseline rather than silently inventing a signal.
            return 1.0
        z = float(np.dot(self.weights, features))
        return float(1.0 / (1.0 + np.exp(-z)))

    def run(self, solver, episode: Episode, step: Step, state_in: str,
            degraded: bool) -> tuple[str, int]:
        best_answer, best_score, calls = "", -1.0, 0
        for attempt in range(self.max_retries + 1):
            temperature = 0.0 if attempt == 0 else self.temperature
            answer = solver.solve(episode, step, state_in, degraded, 1, temperature)[0]
            calls += 1
            confidence = self.score(step_features(episode, step, answer, attempt))
            if confidence > best_score:
                best_answer, best_score = answer, confidence
            if confidence >= self.threshold:
                break
        return best_answer, calls


def step_features(episode: Episode, step: Step, answer: str, attempt: int = 0) -> np.ndarray:
    """Cheap, model-agnostic features for the step-level verifier.

    Nothing here peeks at the ground truth. These are the signals a deployed
    agent could actually compute at run time.
    """
    token = normalize(answer)
    horizon = max(episode.horizon, 1)
    return np.array(
        [
            1.0,                                      # bias
            step.index / horizon,                     # relative depth in trajectory
            min(len(answer), 200) / 200.0,            # verbosity of the raw reply
            1.0 if token else 0.0,                    # produced anything parseable
            1.0 if len(answer.split()) <= 3 else 0.0, # answered tersely, as instructed
            1.0 if token.isdigit() else 0.0,          # shape matches a numeric family
            attempt / 3.0,                            # how many retries so far
        ],
        dtype=np.float64,
    )


N_FEATURES = 7


def build_intervention(cfg: dict) -> Intervention:
    kind = str(cfg.get("kind", "none")).lower()
    if kind == "none":
        return NoIntervention()
    if kind == "vote":
        return VotingIntervention(
            k=int(cfg.get("vote_k", 2)),
            max_samples=int(cfg.get("vote_max_samples", 9)),
            temperature=float(cfg.get("vote_temperature", 0.7)),
        )
    if kind == "verifier":
        weights = None
        path = cfg.get("verifier_path")
        if path:
            weights = np.load(path)["weights"]
        return VerifierGate(
            weights=weights,
            threshold=float(cfg.get("verifier_threshold", 0.5)),
            max_retries=int(cfg.get("verifier_max_retries", 2)),
        )
    raise ValueError(f"unknown intervention '{kind}' (expected none, vote or verifier)")
