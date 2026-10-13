"""The forward horizon law and, more importantly, its inverse.

The quoted arithmetic
---------------------
Chain H dependent steps, each correct with probability p, assume independence
and no recovery:

    S(H) = p ** H                      e.g. 0.9 ** 8 = 0.43

Inverting that gives the naive competence requirement:

    p_naive(H, T) = T ** (1 / H)       e.g. T=0.5, H=32  ->  0.9786

This module exists because ``p_naive`` is wrong in three separate ways, and the
three do not all point the same direction:

1. **Heterogeneity.** Steps are not equally hard. For a fixed difficulty
   profile p_1..p_H with mean p, the product of the p_i is at most p ** H
   (AM-GM). Plugging a measured *mean* per-step accuracy into the naive law is
   therefore provably optimistic. The gap grows with the spread.
2. **Self-conditioning.** Per-step accuracy is not stationary: once an error is
   in the model's own context, the next step gets harder. We model this with a
   degraded state whose per-step accuracy is ``p - delta``. Also optimistic.
3. **Recovery.** Real agents sometimes repair a bad state and still finish
   correctly, which the all-steps-must-be-correct reading forbids. This one
   makes the naive law *pessimistic*.

The net sign is an empirical question, which is the point of the project. This
module just makes both directions computable and invertible.

Model
-----
A two-state Markov chain over the trajectory, state in {CLEAN, DEGRADED},
starting CLEAN, with step-dependent competence p_i:

    CLEAN     -> CLEAN      p_i
    CLEAN     -> DEGRADED   1 - p_i
    DEGRADED  -> CLEAN      recovery * max(p_i - delta, 0)
    DEGRADED  -> DEGRADED   1 - recovery * max(p_i - delta, 0)

S(H) is the CLEAN mass after H steps. Setting ``recovery = 0`` and
``het = 0`` recovers S(H) = p ** H exactly, so the naive law is the corner case.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

EPS = 1e-12


def logit(p: float) -> float:
    p = min(max(p, EPS), 1.0 - EPS)
    return math.log(p / (1.0 - p))


def sigmoid(x: float | np.ndarray):
    return 1.0 / (1.0 + np.exp(-x))


def naive_required_competence(horizon: int, target: float) -> float:
    """p* = T ** (1/H): the inversion everybody would write down first."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    return float(target ** (1.0 / horizon))


def naive_horizon(p: float, target: float) -> float:
    """Largest H with p ** H >= T, as a real number (not rounded down)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    return math.log(target) / math.log(p)


def difficulty_profile(horizon: int, p: float, het: float) -> np.ndarray:
    """A deterministic, reproducible per-step difficulty profile with mean ~p.

    Steps are spread on the logit scale using evenly spaced normal quantiles,
    then shifted so the arithmetic mean of the profile is exactly ``p``. The
    profile is fixed rather than resampled per episode, which is what makes the
    AM-GM gap real: step 7 of a task family is intrinsically harder than step 3,
    and averaging over episodes does not wash that out.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if het <= 0.0:
        return np.full(horizon, float(p))

    from scipy.stats import norm

    quantiles = (np.arange(horizon) + 0.5) / horizon
    z = norm.ppf(quantiles)
    profile = sigmoid(logit(p) + het * z)

    # Shift on the logit scale until the arithmetic mean matches p.
    lo, hi = -20.0, 20.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        candidate = sigmoid(logit(p) + het * z + mid)
        if candidate.mean() < p:
            lo = mid
        else:
            hi = mid
    profile = sigmoid(logit(p) + het * z + 0.5 * (lo + hi))
    return np.clip(profile, EPS, 1.0 - EPS)


@dataclass
class HorizonModel:
    """Forward law and inversion under heterogeneity, degradation and recovery."""

    p: float = 0.95
    delta: float = 0.0
    recovery: float = 0.0
    het: float = 0.0

    def success(self, horizon: int, p: float | None = None) -> float:
        """End-to-end success probability at the given horizon."""
        p_eff = self.p if p is None else p
        profile = difficulty_profile(horizon, p_eff, self.het)

        clean, degraded = 1.0, 0.0
        for p_i in profile:
            p_deg = max(float(p_i) - self.delta, 0.0)
            back = self.recovery * p_deg
            clean, degraded = (
                clean * float(p_i) + degraded * back,
                clean * (1.0 - float(p_i)) + degraded * (1.0 - back),
            )
        return float(clean)

    def required_competence(self, horizon: int, target: float, tol: float = 1e-10) -> float:
        """Invert the forward law: the mean per-step competence that hits T.

        ``success`` is monotone increasing in p, so plain bisection is both
        correct and fast. Returns ``nan`` when the target is unreachable even at
        p -> 1 (possible when delta is large and recovery is zero).
        """
        if not 0.0 < target < 1.0:
            raise ValueError("target must be in (0, 1)")
        lo, hi = EPS, 1.0 - 1e-9
        if self.success(horizon, hi) < target:
            return float("nan")
        if self.success(horizon, lo) > target:
            return float(lo)
        while hi - lo > tol:
            mid = 0.5 * (lo + hi)
            if self.success(horizon, mid) < target:
                lo = mid
            else:
                hi = mid
        return float(0.5 * (lo + hi))

    def reachable_horizon(self, target: float, max_horizon: int = 4096) -> int:
        """Largest H whose end-to-end success is still at least T."""
        best = 0
        for horizon in range(1, max_horizon + 1):
            if self.success(horizon) >= target:
                best = horizon
            else:
                break
        return best

    def inversion_gap(self, horizon: int, target: float) -> float:
        """Required competence under this model minus the naive requirement.

        Positive means the naive arithmetic asks for too little, so a team that
        sized its model with T ** (1/H) will miss its target.
        """
        return self.required_competence(horizon, target) - naive_required_competence(
            horizon, target
        )


def requirement_table(
    model: HorizonModel, horizons: list[int], targets: list[float]
) -> list[dict[str, float]]:
    """Rows of (H, T, naive p*, model p*, gap) ready to print or write to CSV."""
    rows: list[dict[str, float]] = []
    for target in targets:
        for horizon in horizons:
            naive = naive_required_competence(horizon, target)
            corrected = model.required_competence(horizon, target)
            rows.append(
                {
                    "horizon": float(horizon),
                    "target": float(target),
                    "p_star_naive": naive,
                    "p_star_model": corrected,
                    "gap_pp": 100.0 * (corrected - naive),
                }
            )
    return rows
