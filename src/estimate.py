"""Fit the horizon model from logged traces, then invert it.

Estimated quantities, all read straight off the per-step records:

    p_hat        per-step accuracy on a clean prefix
    p_deg        per-step accuracy on a degraded prefix
    delta        p_hat - p_deg, the self-conditioning penalty
    recovery     P(back on the true chain | prefix was degraded)
    het          logit-scale spread of per-step accuracy across step positions

The last one is the term that makes a mean per-step accuracy an optimistic
input to the naive law, so it is worth estimating even though it is the one
most people skip.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from .horizon_model import EPS, HorizonModel, logit, naive_required_competence


@dataclass
class HorizonFit:
    p_hat: float
    p_deg: float
    delta: float
    recovery: float
    het: float
    n_clean_steps: int
    n_degraded_steps: int

    def to_model(self) -> HorizonModel:
        return HorizonModel(p=self.p_hat, delta=self.delta, recovery=self.recovery, het=self.het)


def _flat_steps(records: list[dict]) -> list[dict]:
    return [step for record in records for step in record["steps"]]


def fit(records: list[dict]) -> HorizonFit:
    steps = _flat_steps(records)
    if not steps:
        raise ValueError("no steps to fit: the trace file is empty")

    clean = [s for s in steps if not s["degraded_in"]]
    degraded = [s for s in steps if s["degraded_in"]]

    p_hat = float(np.mean([s["step_correct"] for s in clean])) if clean else float("nan")
    p_deg = float(np.mean([s["step_correct"] for s in degraded])) if degraded else p_hat
    recovery_rate = (
        float(np.mean([s["recovered"] for s in degraded])) if degraded else 0.0
    )

    # Express recovery as the conditional repair probability the model expects:
    # P(back on chain | degraded) = recovery * P(correct step | degraded).
    recovery = 0.0 if p_deg <= EPS else min(1.0, recovery_rate / max(p_deg, EPS))

    return HorizonFit(
        p_hat=p_hat,
        p_deg=p_deg,
        delta=max(0.0, p_hat - p_deg),
        recovery=recovery,
        het=estimate_heterogeneity(clean),
        n_clean_steps=len(clean),
        n_degraded_steps=len(degraded),
    )


def estimate_heterogeneity(clean_steps: list[dict], min_bucket: int = 50) -> float:
    """Logit-scale standard deviation of per-step accuracy across positions.

    Bucket clean steps by their index and take the spread of bucket accuracies
    on the logit scale. The subtlety that matters: a bucket of n Bernoulli
    draws has sampling variance of its own, and on the logit scale that is
    about 1 / (n * p * (1 - p)) by the delta method. Left uncorrected, pure
    noise reads as heterogeneity, and the estimator then invents an inversion
    gap out of nothing. So we subtract the expected sampling variance and clamp
    at zero.

    Returns 0.0 when there is not enough data, which is the conservative
    choice: it makes the corrected model agree with the naive one rather than
    manufacturing a gap.
    """
    buckets: dict[int, list[int]] = {}
    for step in clean_steps:
        buckets.setdefault(int(step["index"]), []).append(int(step["step_correct"]))

    usable = [v for v in buckets.values() if len(v) >= min_bucket]
    if len(usable) < 3:
        return 0.0

    accuracies = [float(np.mean(v)) for v in usable]
    sizes = [len(v) for v in usable]
    clipped = [min(max(a, 1e-3), 1.0 - 1e-3) for a in accuracies]

    observed_var = float(np.var([logit(a) for a in clipped], ddof=1))
    sampling_var = float(np.mean([1.0 / (n * a * (1.0 - a)) for a, n in zip(clipped, sizes)]))
    return float(math.sqrt(max(0.0, observed_var - sampling_var)))


def bootstrap_fit(
    records: list[dict], n_resamples: int = 1000, seed: int = 0
) -> dict[str, tuple[float, float]]:
    """Percentile confidence intervals over episodes (not over steps).

    Resampling episodes rather than steps keeps the within-episode correlation
    intact, which is the whole phenomenon under study.
    """
    rng = np.random.default_rng(seed)
    n = len(records)
    keys = ("p_hat", "p_deg", "delta", "recovery", "het")
    draws: dict[str, list[float]] = {k: [] for k in keys}

    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        sample = [records[i] for i in idx]
        try:
            fitted = asdict(fit(sample))
        except ValueError:
            continue
        for k in keys:
            value = fitted[k]
            if not math.isnan(value):
                draws[k].append(value)

    return {
        k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
        if v
        else (float("nan"), float("nan"))
        for k, v in draws.items()
    }


def observed_success_by_horizon(records: list[dict]) -> dict[int, float]:
    by_horizon: dict[int, list[int]] = {}
    for record in records:
        by_horizon.setdefault(int(record["horizon"]), []).append(int(record["success"]))
    return {h: float(np.mean(v)) for h, v in sorted(by_horizon.items())}


def observed_h50(records: list[dict], target: float = 0.5) -> float:
    """Horizon at which end-to-end success crosses the target.

    Log-linear interpolation between the two bracketing measured horizons,
    which is the right interpolation for a law that is geometric in H.
    """
    curve = observed_success_by_horizon(records)
    horizons = sorted(curve)
    previous_h, previous_s = None, None
    for h in horizons:
        s = curve[h]
        if s < target and previous_h is not None and previous_s >= target:
            if previous_s <= target or s <= 0.0:
                return float(previous_h)
            span = math.log(previous_s) - math.log(s)
            if span <= 0:
                return float(previous_h)
            frac = (math.log(previous_s) - math.log(target)) / span
            return float(previous_h + frac * (h - previous_h))
        previous_h, previous_s = h, s
    return float(horizons[-1]) if horizons else float("nan")


def summarize(records: list[dict], targets: list[float], query_horizons: list[int]) -> dict:
    """Everything a row of the paper's main results table needs."""
    fitted = fit(records)
    model = fitted.to_model()

    requirements = []
    for target in targets:
        for horizon in query_horizons:
            naive = naive_required_competence(horizon, target)
            corrected = model.required_competence(horizon, target)
            requirements.append(
                {
                    "target": target,
                    "horizon": horizon,
                    "p_star_naive": naive,
                    "p_star_model": corrected,
                    "gap_pp": 100.0 * (corrected - naive),
                }
            )

    predicted_h50 = (
        math.log(0.5) / math.log(fitted.p_hat) if 0.0 < fitted.p_hat < 1.0 else float("nan")
    )

    total_calls = sum(int(s["n_calls"]) for s in _flat_steps(records))
    n_steps = max(1, len(_flat_steps(records)))

    return {
        "fit": asdict(fitted),
        "mean_calls_per_step": total_calls / n_steps,
        "success_by_horizon": observed_success_by_horizon(records),
        "h50_observed": observed_h50(records),
        "h50_predicted_naive": predicted_h50,
        "h50_predicted_model": model.reachable_horizon(0.5),
        "requirements": requirements,
    }
