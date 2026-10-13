"""Guardrails for the math. Run with: pytest -q"""

import math

import pytest

from src.horizon_model import (
    HorizonModel,
    difficulty_profile,
    naive_required_competence,
)


def test_naive_law_is_the_corner_case():
    """With no heterogeneity and no recovery, the model must reproduce p^H."""
    model = HorizonModel(p=0.9, delta=0.4, recovery=0.0, het=0.0)
    for horizon in (1, 8, 32):
        assert model.success(horizon) == pytest.approx(0.9**horizon, rel=1e-9)


def test_the_quoted_arithmetic():
    """Eight chained components at 90% each: the number in the project brief."""
    assert HorizonModel(p=0.9).success(8) == pytest.approx(0.4305, abs=1e-4)


def test_inversion_round_trips():
    model = HorizonModel(p=0.95, delta=0.1, recovery=0.2, het=0.3)
    for horizon in (4, 16, 64):
        p_star = model.required_competence(horizon, 0.5)
        assert model.success(horizon, p_star) == pytest.approx(0.5, abs=1e-6)


def test_naive_inversion_matches_closed_form():
    for horizon in (8, 32, 128):
        model = HorizonModel(delta=0.0, recovery=0.0, het=0.0)
        assert model.required_competence(horizon, 0.5) == pytest.approx(
            naive_required_competence(horizon, 0.5), abs=1e-6
        )


def test_heterogeneity_is_optimism():
    """AM-GM: a spread of step difficulties lowers success below p^H."""
    homogeneous = HorizonModel(p=0.9, het=0.0)
    spread = HorizonModel(p=0.9, het=1.0)
    assert spread.success(16) < homogeneous.success(16)
    assert spread.required_competence(16, 0.5) > homogeneous.required_competence(16, 0.5)


def test_recovery_is_pessimism():
    """Letting the agent repair a bad state makes the naive law too harsh."""
    strict = HorizonModel(p=0.9, delta=0.1, recovery=0.0)
    forgiving = HorizonModel(p=0.9, delta=0.1, recovery=0.5)
    assert forgiving.success(16) > strict.success(16)


def test_success_is_monotone_in_p():
    model = HorizonModel(delta=0.05, recovery=0.1, het=0.2)
    values = [model.success(16, p) for p in (0.5, 0.7, 0.9, 0.99)]
    assert values == sorted(values)


def test_profile_mean_matches_requested_p():
    profile = difficulty_profile(32, 0.93, het=0.8)
    assert profile.mean() == pytest.approx(0.93, abs=1e-6)
    assert len(profile) == 32


def test_reachable_horizon_agrees_with_closed_form():
    model = HorizonModel(p=0.95, delta=0.0, recovery=0.0, het=0.0)
    expected = int(math.floor(math.log(0.5) / math.log(0.95)))
    assert model.reachable_horizon(0.5) == expected
