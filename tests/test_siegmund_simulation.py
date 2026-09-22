"""Path-density and numerical regressions for both Siegmund proposal families."""

import math
import warnings
from typing import cast
from unittest.mock import Mock

import numpy as np
import pytest

from efficient_importance_sampling import MultidimensionalSiegmund


def _set_proposals(
    monkeypatch: pytest.MonkeyPatch,
    problem: MultidimensionalSiegmund,
    tilts: list[np.ndarray],
    weights: list[float],
) -> None:
    """Provide controlled proposals; the full mixture retains uniform weights."""
    monkeypatch.setattr(
        problem, "get_feasible_mixture", lambda: (tilts, weights, list(range(len(tilts))))
    )
    monkeypatch.setattr(
        problem,
        "compute_optimal_tilts",
        lambda: {index: (theta, 0.0) for index, theta in enumerate(tilts)},
    )


@pytest.mark.parametrize("use_feasible_mixture", [True, False])
def test_complete_path_density_and_sample_standard_error(
    monkeypatch: pytest.MonkeyPatch, use_feasible_mixture: bool
) -> None:
    """Include all components and the CGF at stopping; [w, 0] has sample SE w/2."""
    covariance = np.array([[1.0, 0.2], [0.2, 1.0]])
    problem = MultidimensionalSiegmund(2, 1.0, 1.0, -0.5 * np.ones(2), covariance)
    tilts = [np.array([0.4, 0.0]), np.array([0.0, 0.8])]
    weights = [0.25, 0.75] if use_feasible_mixture else [0.5, 0.5]
    _set_proposals(monkeypatch, problem, tilts, weights)
    generator = Mock(spec=np.random.Generator)
    generator.choice.side_effect = [0, 1]
    generator.multivariate_normal.side_effect = [
        np.array([0.2, 0.2]),
        np.array([1.3, 1.8]),
        np.array([-2.0, -2.0]),
    ]

    result = problem.simulate_wrong_exit_probability(
        1.0, 2, use_feasible_mixture, cast(np.random.Generator, generator), max_steps=2
    )

    final_position = np.array([1.5, 2.0])
    density = sum(
        weight * math.exp(float(theta @ final_position - 2 * problem.cgf.Lambda(theta)))
        for theta, weight in zip(tilts, weights)
    )
    assert result.estimate == pytest.approx(0.5 / density)
    assert result.std_error == pytest.approx(0.5 / density)
    assert result.relative_error == pytest.approx(1.0)
    assert result.log_probability == pytest.approx(math.log(0.5 / density))
    assert generator.choice.call_count == 2
    for index, call in enumerate(generator.multivariate_normal.call_args_list):
        theta = tilts[0 if index < 2 else 1]
        np.testing.assert_allclose(call.args[0], problem.cgf.mean + covariance @ theta)
        np.testing.assert_array_equal(call.args[1], covariance)


@pytest.mark.parametrize("use_feasible_mixture", [True, False])
@pytest.mark.parametrize("log_weight_magnitude", [400.0, 800.0])
def test_rare_contributions_preserve_log_estimate_and_relative_error(
    monkeypatch: pytest.MonkeyPatch,
    use_feasible_mixture: bool,
    log_weight_magnitude: float,
) -> None:
    """Retain uncertainty when squaring weights, or the weights themselves, underflow."""
    problem = MultidimensionalSiegmund(1, 1.0, 1.0, np.array([-0.5]), np.eye(1))
    _set_proposals(monkeypatch, problem, [np.ones(1)], [1.0])
    generator = Mock(spec=np.random.Generator)
    generator.choice.return_value = 0
    generator.multivariate_normal.side_effect = [
        np.array([log_weight_magnitude]),
        np.array([-2.0]),
    ]

    result = problem.simulate_wrong_exit_probability(
        1.0, 2, use_feasible_mixture, cast(np.random.Generator, generator), max_steps=1
    )

    assert result.log_probability == pytest.approx(-log_weight_magnitude - math.log(2.0))
    assert result.relative_error == pytest.approx(1.0)
    if log_weight_magnitude == 400.0:
        expected = math.exp(-400.0) / 2.0
        assert result.estimate == pytest.approx(expected, rel=1e-12, abs=0.0)
        assert result.std_error == pytest.approx(expected, rel=1e-12, abs=0.0)
    else:
        assert result.estimate == result.std_error == 0.0


@pytest.mark.parametrize("use_feasible_mixture", [True, False])
@pytest.mark.parametrize("position", [-2.0, 2.0, 800.0])
def test_single_sample_keeps_estimate_with_undefined_uncertainty(
    monkeypatch: pytest.MonkeyPatch, use_feasible_mixture: bool, position: float
) -> None:
    """A hit, a miss, and an underflowed hit all have undefined one-sample errors."""
    problem = MultidimensionalSiegmund(1, 1.0, 1.0, np.array([-0.5]), np.eye(1))
    _set_proposals(monkeypatch, problem, [np.ones(1)], [1.0])
    generator = Mock(spec=np.random.Generator)
    generator.choice.return_value = 0
    generator.multivariate_normal.return_value = np.array([position])

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = problem.simulate_wrong_exit_probability(
            1.0, 1, use_feasible_mixture, cast(np.random.Generator, generator), max_steps=1
        )

    assert result.estimate == pytest.approx(math.exp(-position) if position > 0 else 0.0)
    assert result.log_probability == (-position if position > 0 else -math.inf)
    assert math.isnan(result.std_error)
    assert math.isnan(result.relative_error)
    assert result.samples_used == 1


@pytest.mark.parametrize("use_feasible_mixture", [True, False])
def test_no_wrong_exits_report_empirical_zero(
    monkeypatch: pytest.MonkeyPatch, use_feasible_mixture: bool
) -> None:
    """Multiple identical misses keep the established zero/inf diagnostics."""
    problem = MultidimensionalSiegmund(1, 1.0, 1.0, np.array([-0.5]), np.eye(1))
    _set_proposals(monkeypatch, problem, [np.zeros(1)], [1.0])
    generator = Mock(spec=np.random.Generator)
    generator.choice.return_value = 0
    generator.multivariate_normal.return_value = np.array([-2.0])

    result = problem.simulate_wrong_exit_probability(
        1.0, 2, use_feasible_mixture, cast(np.random.Generator, generator), max_steps=1
    )

    assert result.estimate == result.std_error == 0.0
    assert result.log_probability == -math.inf
    assert result.relative_error == math.inf


@pytest.mark.parametrize("use_feasible_mixture", [True, False])
def test_fractional_default_limit_and_incomplete_second_path(
    monkeypatch: pytest.MonkeyPatch, use_feasible_mixture: bool
) -> None:
    """Preserve int(10*b), and discard the whole experiment if a later path stalls."""
    problem = MultidimensionalSiegmund(1, 1.0, 1.0, np.array([-0.5]), np.eye(1))
    _set_proposals(monkeypatch, problem, [np.zeros(1)], [1.0])
    generator = Mock(spec=np.random.Generator)
    generator.choice.return_value = 0
    generator.multivariate_normal.side_effect = [np.array([2.0])] + [np.zeros(1)] * 1015

    with pytest.raises(RuntimeError, match="Sample 2 did not exit within max_steps=1015"):
        problem.simulate_wrong_exit_probability(
            1.5, 2, use_feasible_mixture, cast(np.random.Generator, generator)
        )
    assert generator.multivariate_normal.call_count == 1016


@pytest.mark.parametrize(
    "ell, u, b",
    [(2.0, 1.0, 1e308), (1.0, 2.0, 1e308), (1e-200, 1.0, 1e-200), (1.0, 1e-200, 1e-200)],
)
def test_invalid_scaled_boundaries_fail_before_proposal_construction(
    monkeypatch: pytest.MonkeyPatch, ell: float, u: float, b: float
) -> None:
    problem = MultidimensionalSiegmund(1, ell, u, np.array([-0.5]), np.eye(1))
    mixture = Mock(side_effect=AssertionError("Validation must run first"))
    monkeypatch.setattr(problem, "get_feasible_mixture", mixture)

    with pytest.raises(ValueError, match="boundary magnitudes must be finite and positive"):
        problem.simulate_wrong_exit_probability(b, 2)
    mixture.assert_not_called()


@pytest.mark.parametrize("position", [math.nan, math.inf, -math.inf])
def test_nonfinite_path_cannot_be_reported_as_an_exit(
    monkeypatch: pytest.MonkeyPatch, position: float
) -> None:
    problem = MultidimensionalSiegmund(1, 1.0, 1.0, np.array([-0.5]), np.eye(1))
    _set_proposals(monkeypatch, problem, [np.zeros(1)], [1.0])
    generator = Mock(spec=np.random.Generator)
    generator.choice.return_value = 0
    generator.multivariate_normal.return_value = np.array([position])

    with pytest.raises(RuntimeError, match="position became nonfinite"):
        problem.simulate_wrong_exit_probability(
            1.0, 2, rng=cast(np.random.Generator, generator), max_steps=1
        )
