"""Stopping-time, path-density, and numerical regressions for sum-intersection sampling."""

import math
from typing import cast
from unittest.mock import Mock

import numpy as np
import pytest

from efficient_importance_sampling import MultidimensionalSiegmund, SumIntersectionRule


def _zero_tilt_problem(
    monkeypatch: pytest.MonkeyPatch, dimension: int, order: int
) -> SumIntersectionRule:
    """Use the original Gaussian law to isolate stopping and event classification."""
    problem = SumIntersectionRule(dimension, order, -0.5 * np.ones(dimension), np.eye(dimension))
    monkeypatch.setattr(problem, "compute_feasible_mixture", lambda: ([np.zeros(dimension)], [1.0]))
    return problem


@pytest.mark.parametrize(
    "order, positions, expected",
    [
        pytest.param(2, [[0.5, 0.5, -5.0], [-0.6, 0.6, -5.0]], 0.0, id="strict-boundary"),
        pytest.param(2, [[-4.0, 0.7, 0.6]], 1.0, id="absolute-not-signed-order"),
        pytest.param(2, [[0.1, 0.2, 5.0], [0.6, 0.7, 5.0]], 1.0, id="smallest-not-largest"),
        pytest.param(2, [[-1.0, -2.0, 3.0]], 0.0, id="too-few-positive-coordinates"),
        pytest.param(2, [[0.0, 2.0, -3.0]], 0.0, id="zero-is-not-positive"),
        pytest.param(2, [[0.6, 0.7, 0.8]], 1.0, id="more-than-L-positive"),
        pytest.param(1, [[2.0, 0.0], [-2.0, -2.0]], 0.0, id="order-one-simultaneous-exit"),
        pytest.param(1, [[1.5, -1.5]], 1.0, id="order-one-wrong-exit"),
    ],
)
def test_stopping_and_error_event_follow_the_sum_intersection_rule(
    monkeypatch: pytest.MonkeyPatch, order: int, positions: list[list[float]], expected: float
) -> None:
    """Follow two identical paths, including exits at the last permitted step."""
    dimension = len(positions[0])
    problem = _zero_tilt_problem(monkeypatch, dimension, order)
    generator = Mock(spec=np.random.Generator)
    generator.choice.return_value = 0
    increments = np.diff(np.asarray([[0.0] * dimension, *positions]), axis=0)
    generator.multivariate_normal.side_effect = list(increments) * 2

    result = problem.simulate_wrong_exit_probability(
        b=1.0, n_samples=2, rng=cast(np.random.Generator, generator), max_steps=len(positions)
    )

    assert result.estimate == pytest.approx(expected)
    assert result.std_error == 0.0
    assert result.samples_used == 2
    assert result.log_probability == (0.0 if expected else -np.inf)
    assert result.relative_error == (0.0 if expected else np.inf)
    assert generator.choice.call_count == 2
    assert generator.multivariate_normal.call_count == 2 * len(positions)


def test_path_weight_uses_all_components_and_elapsed_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unequal mixture weights and nonzero CGFs distinguish the full path density."""
    covariance = 0.8 * np.eye(3) + 0.2 * np.ones((3, 3))
    problem = SumIntersectionRule(3, 2, -0.5 * np.ones(3), covariance)
    tilts = [np.array([0.4, 0.0, 0.0]), np.array([0.0, 0.8, 0.0])]
    weights = [0.25, 0.75]
    monkeypatch.setattr(problem, "compute_feasible_mixture", lambda: (tilts, weights))
    generator = Mock(spec=np.random.Generator)
    generator.choice.side_effect = [0, 1]
    generator.multivariate_normal.side_effect = [
        np.array([0.2, 0.2, -0.2]),
        np.array([0.8, 1.8, -0.8]),
        np.array([-1.0, -1.0, -2.0]),
    ]

    result = problem.simulate_wrong_exit_probability(
        1.0, n_samples=2, rng=cast(np.random.Generator, generator), max_steps=2
    )

    final_position = np.array([1.0, 2.0, -1.0])
    density = sum(
        weight * math.exp(float(theta @ final_position - 2 * problem.cgf.Lambda(theta)))
        for theta, weight in zip(tilts, weights)
    )
    # Contributions are [1/density, 0]; sample SE with ddof=1 equals their mean.
    assert result.estimate == pytest.approx(0.5 / density)
    assert result.std_error == pytest.approx(0.5 / density)
    assert result.relative_error == pytest.approx(1.0)
    assert result.log_probability == pytest.approx(math.log(0.5 / density))
    assert generator.choice.call_count == 2
    for index, call in enumerate(generator.multivariate_normal.call_args_list):
        theta = tilts[0 if index < 2 else 1]
        np.testing.assert_allclose(call.args[0], problem.cgf.mean + covariance @ theta)
        np.testing.assert_array_equal(call.args[1], covariance)


@pytest.mark.parametrize("log_weight_magnitude", [400.0, 800.0])
def test_rare_contributions_preserve_log_estimate_and_relative_error(
    monkeypatch: pytest.MonkeyPatch, log_weight_magnitude: float
) -> None:
    """Avoid both variance underflow and loss of a finite log probability estimate."""
    problem = _zero_tilt_problem(monkeypatch, 3, 2)
    monkeypatch.setattr(
        problem, "compute_feasible_mixture", lambda: ([np.array([1.0, 1.0, 0.0])], [1.0])
    )
    generator = Mock(spec=np.random.Generator)
    generator.choice.return_value = 0
    magnitude = log_weight_magnitude / 2.0
    generator.multivariate_normal.side_effect = [
        np.array([magnitude, magnitude, -magnitude]),
        np.array([-2.0, -2.0, -2.0]),
    ]

    result = problem.simulate_wrong_exit_probability(
        1.0, n_samples=2, rng=cast(np.random.Generator, generator), max_steps=1
    )

    assert result.log_probability == pytest.approx(-log_weight_magnitude - math.log(2.0))
    assert result.relative_error == pytest.approx(1.0)
    if log_weight_magnitude == 400.0:
        expected = math.exp(-400.0) / 2.0
        assert result.estimate == pytest.approx(expected, rel=1e-12, abs=0.0)
        assert result.std_error == pytest.approx(expected, rel=1e-12, abs=0.0)
    else:
        assert result.estimate == result.std_error == 0.0


@pytest.mark.parametrize("max_steps", [None, 1, 3])
def test_unfinished_paths_abort_without_a_partial_estimate(
    monkeypatch: pytest.MonkeyPatch, max_steps: int | None
) -> None:
    """A successful first path cannot hide an unfinished second path."""
    problem = _zero_tilt_problem(monkeypatch, 3, 2)
    generator = Mock(spec=np.random.Generator)
    generator.choice.return_value = 0
    limit = 1010 if max_steps is None else max_steps
    generator.multivariate_normal.side_effect = [np.ones(3) * 2] + [np.zeros(3)] * limit

    with pytest.raises(RuntimeError, match=f"Sample 2 did not exit within max_steps={limit}"):
        problem.simulate_wrong_exit_probability(
            1.5, n_samples=2, rng=cast(np.random.Generator, generator), max_steps=max_steps
        )

    assert generator.multivariate_normal.call_count == limit + 1


def test_seeded_simulation_is_reproducible() -> None:
    """Exercise the real proposals and sampler with the same seed twice."""
    problem = SumIntersectionRule(3, 2, -0.5 * np.ones(3), 0.8 * np.eye(3) + 0.2 * np.ones((3, 3)))
    first = problem.simulate_wrong_exit_probability(1.0, 50, np.random.default_rng(101))
    second = problem.simulate_wrong_exit_probability(1.0, 50, np.random.default_rng(101))

    assert first.estimate == second.estimate
    assert first.std_error == second.std_error
    assert first.relative_error == second.relative_error
    assert first.log_probability == second.log_probability
    assert first.samples_used == second.samples_used == 50


def test_order_one_estimate_agrees_with_siegmund() -> None:
    """At L=1 the seeded path experiment reduces to the existing unit-boundary rule."""
    mean = -0.5 * np.ones(3)
    covariance = 0.8 * np.eye(3) + 0.2 * np.ones((3, 3))
    problem = SumIntersectionRule(3, 1, mean, covariance)
    siegmund = MultidimensionalSiegmund(3, 1.0, 1.0, mean, covariance)

    actual = problem.simulate_wrong_exit_probability(1.0, 100, np.random.default_rng(211))
    expected = siegmund.simulate_wrong_exit_probability(1.0, 100, rng=np.random.default_rng(211))

    assert actual.estimate == pytest.approx(expected.estimate, rel=1e-12)
    assert actual.log_probability == pytest.approx(expected.log_probability, rel=1e-12)


def test_small_threshold_matches_gaussian_one_step_probability() -> None:
    """At b near zero, the event approaches a binomial Gaussian sign probability."""
    problem = SumIntersectionRule(3, 2, -0.5 * np.ones(3), np.eye(3))
    positive_probability = 0.5 * math.erfc(0.5 / math.sqrt(2.0))
    expected = 3 * positive_probability**2 * (1 - positive_probability) + positive_probability**3

    result = problem.simulate_wrong_exit_probability(1e-8, 6000, np.random.default_rng(31))

    assert abs(result.estimate - expected) < 6.0 * result.std_error
    assert 0.0 < result.relative_error < 0.05


@pytest.mark.parametrize(
    "name, value, message",
    [
        *[("n_samples", value, "at least two") for value in [0, 1, -1, 1.5, True, np.bool_(True)]],
        *[("b", value, "finite and positive") for value in [0.0, -1.0, np.nan, np.inf, True, "1"]],
        *[("max_steps", value, "positive integer") for value in [0, -1, 1.5, np.nan, np.inf, True]],
    ],
)
def test_invalid_arguments_fail_before_proposal_construction(
    monkeypatch: pytest.MonkeyPatch, name: str, value: object, message: str
) -> None:
    problem = _zero_tilt_problem(monkeypatch, 3, 2)
    mixture = Mock(side_effect=AssertionError("Validation must run first"))
    monkeypatch.setattr(problem, "compute_feasible_mixture", mixture)
    arguments = {"b": 1.0, "n_samples": 2, name: value}

    with pytest.raises(ValueError, match=message):
        problem.simulate_wrong_exit_probability(**arguments)  # type: ignore[arg-type]
    mixture.assert_not_called()


@pytest.mark.parametrize("mean", [np.zeros(3), np.array([-0.5, -0.5, 0.1])])
def test_simulation_rejects_nonnegative_drifts(mean: np.ndarray) -> None:
    problem = SumIntersectionRule(3, 2, mean, np.eye(3))
    with pytest.raises(ValueError, match="strictly negative"):
        problem.simulate_wrong_exit_probability(1.0, 2)


def test_failed_proposal_construction_does_not_start_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem = _zero_tilt_problem(monkeypatch, 3, 2)
    monkeypatch.setattr(
        problem, "compute_feasible_mixture", Mock(side_effect=RuntimeError("invalid proposal"))
    )
    generator = Mock(spec=np.random.Generator)

    with pytest.raises(RuntimeError, match="invalid proposal"):
        problem.simulate_wrong_exit_probability(1.0, 2, cast(np.random.Generator, generator))
    generator.choice.assert_not_called()


def test_nonfinite_path_cannot_be_reported_as_an_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    problem = _zero_tilt_problem(monkeypatch, 3, 2)
    generator = Mock(spec=np.random.Generator)
    generator.choice.return_value = 0
    generator.multivariate_normal.return_value = np.array([np.nan, 2.0, 2.0])

    with pytest.raises(RuntimeError, match="position became nonfinite"):
        problem.simulate_wrong_exit_probability(1.0, 2, cast(np.random.Generator, generator))


@pytest.mark.parametrize("dimension", [3.0, 3.5, True, np.bool_(True)])
def test_dimension_must_be_an_integer(dimension: object) -> None:
    with pytest.raises(ValueError, match="d must be at least 2 and an integer"):
        SumIntersectionRule(dimension, 2, -np.ones(3), np.eye(3))  # type: ignore[arg-type]


@pytest.mark.parametrize("order", [2.0, 1.5, True, np.bool_(True)])
def test_order_must_be_an_integer(order: object) -> None:
    with pytest.raises(ValueError, match="L must satisfy"):
        SumIntersectionRule(3, order, -np.ones(3), np.eye(3))  # type: ignore[arg-type]
