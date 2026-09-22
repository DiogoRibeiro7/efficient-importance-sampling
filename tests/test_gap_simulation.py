"""Scientific regressions for the gap-rule probability estimator."""

import math
from typing import cast
from unittest.mock import Mock

import numpy as np
import pytest

from efficient_importance_sampling import GapRule


def _zero_tilt_problem(
    monkeypatch: pytest.MonkeyPatch, dimension: int, signal_count: int
) -> GapRule:
    """Use the original law to isolate the stopping rule and selected identities."""
    mean = np.concatenate([np.full(signal_count, 0.5), np.full(dimension - signal_count, -0.5)])
    problem = GapRule(dimension, signal_count, mean, np.eye(dimension))
    monkeypatch.setattr(problem, "compute_feasible_mixture", lambda: ([np.zeros(dimension)], [1.0]))
    return problem


@pytest.mark.parametrize(
    "signal_count, positions, expected",
    [
        pytest.param(1, [[0.5, -0.5], [-1.0, 1.0]], 1.0, id="strict-boundary"),
        pytest.param(1, [[0.0, 0.0], [2.0, 0.0]], 0.0, id="boundary-tie"),
        pytest.param(
            2,
            [[100.0, 1.0, 0.5, -100.0], [100.0, -2.0, 3.0, -100.0]],
            1.0,
            id="adjacent-gap-not-total-spread",
        ),
        pytest.param(2, [[2.0, 2.0, 0.0, 0.0]], 0.0, id="ties-within-groups"),
        pytest.param(2, [[0.0, 0.0, 2.0, 2.0]], 1.0, id="multiple-wrong-selections"),
        pytest.param(2, [[8.0, 4.0, 6.0, 3.0]], 1.0, id="positive-signs-do-not-determine-error"),
        pytest.param(2, [[-3.0, -4.0, -1.0, -0.5]], 1.0, id="negative-selected-coordinates"),
        pytest.param(3, [[3.0, 2.0, -2.0, 0.0]], 1.0, id="select-all-but-one-wrong"),
        pytest.param(3, [[3.0, 2.0, 0.0, -2.0]], 0.0, id="select-all-but-one-correct"),
        pytest.param(
            2, [[0.0, 0.5, 0.3, 0.0], [2.0, 3.0, 0.0, 0.0]], 0.0, id="membership-at-stopping-only"
        ),
    ],
)
def test_gap_stopping_and_selected_set(
    monkeypatch: pytest.MonkeyPatch,
    signal_count: int,
    positions: list[list[float]],
    expected: float,
) -> None:
    """Two prescribed paths include valid exits on the last permitted step."""
    dimension = len(positions[0])
    problem = _zero_tilt_problem(monkeypatch, dimension, signal_count)
    generator = Mock(spec=np.random.Generator)
    generator.choice.return_value = 0
    increments = np.diff(np.asarray([[0.0] * dimension, *positions]), axis=0)
    generator.multivariate_normal.side_effect = list(increments) * 2

    result = problem.simulate_wrong_exit_probability(
        1.0, 2, cast(np.random.Generator, generator), max_steps=len(positions)
    )

    assert result.estimate == pytest.approx(expected)
    assert result.std_error == 0.0
    assert result.samples_used == 2
    assert result.log_probability == (0.0 if expected else -np.inf)
    assert result.relative_error == (0.0 if expected else np.inf)
    assert generator.choice.call_count == 2
    assert generator.multivariate_normal.call_count == 2 * len(positions)


def test_gap_path_weight_uses_complete_mixture_and_step_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use unequal weights and nonzero CGFs to check the exact stopped-path density."""
    covariance = 0.8 * np.eye(4) + 0.2 * np.ones((4, 4))
    problem = GapRule(4, 2, np.array([0.5, 0.7, -0.2, -0.6]), covariance)
    tilts = [np.array([-0.4, 0.0, 0.4, 0.0]), np.array([0.0, -0.8, 0.0, 0.8])]
    weights = [0.3, 0.7]
    monkeypatch.setattr(problem, "compute_feasible_mixture", lambda: (tilts, weights))
    generator = Mock(spec=np.random.Generator)
    generator.choice.side_effect = [0, 1]
    first_position = np.array([0.1, 0.2, 0.3, 0.4])
    final_position = np.array([1.5, 0.0, 3.0, -2.0])
    generator.multivariate_normal.side_effect = [
        first_position,
        final_position - first_position,
        np.array([3.0, 2.0, -1.0, -2.0]),
    ]

    result = problem.simulate_wrong_exit_probability(
        1.0, 2, cast(np.random.Generator, generator), max_steps=2
    )

    density = sum(
        weight * math.exp(float(theta @ final_position - 2 * problem.cgf.Lambda(theta)))
        for theta, weight in zip(tilts, weights)
    )
    # Contributions [1/density, 0] have sample standard error equal to their mean.
    assert result.estimate == pytest.approx(0.5 / density)
    assert result.std_error == pytest.approx(0.5 / density)
    assert result.relative_error == pytest.approx(1.0)
    assert result.log_probability == pytest.approx(math.log(0.5 / density))
    assert generator.choice.call_count == 2
    for index, call in enumerate(generator.multivariate_normal.call_args_list):
        theta = tilts[0 if index < 2 else 1]
        np.testing.assert_allclose(call.args[0], problem.cgf.mean + covariance @ theta)
        np.testing.assert_array_equal(call.args[1], covariance)


@pytest.mark.parametrize("magnitude", [400.0, 800.0])
def test_gap_likelihood_preserves_rare_probability_diagnostics(
    monkeypatch: pytest.MonkeyPatch, magnitude: float
) -> None:
    """The two-coordinate ray has log weight minus the observed wrong-selection gap."""
    problem = _zero_tilt_problem(monkeypatch, 2, 1)
    monkeypatch.setattr(
        problem, "compute_feasible_mixture", lambda: ([np.array([-1.0, 1.0])], [1.0])
    )
    generator = Mock(spec=np.random.Generator)
    generator.choice.return_value = 0
    generator.multivariate_normal.side_effect = [
        np.array([-magnitude / 2.0, magnitude / 2.0]),
        np.array([2.0, -2.0]),
    ]

    result = problem.simulate_wrong_exit_probability(
        1.0, 2, cast(np.random.Generator, generator), max_steps=1
    )

    assert result.log_probability == pytest.approx(-magnitude - math.log(2.0))
    assert result.relative_error == pytest.approx(1.0)
    expected = math.exp(-magnitude) / 2.0
    assert result.estimate == pytest.approx(expected, rel=1e-12, abs=0.0)
    assert result.std_error == pytest.approx(expected, rel=1e-12, abs=0.0)


@pytest.mark.parametrize("max_steps", [None, 1, 3])
def test_unfinished_gap_paths_abort_the_whole_estimate(
    monkeypatch: pytest.MonkeyPatch, max_steps: int | None
) -> None:
    problem = _zero_tilt_problem(monkeypatch, 2, 1)
    generator = Mock(spec=np.random.Generator)
    generator.choice.return_value = 0
    limit = 1010 if max_steps is None else max_steps
    generator.multivariate_normal.side_effect = [np.array([-2.0, 2.0])] + [np.zeros(2)] * limit

    with pytest.raises(RuntimeError, match=f"Sample 2 did not exit within max_steps={limit}"):
        problem.simulate_wrong_exit_probability(
            1.5, 2, cast(np.random.Generator, generator), max_steps=max_steps
        )

    assert generator.multivariate_normal.call_count == limit + 1


def test_seeded_gap_simulation_is_reproducible() -> None:
    problem = GapRule(4, 2, np.array([0.5, 0.8, -0.2, -0.6]), np.eye(4))
    first = problem.simulate_wrong_exit_probability(1.0, 50, np.random.default_rng(11))
    second = problem.simulate_wrong_exit_probability(1.0, 50, np.random.default_rng(11))

    assert first.estimate == second.estimate
    assert first.std_error == second.std_error
    assert first.relative_error == second.relative_error
    assert first.log_probability == second.log_probability
    assert first.samples_used == second.samples_used == 50


@pytest.mark.parametrize("offset", [-10.0, 10.0])
def test_common_drift_offset_preserves_the_gap_experiment(offset: float) -> None:
    """Remark 5.1 permits drifts of the same sign when their groups are separated."""
    mean = np.array([0.5, 0.8, -0.2, -0.6])
    covariance = 0.8 * np.eye(4) + 0.2 * np.ones((4, 4))
    original = GapRule(4, 2, mean, covariance)
    shifted = GapRule(4, 2, mean + offset, covariance)

    expected = original.simulate_wrong_exit_probability(1.0, 50, np.random.default_rng(11))
    actual = shifted.simulate_wrong_exit_probability(1.0, 50, np.random.default_rng(11))

    assert actual.estimate == pytest.approx(expected.estimate, rel=1e-8)
    assert actual.std_error == pytest.approx(expected.std_error, rel=1e-8)
    assert actual.relative_error == pytest.approx(expected.relative_error, rel=1e-8)
    assert actual.log_probability == pytest.approx(expected.log_probability, rel=1e-8)


def test_small_threshold_matches_gaussian_difference_probability() -> None:
    """For two coordinates, a vanishing threshold selects the larger first increment."""
    rho = 0.3
    covariance = np.array([[1.0, rho], [rho, 1.0]])
    problem = GapRule(2, 1, np.array([0.5, -0.5]), covariance)
    # X_noise - X_signal is N(-1, 2*(1-rho)); compare with its exact positive tail.
    expected = 0.5 * math.erfc(1.0 / (2.0 * math.sqrt(1.0 - rho)))

    result = problem.simulate_wrong_exit_probability(1e-8, 6000, np.random.default_rng(71))

    assert abs(result.estimate - expected) < 6.0 * result.std_error
    assert 0.0 < result.relative_error < 0.05


@pytest.mark.parametrize(
    "mean",
    [np.zeros(4), np.array([0.5, 0.0, 0.0, -0.5]), np.array([0.5, -0.2, 0.2, -0.5])],
)
def test_simulation_rejects_unseparated_drift_groups(
    monkeypatch: pytest.MonkeyPatch, mean: np.ndarray
) -> None:
    problem = GapRule(4, 2, mean, np.eye(4))
    mixture = Mock(side_effect=AssertionError("Drifts must be validated first"))
    monkeypatch.setattr(problem, "compute_feasible_mixture", mixture)

    with pytest.raises(ValueError, match="must strictly exceed"):
        problem.simulate_wrong_exit_probability(1.0, 2)
    mixture.assert_not_called()


@pytest.mark.parametrize(
    "name, value, message",
    [
        ("b", np.nan, "finite and positive"),
        ("n_samples", 1, "at least two"),
        ("max_steps", 0, "positive integer"),
    ],
)
def test_gap_sampling_controls_are_validated_before_proposals(
    monkeypatch: pytest.MonkeyPatch, name: str, value: object, message: str
) -> None:
    problem = _zero_tilt_problem(monkeypatch, 4, 2)
    mixture = Mock(side_effect=AssertionError("Arguments must be validated first"))
    monkeypatch.setattr(problem, "compute_feasible_mixture", mixture)
    arguments = {"b": 1.0, "n_samples": 2, name: value}

    with pytest.raises(ValueError, match=message):
        problem.simulate_wrong_exit_probability(**arguments)  # type: ignore[arg-type]
    mixture.assert_not_called()


@pytest.mark.parametrize("dimension", [4.0, 3.5, True, np.bool_(True)])
def test_gap_dimension_must_be_an_integer(dimension: object) -> None:
    with pytest.raises(ValueError, match="d must be at least 2 and an integer"):
        GapRule(dimension, 2, np.array([0.5, 0.5, -0.5, -0.5]), np.eye(4))  # type: ignore[arg-type]


@pytest.mark.parametrize("signal_count", [2.0, 1.5, True, np.bool_(True)])
def test_gap_signal_count_must_be_an_integer(signal_count: object) -> None:
    with pytest.raises(ValueError, match="m must satisfy"):
        GapRule(4, signal_count, np.array([0.5, 0.5, -0.5, -0.5]), np.eye(4))  # type: ignore[arg-type]


def test_failed_gap_proposal_does_not_draw_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    problem = _zero_tilt_problem(monkeypatch, 4, 2)
    monkeypatch.setattr(
        problem, "compute_feasible_mixture", Mock(side_effect=RuntimeError("failed"))
    )
    generator = Mock(spec=np.random.Generator)

    with pytest.raises(RuntimeError, match="failed"):
        problem.simulate_wrong_exit_probability(1.0, 2, cast(np.random.Generator, generator))
    generator.choice.assert_not_called()
