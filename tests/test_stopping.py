"""Deterministic path regressions for the Siegmund stopping rule."""

from typing import cast
from unittest.mock import Mock

import numpy as np
import pytest

from efficient_importance_sampling import MultidimensionalSiegmund


def _zero_tilt_problem(
    monkeypatch: pytest.MonkeyPatch,
    ell: float,
    u: float,
    dimension: int,
) -> MultidimensionalSiegmund:
    """Isolate stopping behaviour using the original Gaussian as the proposal."""
    problem = MultidimensionalSiegmund(
        d=dimension,
        ell=ell,
        u=u,
        mean=-0.5 * np.ones(dimension),
        covariance=np.eye(dimension),
    )
    monkeypatch.setattr(
        problem, "get_feasible_mixture", lambda: ([np.zeros(dimension)], [1.0], [1])
    )
    monkeypatch.setattr(problem, "compute_optimal_tilts", lambda: {1: (np.zeros(dimension), 0.0)})
    return problem


@pytest.mark.parametrize("use_feasible_mixture", [True, False])
@pytest.mark.parametrize(
    ("ell", "u", "b", "positions", "expected_estimate"),
    [
        pytest.param(1.0, 3.0, 2.0, [[3.0], [-3.0]], 0.0, id="larger-upper-boundary"),
        pytest.param(3.0, 1.0, 2.0, [[-3.0], [3.0]], 1.0, id="larger-lower-boundary"),
        pytest.param(1.0, 3.0, 2.0, [[6.0], [6.5]], 1.0, id="upper-equality"),
        pytest.param(3.0, 1.0, 2.0, [[-6.0], [-6.5]], 0.0, id="lower-equality"),
        pytest.param(1.0, 2.0, 1.0, [[2.5, 0.0], [2.5, -1.5]], 1.0, id="wait-for-all-coordinates"),
        pytest.param(
            1.0,
            2.0,
            1.0,
            [[2.5, 0.0], [0.0, -1.5], [-1.5, -1.5]],
            0.0,
            id="crossings-must-be-simultaneous",
        ),
        pytest.param(1.0, 1.0, 2.0, [[1.0], [2.5]], 1.0, id="symmetric-boundaries"),
        pytest.param(1.0, 1.0, 1.0, [[1.5]], 1.0, id="one-step-exit"),
    ],
)
def test_simulation_stops_at_the_correct_exit(
    monkeypatch: pytest.MonkeyPatch,
    use_feasible_mixture: bool,
    ell: float,
    u: float,
    b: float,
    positions: list[list[float]],
    expected_estimate: float,
) -> None:
    """Follow prescribed paths and retain valid exits on the final permitted step."""
    dimension = len(positions[0])
    problem = _zero_tilt_problem(monkeypatch, ell, u, dimension)
    generator = Mock(spec=np.random.Generator)
    generator.choice.return_value = 0
    # Convert prescribed positions into increments consumed by the public simulator.
    path = np.asarray([[0.0] * dimension, *positions])
    generator.multivariate_normal.side_effect = list(np.diff(path, axis=0))

    result = problem.simulate_wrong_exit_probability(
        b=b,
        n_samples=1,
        use_feasible_mixture=use_feasible_mixture,
        rng=cast(np.random.Generator, generator),
        max_steps=len(positions),
    )

    assert result.estimate == pytest.approx(expected_estimate)
    assert result.samples_used == 1
    assert generator.multivariate_normal.call_count == len(positions)


@pytest.mark.parametrize("max_steps", [None, 1, 3])
def test_unfinished_paths_raise_instead_of_returning_zero(
    monkeypatch: pytest.MonkeyPatch, max_steps: int | None
) -> None:
    """A bounded simulation must report incomplete paths instead of a false estimate."""
    problem = _zero_tilt_problem(monkeypatch, 1.0, 1.0, 1)
    generator = Mock(spec=np.random.Generator)
    generator.choice.return_value = 0
    generator.multivariate_normal.return_value = np.zeros(1)
    expected_steps = 1020 if max_steps is None else max_steps

    with pytest.raises(RuntimeError, match=f"max_steps={expected_steps}"):
        problem.simulate_wrong_exit_probability(
            b=2.0,
            n_samples=1,
            rng=cast(np.random.Generator, generator),
            max_steps=max_steps,
        )

    assert generator.multivariate_normal.call_count == expected_steps


@pytest.mark.parametrize("max_steps", [0, -1, 1.5, True, np.bool_(True), np.nan, np.inf])
def test_simulation_rejects_invalid_step_limits(
    monkeypatch: pytest.MonkeyPatch, max_steps: object
) -> None:
    """Reject invalid limits before constructing a proposal or drawing samples."""
    problem = _zero_tilt_problem(monkeypatch, 1.0, 1.0, 1)
    mixture = Mock(side_effect=AssertionError("Validation must precede optimisation"))
    monkeypatch.setattr(problem, "get_feasible_mixture", mixture)

    with pytest.raises(ValueError, match="max_steps must be a positive integer"):
        problem.simulate_wrong_exit_probability(
            b=1.0,
            n_samples=1,
            max_steps=max_steps,  # type: ignore[arg-type]
        )

    mixture.assert_not_called()
