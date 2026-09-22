"""Analytical regressions for Gaussian auxiliary proposal construction."""

import numpy as np
import pytest

from efficient_importance_sampling import CumulantFunction, GapRule, MultidimensionalSiegmund


@pytest.mark.parametrize(
    ("mean", "expected"),
    [(-0.5, 1.0), (-10.0, 20.0), (-1e-6, 2e-6), (0.0, 0.0), (0.5, 0.0)],
)
def test_gaussian_ray_endpoint(mean: float, expected: float) -> None:
    """Cover small roots, roots above the old bound, and rays with no positive root."""
    cgf = CumulantFunction(np.array([mean]), np.eye(1))

    theta = cgf.optimal_ray_tilt(np.ones(1))

    np.testing.assert_allclose(theta, [expected], atol=1e-14)
    assert cgf.Lambda(theta) == pytest.approx(0.0, abs=1e-12)
    if expected > 0.0:
        assert cgf.Lambda(0.5 * theta) < 0.0
        assert cgf.Lambda(1.01 * theta) > 0.0


def test_correlated_gap_ray_uses_difference_variance() -> None:
    """A pairwise tilt must account for covariance between signal and noise."""
    cgf = CumulantFunction(np.array([0.5, -0.5]), np.array([[1.0, 0.25], [0.25, 1.0]]))

    theta = cgf.optimal_ray_tilt(np.array([-1.0, 1.0]))

    np.testing.assert_allclose(theta, [-4.0 / 3.0, 4.0 / 3.0])
    assert cgf.Lambda(theta) == pytest.approx(0.0, abs=1e-12)
    assert cgf.Lambda(1.01 * theta) > 0.0
    np.testing.assert_allclose(cgf.optimal_ray_tilt(np.array([-3.0, 3.0])), theta)


def test_zero_ray_returns_zero_tilt() -> None:
    """A zero direction must not cause division by zero."""
    cgf = CumulantFunction(-np.ones(2), np.eye(2))

    np.testing.assert_array_equal(cgf.optimal_ray_tilt(np.zeros(2)), np.zeros(2))


@pytest.mark.parametrize(
    "direction",
    [np.array([1.0]), np.array([[1.0, 0.0]]), np.array([np.nan, 0.0]), np.array([np.inf, 0.0])],
)
def test_ray_tilt_rejects_invalid_directions(direction: np.ndarray) -> None:
    """Direction inputs must be finite vectors with the model's dimension."""
    cgf = CumulantFunction(-np.ones(2), np.eye(2))

    with pytest.raises(ValueError, match="x must"):
        cgf.optimal_ray_tilt(direction)


def test_siegmund_mixture_contains_exact_coordinate_tilts() -> None:
    """The public mixture builder must include all coordinate roots without clipping."""
    problem = MultidimensionalSiegmund(
        d=3,
        ell=1.0,
        u=2.0,
        mean=np.array([-0.5, -2.0, -6.0]),
        covariance=np.array([[1.0, 0.2, 0.1], [0.2, 2.0, 0.1], [0.1, 0.1, 0.5]]),
    )

    tilts, weights, indices = problem.get_feasible_mixture()

    assert len(tilts) == len(weights) == len(indices) == 6
    np.testing.assert_allclose(tilts[3:], np.diag([1.0, 2.0, 24.0]))
    assert indices == [1, 2, 4, -1, -2, -3]
    np.testing.assert_allclose(weights, np.full(6, 1.0 / 6.0))
    for theta in tilts[3:]:
        assert problem.cgf.Lambda(theta) == pytest.approx(0.0, abs=1e-12)
        assert problem.cgf.Lambda(1.01 * theta) > 0.0


def test_gap_mixture_constructs_every_signal_noise_pair() -> None:
    """The public gap builder must run successfully and preserve pair ordering."""
    problem = GapRule(
        d=4,
        m=2,
        mean=np.array([0.5, 1.0, -0.5, -1.0]),
        covariance=0.75 * np.eye(4) + 0.25 * np.ones((4, 4)),
    )
    expected = np.array(
        [
            [-4.0 / 3.0, 0.0, 4.0 / 3.0, 0.0],
            [-2.0, 0.0, 0.0, 2.0],
            [0.0, -2.0, 2.0, 0.0],
            [0.0, -8.0 / 3.0, 0.0, 8.0 / 3.0],
        ]
    )

    tilts, weights = problem.compute_feasible_mixture()

    assert len(tilts) == len(weights) == 8
    np.testing.assert_allclose(tilts[4:], expected)
    np.testing.assert_allclose(weights, np.full(8, 0.125))
    for theta in tilts[4:]:
        assert problem.cgf.Lambda(theta) == pytest.approx(0.0, abs=1e-12)
        assert problem.cgf.Lambda(1.01 * theta) > 0.0
