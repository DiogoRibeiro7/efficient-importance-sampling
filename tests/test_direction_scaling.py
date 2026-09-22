"""Analytical checks for scale-invariant Gaussian support and ray calculations."""

import math

import numpy as np
import pytest

from efficient_importance_sampling import CumulantFunction


@pytest.mark.parametrize("magnitude", [1e-300, 1e-200, 1.0, 1e200, 1e300])
def test_support_and_maximiser_preserve_positive_homogeneity(magnitude: float) -> None:
    """For N((-1/2,-1/2), I), I((1,2)) = 3/2 + sqrt(5/2)."""
    cgf = CumulantFunction(-0.5 * np.ones(2), np.eye(2))
    point: np.ndarray = magnitude * np.array([1.0, 2.0])
    expected_rate: float = magnitude * (1.5 + math.sqrt(2.5))
    expected_theta: np.ndarray = 0.5 + np.array([1.0, 2.0]) / math.sqrt(10.0)

    # Trap arithmetic failures rather than merely checking for a finite result.
    with np.errstate(over="raise", under="raise", invalid="raise", divide="raise"):
        rate: float = cgf.rate_function_I(point)
        theta: np.ndarray = cgf.optimal_theta(point)

    assert rate == pytest.approx(expected_rate, rel=1e-12, abs=0.0)
    np.testing.assert_allclose(theta, expected_theta, rtol=1e-12, atol=0.0)
    assert float(theta @ point) == pytest.approx(expected_rate, rel=1e-12, abs=0.0)
    assert cgf.Lambda(theta) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("magnitude", [1e-300, 1e-200, 1.0, 1e200, 1e300])
def test_correlated_signed_direction_keeps_its_support_and_ray_endpoint(magnitude: float) -> None:
    """For a signal/noise contrast, the variance is 3/2 and the ray root is 4/3."""
    cgf = CumulantFunction(np.array([0.5, -0.5]), np.array([[1.0, 0.25], [0.25, 1.0]]))
    direction: np.ndarray = magnitude * np.array([-1.0, 1.0])
    expected_theta: np.ndarray = np.array([-4.0 / 3.0, 4.0 / 3.0])

    with np.errstate(over="raise", under="raise", invalid="raise", divide="raise"):
        ray: np.ndarray = cgf.optimal_ray_tilt(direction)
        support: float = cgf.rate_function_I(direction)
        optimum: np.ndarray = cgf.optimal_theta(direction)

    np.testing.assert_allclose(ray, expected_theta, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(optimum, expected_theta, rtol=1e-12, atol=0.0)
    assert support == pytest.approx(magnitude * 8.0 / 3.0, rel=1e-12, abs=0.0)
    assert cgf.Lambda(ray) == pytest.approx(0.0, abs=1e-12)
    assert cgf.Lambda(0.5 * ray) < 0.0
    assert cgf.Lambda(1.01 * ray) > 0.0


@pytest.mark.parametrize("magnitude", [1e-300, 1e300])
def test_ray_endpoint_is_distinct_from_unrestricted_support_maximiser(magnitude: float) -> None:
    """Normalisation must preserve the ray restriction, not replace it by support maximisation."""
    cgf = CumulantFunction(-0.5 * np.ones(2), np.eye(2))
    direction: np.ndarray = magnitude * np.array([1.0, 2.0])

    with np.errstate(over="raise", under="raise", invalid="raise", divide="raise"):
        ray: np.ndarray = cgf.optimal_ray_tilt(direction)

    np.testing.assert_allclose(ray, [0.6, 1.2], rtol=1e-12, atol=0.0)
    assert cgf.Lambda(ray) == pytest.approx(0.0, abs=1e-12)
    assert not np.allclose(ray, cgf.optimal_theta(direction))


@pytest.mark.parametrize("magnitude", [1e-300, 1e300])
@pytest.mark.parametrize("direction", [np.array([-1.0, -2.0]), np.array([-1.0, 1.0])])
def test_nonnegative_projected_drift_has_only_the_zero_ray_endpoint(
    magnitude: float, direction: np.ndarray
) -> None:
    """Strictly positive and zero projected drifts stay infeasible after rescaling."""
    cgf = CumulantFunction(-0.5 * np.ones(2), np.eye(2))

    with np.errstate(over="raise", under="raise", invalid="raise", divide="raise"):
        ray: np.ndarray = cgf.optimal_ray_tilt(magnitude * direction)

    np.testing.assert_array_equal(ray, np.zeros(2))


@pytest.mark.parametrize("magnitude", [1e-300, 1e300])
def test_zero_mean_has_zero_support_and_tilts_at_extreme_scales(magnitude: float) -> None:
    """With zero mean the CGF sublevel set consists only of the origin."""
    cgf = CumulantFunction(np.zeros(2), np.eye(2))
    point: np.ndarray = magnitude * np.array([1.0, -2.0])

    with np.errstate(over="raise", under="raise", invalid="raise", divide="raise"):
        assert cgf.rate_function_I(point) == 0.0
        np.testing.assert_array_equal(cgf.optimal_theta(point), np.zeros(2))
        np.testing.assert_array_equal(cgf.optimal_ray_tilt(point), np.zeros(2))


@pytest.mark.parametrize(
    "point",
    [np.array([1.0]), np.array([[1.0, 2.0]]), np.array([np.inf, 0.0]), np.array([np.nan, 0.0])],
)
def test_support_maximiser_rejects_invalid_directions(point: np.ndarray) -> None:
    """Normalisation must retain the public vector-validation contract."""
    cgf = CumulantFunction(-0.5 * np.ones(2), np.eye(2))

    with pytest.raises(ValueError, match="x must"):
        cgf.optimal_theta(point)
