"""Foundational tests for the public numerical API."""

from __future__ import annotations

import numpy as np
import pytest

from efficient_importance_sampling import (
    CumulantFunction,
    GapRule,
    SimulationResult,
)


def test_cumulant_function_matches_gaussian_closed_form() -> None:
    """The CGF must equal the multivariate Gaussian analytical expression."""
    mean = np.array([-0.5, 0.25])
    covariance = np.array([[1.0, 0.2], [0.2, 2.0]])
    theta = np.array([0.4, -0.3])
    cgf = CumulantFunction(mean, covariance)

    expected = mean @ theta + 0.5 * theta @ covariance @ theta

    assert cgf.Lambda(theta) == pytest.approx(expected)
    np.testing.assert_allclose(cgf.grad_Lambda(theta), mean + covariance @ theta)
    np.testing.assert_allclose(cgf.hessian_Lambda(theta), covariance)


def test_rate_function_is_zero_at_the_mean() -> None:
    """The Gaussian rate function reaches its minimum at the mean."""
    mean = np.array([-0.5, -0.25])
    covariance = np.array([[1.0, 0.1], [0.1, 1.5]])
    cgf = CumulantFunction(mean, covariance)

    assert cgf.rate_function_I(mean) == pytest.approx(0.0)


def test_optimal_theta_maps_back_to_requested_point() -> None:
    """The exponential tilt should map the Gaussian mean to the target point."""
    mean = np.array([-0.5, -0.25])
    covariance = np.array([[1.0, 0.1], [0.1, 1.5]])
    target = np.array([0.2, 0.4])
    cgf = CumulantFunction(mean, covariance)

    theta = cgf.optimal_theta(target)

    np.testing.assert_allclose(cgf.grad_Lambda(theta), target)


def test_gap_rule_rejects_invalid_signal_count() -> None:
    """A gap rule needs at least one coordinate outside the signal set."""
    with pytest.raises(ValueError, match="m must be less than d"):
        GapRule(
            d=2,
            m=2,
            mean=np.array([0.5, 0.5]),
            covariance=np.eye(2),
        )


def test_simulation_result_is_a_typed_value_object() -> None:
    """Simulation diagnostics should remain available as structured data."""
    result = SimulationResult(
        estimate=0.01,
        std_error=0.001,
        relative_error=0.1,
        samples_used=1_000,
        computation_time=0.2,
        log_probability=float(np.log(0.01)),
    )

    assert result.samples_used == 1_000
    assert result.relative_error == pytest.approx(0.1)
