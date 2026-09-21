"""Foundational tests for the public numerical API."""

from __future__ import annotations

import numpy as np
import pytest

from efficient_importance_sampling import (
    CumulantFunction,
    GapRule,
    MultidimensionalSiegmund,
    SimulationResult,
    SumIntersectionRule,
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


def test_optimal_theta_attains_constrained_rate_function() -> None:
    """The analytical tilt must lie on the CGF boundary and attain the support."""
    mean = np.array([-0.5, -0.25])
    covariance = np.array([[1.0, 0.1], [0.1, 1.5]])
    target = np.array([0.2, 0.4])
    cgf = CumulantFunction(mean, covariance)

    theta = cgf.optimal_theta(target)

    assert cgf.Lambda(theta) == pytest.approx(0.0, abs=1e-12)
    assert theta @ target == pytest.approx(cgf.rate_function_I(target))

    # The KKT condition requires the boundary normal to be parallel to x.
    gradient = cgf.grad_Lambda(theta)
    assert gradient[0] * target[1] == pytest.approx(
        gradient[1] * target[0],
        abs=1e-12,
    )


def test_rate_function_handles_degenerate_directions() -> None:
    """A zero objective or zero Gaussian drift has the zero tilt as a maximiser."""
    drifted = CumulantFunction(np.array([-0.5, -0.25]), np.eye(2))
    centred = CumulantFunction(np.zeros(2), np.eye(2))

    assert drifted.rate_function_I(np.zeros(2)) == 0.0
    np.testing.assert_array_equal(drifted.optimal_theta(np.zeros(2)), np.zeros(2))
    assert centred.rate_function_I(np.array([1.0, 2.0])) == 0.0
    np.testing.assert_array_equal(
        centred.optimal_theta(np.array([1.0, 2.0])),
        np.zeros(2),
    )


@pytest.mark.parametrize("point", [np.array([1.0]), np.array([1.0, np.nan])])
def test_rate_function_validates_target_point(point: np.ndarray) -> None:
    """Rate-function targets must match the model dimension and be finite."""
    cgf = CumulantFunction(np.zeros(2), np.eye(2))

    with pytest.raises(ValueError, match="x must"):
        cgf.rate_function_I(point)


def test_gap_rule_rejects_invalid_signal_count() -> None:
    """A gap rule needs at least one coordinate outside the signal set."""
    with pytest.raises(ValueError, match="m must satisfy"):
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


@pytest.mark.parametrize(
    ("mean", "covariance", "message"),
    [
        (
            np.array([0.0, 1.0]),
            np.array([[1.0, 0.2], [0.1, 1.0]]),
            "symmetric",
        ),
        (
            np.array([0.0, 1.0]),
            np.array([[1.0, 2.0], [2.0, 1.0]]),
            "positive definite",
        ),
        (
            np.array([0.0, np.nan]),
            np.eye(2),
            "finite",
        ),
        (
            np.array([0.0, 1.0]),
            np.eye(3),
            "dimensions",
        ),
    ],
)
def test_cumulant_function_rejects_invalid_distribution_parameters(
    mean: np.ndarray,
    covariance: np.ndarray,
    message: str,
) -> None:
    """Invalid Gaussian parameters must fail before optimisation begins."""
    with pytest.raises(ValueError, match=message):
        CumulantFunction(mean, covariance)


def test_siegmund_problem_validates_dimensions_and_boundaries() -> None:
    """The declared dimension and both stopping boundaries must be meaningful."""
    with pytest.raises(ValueError, match="dimension"):
        MultidimensionalSiegmund(
            d=3,
            ell=1.0,
            u=1.0,
            mean=np.zeros(2),
            covariance=np.eye(2),
        )

    with pytest.raises(ValueError, match="finite and positive"):
        MultidimensionalSiegmund(
            d=2,
            ell=0.0,
            u=1.0,
            mean=np.zeros(2),
            covariance=np.eye(2),
        )


@pytest.mark.parametrize("order", [0, 3])
def test_sum_intersection_rule_validates_order(order: int) -> None:
    """The order parameter must describe a non-empty proper subset."""
    with pytest.raises(ValueError, match="L must satisfy"):
        SumIntersectionRule(
            d=3,
            L=order,
            mean=-np.ones(3),
            covariance=np.eye(3),
        )


@pytest.fixture
def simple_siegmund_problem(
    monkeypatch: pytest.MonkeyPatch,
) -> MultidimensionalSiegmund:
    """Create a small simulation problem with deterministic mixture construction."""
    problem = MultidimensionalSiegmund(
        d=1,
        ell=1.0,
        u=1.0,
        mean=np.array([-0.5]),
        covariance=np.eye(1),
    )
    monkeypatch.setattr(
        problem,
        "get_feasible_mixture",
        lambda: ([np.array([0.0])], [1.0], [1]),
    )
    return problem


def test_simulation_is_reproducible_with_seeded_generator(
    simple_siegmund_problem: MultidimensionalSiegmund,
) -> None:
    """Equal generator seeds must produce equal Monte Carlo diagnostics."""
    first = simple_siegmund_problem.simulate_wrong_exit_probability(
        b=1.0,
        n_samples=50,
        rng=np.random.default_rng(2025),
    )
    second = simple_siegmund_problem.simulate_wrong_exit_probability(
        b=1.0,
        n_samples=50,
        rng=np.random.default_rng(2025),
    )

    assert first.estimate == second.estimate
    assert first.std_error == second.std_error
    assert first.relative_error == second.relative_error
    assert first.log_probability == second.log_probability
    assert first.samples_used == second.samples_used == 50


@pytest.mark.parametrize("n_samples", [0, -1, 1.5, True])
def test_simulation_rejects_invalid_sample_counts(
    simple_siegmund_problem: MultidimensionalSiegmund,
    n_samples: object,
) -> None:
    """Sample counts must be positive integers."""
    with pytest.raises(ValueError, match="positive integer"):
        simple_siegmund_problem.simulate_wrong_exit_probability(
            b=1.0,
            n_samples=n_samples,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("boundary_scale", [0.0, -1.0, np.inf, np.nan])
def test_simulation_rejects_invalid_boundary_scale(
    simple_siegmund_problem: MultidimensionalSiegmund,
    boundary_scale: float,
) -> None:
    """The simulation scale must define finite positive stopping boundaries."""
    with pytest.raises(ValueError, match="finite and positive"):
        simple_siegmund_problem.simulate_wrong_exit_probability(
            b=boundary_scale,
            n_samples=1,
        )


def test_mixture_likelihood_ratio_uses_all_components() -> None:
    """The path weight must be evaluated against the complete proposal mixture."""
    problem = MultidimensionalSiegmund(
        d=1,
        ell=1.0,
        u=1.0,
        mean=np.array([-0.5]),
        covariance=np.eye(1),
    )
    position = np.array([2.0])
    tilts = [np.array([0.0]), np.array([1.0])]
    weights = [0.25, 0.75]

    actual = problem._mixture_log_likelihood_ratio(
        position=position,
        n_steps=3,
        tilts=tilts,
        weights=weights,
    )
    expected = -np.log(
        sum(
            weight * np.exp(theta @ position - 3 * problem.cgf.Lambda(theta))
            for theta, weight in zip(tilts, weights)
        )
    )

    assert actual == pytest.approx(expected)


def test_duplicate_zero_tilts_preserve_unit_likelihood_ratio() -> None:
    """Mixture weights must not multiply an unchanged proposal distribution."""
    problem = MultidimensionalSiegmund(
        d=1,
        ell=1.0,
        u=1.0,
        mean=np.array([-0.5]),
        covariance=np.eye(1),
    )

    actual = problem._mixture_log_likelihood_ratio(
        position=np.array([3.0]),
        n_steps=5,
        tilts=[np.zeros(1), np.zeros(1)],
        weights=[0.1, 0.9],
    )

    assert actual == pytest.approx(0.0)
