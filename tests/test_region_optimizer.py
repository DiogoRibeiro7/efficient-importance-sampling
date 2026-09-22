"""Analytical and failure regressions for Gaussian Siegmund region optimisation."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from efficient_importance_sampling import (
    CumulantFunction,
    MultidimensionalSiegmund,
    RegionOptimizer,
)


@pytest.mark.parametrize("mean", [-1e-6, -0.5, -1e6])
def test_one_dimensional_region_recovers_positive_root(mean: float) -> None:
    """Drift scaling must preserve the known root, including very small drifts."""
    problem = MultidimensionalSiegmund(1, 2.0, 3.0, np.array([mean]), np.array([[2.0]]))

    theta, rate = problem.compute_optimal_tilts()[1]

    np.testing.assert_allclose(theta, [-mean], rtol=1e-7, atol=0.0)
    assert rate == pytest.approx(-3.0 * mean, rel=1e-7)


def test_all_independent_regions_have_known_positive_rates() -> None:
    """Equal boundaries and independent equal drifts give indicator-vector tilts."""
    problem = MultidimensionalSiegmund(3, 1.0, 1.0, -0.5 * np.ones(3), np.eye(3))

    regions = problem.compute_optimal_tilts()

    assert len(regions) == 7
    for index, (theta, rate) in regions.items():
        expected = np.array([float(bool(index & (1 << k))) for k in range(3)])
        np.testing.assert_allclose(theta, expected, atol=1e-6)
        assert rate == pytest.approx(float(np.sum(expected)), rel=1e-7)
        assert problem.cgf.Lambda(theta) == pytest.approx(0.0, abs=1e-10)


@pytest.mark.parametrize("boundary_scale", [1.0, 20.0])
def test_asymmetric_boundaries_have_known_interior_solution(boundary_scale: float) -> None:
    """The optimum depends on the ratio of the boundaries, while the rate scales."""
    problem = MultidimensionalSiegmund(
        2, 2.0 * boundary_scale, boundary_scale, -0.5 * np.ones(2), np.eye(2)
    )

    theta, rate = problem.compute_optimal_tilts()[1]

    np.testing.assert_allclose(theta, [0.5 + np.sqrt(0.1), 0.5 - 2.0 * np.sqrt(0.1)], atol=1e-6)
    assert rate == pytest.approx(boundary_scale * (np.sqrt(2.5) - 0.5), rel=1e-7)


def test_correlated_region_matches_analytical_solution() -> None:
    """A correlated Gaussian requires tilting the complementary coordinate too."""
    problem = MultidimensionalSiegmund(
        2, 1.0, 1.0, -0.5 * np.ones(2), np.array([[1.0, 0.6], [0.6, 1.0]])
    )

    theta, rate = problem.compute_optimal_tilts()[1]

    np.testing.assert_allclose(theta, [0.9375, -0.3125], atol=1e-6)
    assert rate == pytest.approx(1.25, rel=1e-7)
    assert problem.cgf.Lambda(theta) == pytest.approx(0.0, abs=1e-10)


def test_region_with_active_sign_bound_is_not_unconstrained_support() -> None:
    """The second tilt must stay at zero when the unconstrained optimum is positive."""
    problem = MultidimensionalSiegmund(2, 1.0, 1.0, np.array([-0.5, -2.0]), np.eye(2))

    theta, rate = problem.compute_optimal_tilts()[1]

    np.testing.assert_allclose(theta, [1.0, 0.0], atol=1e-6)
    assert rate == pytest.approx(1.0, rel=1e-7)


@pytest.mark.parametrize("mean", [np.zeros(2), np.array([0.5, -0.5])])
def test_zero_is_returned_when_it_is_the_only_feasible_tilt(mean: np.ndarray) -> None:
    """A zero rate is valid when all directions in the orthant have nonnegative drift."""
    optimizer = RegionOptimizer(CumulantFunction(mean, np.eye(2)))

    theta, rate = optimizer.solve_kkt_system([0], {"siegmund": {"u": 1.0, "ell": 1.0}})

    np.testing.assert_array_equal(theta, np.zeros(2))
    assert rate == 0.0


def test_feasible_mixture_uses_solved_region_tilts() -> None:
    """The public mixture must not include a spurious zero component in one dimension."""
    problem = MultidimensionalSiegmund(1, 1.0, 1.0, np.array([-0.5]), np.eye(1))

    tilts, weights, indices = problem.get_feasible_mixture()

    np.testing.assert_allclose(tilts, [[1.0], [1.0]], atol=1e-7)
    assert weights == [0.5, 0.5]
    assert indices == [1, -1]


def test_solver_failure_is_not_replaced_with_a_zero_tilt() -> None:
    """A failed numerical solve must stop construction with its diagnostic message."""
    problem = MultidimensionalSiegmund(1, 1.0, 1.0, np.array([-0.5]), np.eye(1))
    failed = SimpleNamespace(success=False, message="iteration limit", x=np.zeros(1))

    with patch("efficient_importance_sampling.core.opt.minimize", return_value=failed):
        with pytest.raises(RuntimeError, match="iteration limit"):
            problem.get_feasible_mixture()


def test_valid_optimum_is_accepted_despite_line_search_status() -> None:
    """SLSQP may flag a line-search failure at a point satisfying all KKT conditions."""
    problem = MultidimensionalSiegmund(1, 1.0, 1.0, np.array([-0.5]), np.eye(1))
    result = SimpleNamespace(success=False, message="line-search failure", x=np.array([2.0]))

    with patch("efficient_importance_sampling.core.opt.minimize", return_value=result):
        theta, rate = problem.compute_optimal_tilts()[1]

    np.testing.assert_allclose(theta, [1.0])
    assert rate == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("candidate", "message"),
    [
        (np.array([np.nan, 0.0]), "invalid tilt"),
        (np.array([1.0]), "invalid tilt"),
        (np.array([-0.1, 0.0]), "infeasible tilt"),
        (np.array([10.0, 0.0]), "infeasible tilt"),
        (np.zeros(2), "zero tilt"),
        (np.array([np.sqrt(2.0), 0.0]), "KKT residual"),
    ],
)
def test_success_flag_does_not_override_mathematical_checks(
    candidate: np.ndarray, message: str
) -> None:
    """Even a feasible boundary point is rejected if it is not optimal for the region."""
    problem = MultidimensionalSiegmund(2, 2.0, 1.0, -0.5 * np.ones(2), np.eye(2))
    result = SimpleNamespace(success=True, x=candidate)

    with patch("efficient_importance_sampling.core.opt.minimize", return_value=result):
        with pytest.raises(RuntimeError, match=message):
            problem.get_feasible_mixture()


def test_failed_region_does_not_cache_an_incomplete_mixture() -> None:
    """Retrying after a later region fails must recompute every requested region."""
    problem = MultidimensionalSiegmund(2, 1.0, 1.0, -0.5 * np.ones(2), np.eye(2))

    with patch.object(
        problem.optimizer,
        "solve_kkt_system",
        side_effect=[(np.array([1.0, 0.0]), 1.0), RuntimeError("second region failed")],
    ):
        with pytest.raises(RuntimeError, match="second region failed"):
            problem.compute_optimal_tilts()

    assert set(problem.compute_optimal_tilts()) == {1, 2, 3}


@pytest.mark.parametrize("indices", [[], [0, 0], [-1], [2]])
def test_region_rejects_invalid_indices(indices: list[int]) -> None:
    """A Siegmund region is indexed by a nonempty subset of the model coordinates."""
    optimizer = RegionOptimizer(CumulantFunction(-np.ones(2), np.eye(2)))

    with pytest.raises(ValueError, match="distinct valid coordinate indices"):
        optimizer.solve_kkt_system(indices, {"siegmund": {"u": 1.0, "ell": 1.0}})


@pytest.mark.parametrize("upper", [0.0, np.inf])
def test_region_solver_validates_boundaries(upper: float) -> None:
    """Direct calls to the public optimizer must validate their own boundaries."""
    optimizer = RegionOptimizer(CumulantFunction(-np.ones(2), np.eye(2)))

    with pytest.raises(ValueError, match="finite and positive"):
        optimizer.solve_kkt_system([0], {"siegmund": {"u": upper, "ell": 1.0}})
