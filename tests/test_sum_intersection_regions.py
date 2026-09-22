"""Analytical and independent LP checks for sum-intersection region optimisation."""

import itertools
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from scipy.optimize import linprog  # type: ignore[import-untyped]

from efficient_importance_sampling import CumulantFunction, RegionOptimizer


@pytest.mark.parametrize(
    ("magnitudes", "order", "expected"),
    [
        (np.array([10.0, 1.0, 1.0, 1.0]), 2, 3.0),
        (np.array([9.0, 2.0, 1.0, 1.0, 1.0]), 3, 2.5),
        (np.ones(4), 3, 4.0 / 3.0),
        (np.array([1.0, 0.0, 0.0]), 2, 0.0),
    ],
)
def test_ordered_objective_matches_independent_region_lp(
    magnitudes: np.ndarray, order: int, expected: float
) -> None:
    """Check all objective branches against the region definition, not a second sort formula."""
    subsets = list(itertools.combinations(range(magnitudes.size), order))
    incidence = np.zeros((len(subsets), magnitudes.size))
    for row, subset in enumerate(subsets):
        incidence[row, list(subset)] = 1.0
    result = linprog(magnitudes, A_ub=-incidence, b_ub=-np.ones(len(subsets)), bounds=(0.0, None))

    assert result.success
    actual = RegionOptimizer._sum_intersection_objective(magnitudes, order)
    assert actual == pytest.approx(expected)
    assert actual == pytest.approx(result.fun)


@pytest.mark.parametrize("indices", [[0], [0, 1], [0, 1, 2]])
def test_order_one_agrees_with_siegmund_rate(indices: list[int]) -> None:
    """For L=1, every independent unit-boundary region has rate equal to its size."""
    optimizer = RegionOptimizer(CumulantFunction(-0.5 * np.ones(3), np.eye(3)))

    theta, rate = optimizer.solve_kkt_system(indices, {"sum_intersection": {"L": 1}})

    expected = np.zeros(3)
    expected[indices] = 1.0
    np.testing.assert_allclose(theta, expected, atol=1e-6)
    assert rate == pytest.approx(float(len(indices)), rel=1e-7)


@pytest.mark.parametrize("order", [2, 3])
def test_independent_regions_have_analytical_rates(order: int) -> None:
    """Independent equal negative drifts give indicator tilts and rate |A|/L."""
    optimizer = RegionOptimizer(CumulantFunction(-0.5 * np.ones(4), np.eye(4)))

    for count in range(order, 5):
        indices = list(range(count))
        theta, rate = optimizer.solve_kkt_system(indices, {"sum_intersection": {"L": order}})
        expected = np.zeros(4)
        expected[indices] = 1.0
        np.testing.assert_allclose(theta, expected, atol=1e-6)
        assert rate == pytest.approx(count / order, rel=1e-7)
        assert optimizer.cgf.Lambda(theta) == pytest.approx(0.0, abs=1e-10)


@pytest.mark.parametrize("scale", [1e-6, 1.0, 1e6])
def test_sum_intersection_rate_scales_with_drift(scale: float) -> None:
    """Normalisation must not lose small or large positive rates."""
    optimizer = RegionOptimizer(CumulantFunction(-0.5 * scale * np.ones(3), 2.0 * np.eye(3)))

    theta, rate = optimizer.solve_kkt_system([0, 1, 2], {"sum_intersection": {"L": 2}})

    np.testing.assert_allclose(theta, 0.5 * scale * np.ones(3), rtol=1e-7, atol=0.0)
    assert rate == pytest.approx(0.75 * scale, rel=1e-7)


def test_correlated_region_matches_closed_form_support() -> None:
    """The active objective is half the total magnitude in this symmetric region."""
    rho = 0.3
    cgf = CumulantFunction(-0.5 * np.ones(4), (1.0 - rho) * np.eye(4) + rho * np.ones((4, 4)))
    offset = 0.5 / (1.0 + 3.0 * rho)
    contrast = 0.5 / np.sqrt((1.0 + 3.0 * rho) * (1.0 - rho))
    expected = np.array(
        [offset + contrast, offset + contrast, offset - contrast, offset - contrast]
    )

    theta, rate = RegionOptimizer(cgf).solve_kkt_system([0, 1], {"sum_intersection": {"L": 2}})

    np.testing.assert_allclose(theta, expected, atol=1e-6)
    assert rate == pytest.approx(2.0 * contrast, rel=1e-7)
    assert cgf.Lambda(theta) == pytest.approx(0.0, abs=1e-10)


def test_middle_ordered_tail_can_determine_the_optimum() -> None:
    """A known dual certificate exposes errors that only testing L=1 or total sums misses."""
    cgf = CumulantFunction(
        np.array([-0.3, -1.21, -0.21, -0.21, -0.21]), np.diag([0.1, 1.0, 1.0, 1.0, 1.0])
    )

    theta, rate = RegionOptimizer(cgf).solve_kkt_system(
        list(range(5)), {"sum_intersection": {"L": 3}}
    )

    # At this point the ell=2 tail has value 2.5, below the ell=1 and ell=3 tails.
    # (0, 1/2, 1/2, 1/2, 1/2) is a feasible dual vector attaining that value.
    np.testing.assert_allclose(theta, [3.0, 2.0, 1.0, 1.0, 1.0], atol=1e-5)
    assert rate == pytest.approx(2.5, rel=1e-7)
    assert cgf.Lambda(theta) == pytest.approx(0.0, abs=1e-10)


@pytest.mark.parametrize("mean", [np.zeros(3), np.ones(3)])
def test_nonnegative_orthant_drift_has_only_zero_tilt(mean: np.ndarray) -> None:
    """Zero is legitimate when no nonzero point in the orthant has a nonpositive CGF."""
    optimizer = RegionOptimizer(CumulantFunction(mean, np.eye(3)))

    theta, rate = optimizer.solve_kkt_system([0, 1, 2], {"sum_intersection": {"L": 2}})

    np.testing.assert_array_equal(theta, np.zeros(3))
    assert rate == 0.0


def test_failed_sum_intersection_candidate_keeps_solver_diagnostic() -> None:
    """Failed solves must not be silently converted to zero tilts."""
    optimizer = RegionOptimizer(CumulantFunction(-0.5 * np.ones(3), np.eye(3)))
    result = SimpleNamespace(success=False, message="iteration limit", x=np.zeros(7))

    with patch("efficient_importance_sampling.core.opt.minimize", return_value=result):
        with pytest.raises(RuntimeError, match="iteration limit"):
            optimizer.solve_kkt_system([0, 1], {"sum_intersection": {"L": 2}})


def test_dual_certificate_can_override_line_search_status() -> None:
    """A valid mathematical certificate establishes optimality despite a solver warning."""
    optimizer = RegionOptimizer(CumulantFunction(-0.5 * np.ones(3), np.eye(3)))
    result = SimpleNamespace(
        success=False,
        message="line-search failure",
        x=np.array([2.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0]),
    )

    with patch("efficient_importance_sampling.core.opt.minimize", return_value=result):
        theta, rate = optimizer.solve_kkt_system([0, 1], {"sum_intersection": {"L": 2}})

    np.testing.assert_allclose(theta, [1.0, 1.0, 0.0])
    assert rate == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("candidate", "message"),
    [
        (np.zeros(3), "invalid candidate"),
        (np.full(7, np.nan), "invalid candidate"),
        (np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), "infeasible candidate"),
        (np.array([10.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0]), "infeasible candidate"),
        (np.array([2.0, 2.0, 0.0, 3.0, 1.0, 0.0, 2.0]), "infeasible candidate"),
        (np.zeros(7), "zero tilt"),
        # Gaussian boundary and linear constraints hold, but the rate is suboptimal.
        (np.array([2.4, 1.2, 0.0, 1.2, 1.2, 0.0, 1.2]), "optimality check"),
    ],
)
def test_sum_intersection_success_requires_mathematical_validation(
    candidate: np.ndarray, message: str
) -> None:
    """Check feasibility and the dual rate bound independently of the success flag."""
    optimizer = RegionOptimizer(CumulantFunction(-0.5 * np.ones(3), np.eye(3)))
    result = SimpleNamespace(success=True, x=candidate)

    with patch("efficient_importance_sampling.core.opt.minimize", return_value=result):
        with pytest.raises(RuntimeError, match=message):
            optimizer.solve_kkt_system([0, 1], {"sum_intersection": {"L": 2}})


@pytest.mark.parametrize("order", [0, 3, 1.5, True])
def test_sum_intersection_rejects_invalid_order(order: int | float) -> None:
    """The order must be an integer strictly between zero and the dimension."""
    optimizer = RegionOptimizer(CumulantFunction(-np.ones(3), np.eye(3)))

    with pytest.raises(ValueError, match="L must be an integer"):
        optimizer.solve_kkt_system([0, 1, 2], {"sum_intersection": {"L": order}})


def test_sum_intersection_region_must_have_at_least_l_coordinates() -> None:
    """A wrong-exit region requires at least L positive coordinates."""
    optimizer = RegionOptimizer(CumulantFunction(-np.ones(3), np.eye(3)))

    with pytest.raises(ValueError, match="at least L coordinates"):
        optimizer.solve_kkt_system([0], {"sum_intersection": {"L": 2}})


def test_unknown_region_type_is_not_swallowed_by_a_fallback() -> None:
    """Unsupported region definitions must fail explicitly."""
    optimizer = RegionOptimizer(CumulantFunction(-np.ones(3), np.eye(3)))

    with pytest.raises(ValueError, match="Unknown constraint type"):
        optimizer.solve_kkt_system([0], {"unknown": True})
