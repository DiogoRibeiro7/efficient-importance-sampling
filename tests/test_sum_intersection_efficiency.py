"""Analytical and failure-path checks for the numerical condition (H-SI)."""

import itertools
import math
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from efficient_importance_sampling import (
    CumulantFunction,
    RegionOptimizer,
    SumIntersectionEfficiencyResult,
    SumIntersectionRule,
)


@pytest.mark.parametrize("order", [1, 2, 3])
@pytest.mark.parametrize("rho", [0.0, 0.4])
def test_coverage_matches_exchangeable_gaussian_solution(order: int, rho: float) -> None:
    """Equation (44) has a known tilt and objective in the exchangeable model."""
    covariance = (1.0 - rho) * np.eye(5) + rho * np.ones((5, 5))
    cgf = CumulantFunction(-0.5 * np.ones(5), covariance)
    indices = [4, 1, 3, 0][: order + 1]
    expected = np.zeros(5)
    expected[indices] = 1.0 / (1.0 + order * rho)

    theta, value = RegionOptimizer(cgf).solve_sum_intersection_coverage(indices, order)

    np.testing.assert_allclose(theta, expected, atol=1e-7)
    np.testing.assert_array_equal(theta[[k for k in range(5) if k not in indices]], 0.0)
    assert value == pytest.approx((order + 1.0) / (order * (1.0 + order * rho)), rel=1e-7)
    assert cgf.Lambda(theta) == pytest.approx(0.0, abs=1e-10)


def test_coverage_with_unequal_drifts_is_not_an_equal_coordinate_ray() -> None:
    """For L=1, the supported linear objective has an ellipsoid support solution."""
    cgf = CumulantFunction(np.array([-5.0, -0.5, -0.7]), np.eye(3))
    offset = np.sqrt(101.0 / 8.0)
    expected = np.array([5.0 + offset, 0.5 + offset, 0.0])

    theta, value = RegionOptimizer(cgf).solve_sum_intersection_coverage([0, 1], 1)

    np.testing.assert_allclose(theta, expected, atol=1e-6)
    assert value == pytest.approx(float(np.sum(expected)), rel=1e-7)
    assert cgf.Lambda(theta) == pytest.approx(0.0, abs=1e-10)


@pytest.mark.parametrize("rho, status", [(0.0, "satisfied"), (0.3, "not_satisfied")])
def test_condition_matches_closed_form_on_both_sides(rho: float, status: str) -> None:
    """The four-dimensional L=2 model has explicit region and coverage rates."""
    problem = SumIntersectionRule(
        4, 2, -0.5 * np.ones(4), (1.0 - rho) * np.eye(4) + rho * np.ones((4, 4))
    )
    expected_rate = 1.0 / np.sqrt((1.0 + 3.0 * rho) * (1.0 - rho))
    expected_coverage = 1.0 / (1.0 + rho) + 1.5 / (1.0 + 2.0 * rho)

    result = problem.check_efficiency_condition()

    assert isinstance(result, SumIntersectionEfficiencyResult)
    assert result.minimum_region_rate == pytest.approx(expected_rate, rel=1e-7)
    assert result.required_bound == pytest.approx(2.0 * expected_rate, rel=1e-7)
    assert result.coverage_bound == pytest.approx(expected_coverage, rel=1e-7)
    assert result.margin == pytest.approx(expected_coverage - 2.0 * expected_rate, abs=1e-7)
    assert result.status == status
    assert len(result.critical_region) == len(result.weakest_subset) == 2
    assert result.extra_coordinate not in result.weakest_subset


def test_equality_is_reported_as_borderline() -> None:
    """A root of the analytical H-SI margin must not be presented as a certificate."""
    # Root of 1/(1+rho) + 3/(2+4*rho) - 2/sqrt((1+3*rho)*(1-rho)).
    rho = 0.28316476700388776
    problem = SumIntersectionRule(
        4, 2, -0.5 * np.ones(4), (1.0 - rho) * np.eye(4) + rho * np.ones((4, 4))
    )

    result = problem.check_efficiency_condition()

    assert result.margin == pytest.approx(0.0, abs=1e-7)
    assert result.status == "borderline"


def test_requested_tolerance_can_make_a_clear_margin_borderline() -> None:
    """Comparison tolerances are reported and applied without changing the rates."""
    problem = SumIntersectionRule(3, 2, -0.5 * np.ones(3), np.eye(3))

    result = problem.check_efficiency_condition(rtol=0.1, atol=0.5)

    assert result.coverage_bound == pytest.approx(2.5)
    assert result.required_bound == pytest.approx(2.0)
    assert result.margin == pytest.approx(0.5)
    assert result.tolerance == pytest.approx(0.75)
    assert result.status == "borderline"


def test_diagnostic_uses_nested_pairs_instead_of_independent_minima() -> None:
    """Distinct synthetic values isolate the A subset B restriction in (H-SI)."""
    problem = SumIntersectionRule(4, 2, -0.5 * np.ones(4), np.eye(4))
    pairs = list(itertools.combinations(range(4), 2))
    # The minimum z belongs to (0, 1), absent from the minimum-s subset (0, 2, 3).
    z_values: dict[tuple[int, ...], float] = {
        pair: float(index + 10) for index, pair in enumerate(pairs)
    }
    z_values[(0, 1)] = 1.0
    s_values: dict[tuple[int, ...], float] = {
        (0, 1, 2): 20.0,
        (0, 1, 3): 30.0,
        (0, 2, 3): 0.1,
        (1, 2, 3): 0.2,
    }
    rate_values: dict[tuple[int, ...], float] = {pair: 3.0 for pair in pairs}
    rate_values[(1, 3)] = 2.0

    def region(indices: list[int], constraints: object) -> tuple[np.ndarray, float]:
        return np.zeros(4), rate_values[tuple(indices)]

    def auxiliary(indices: list[int]) -> tuple[np.ndarray, float]:
        return np.zeros(4), z_values[tuple(indices)]

    def coverage(indices: list[int], order: int) -> tuple[np.ndarray, float]:
        return np.zeros(4), s_values[tuple(indices)]

    with (
        patch.object(problem.optimizer, "solve_kkt_system", side_effect=region),
        patch.object(problem.optimizer, "solve_sum_intersection_auxiliary", side_effect=auxiliary),
        patch.object(problem.optimizer, "solve_sum_intersection_coverage", side_effect=coverage),
    ):
        result = problem.check_efficiency_condition()

    assert result.critical_region == (1, 3)
    assert result.weakest_subset == (0, 2)
    assert result.extra_coordinate == 3
    assert result.coverage_bound == pytest.approx(11.1)
    assert result.margin == pytest.approx(7.1)
    assert result.status == "satisfied"


def test_every_large_subset_is_solved_once_above_one_hundred() -> None:
    """The independent model checks both the value and complete 126-subset coverage."""
    problem = SumIntersectionRule(9, 3, -0.5 * np.ones(9), np.eye(9))
    with patch.object(
        problem.optimizer,
        "solve_sum_intersection_coverage",
        wraps=problem.optimizer.solve_sum_intersection_coverage,
    ) as coverage:
        result = problem.check_efficiency_condition()

    assert coverage.call_count == math.comb(9, 4) == 126
    supports = [tuple(call.args[0]) for call in coverage.call_args_list]
    assert supports == list(itertools.combinations(range(9), 4))
    assert result.minimum_region_rate == pytest.approx(1.0, rel=1e-7)
    assert result.coverage_bound == pytest.approx(7.0 / 3.0, rel=1e-7)
    assert result.margin == pytest.approx(1.0 / 3.0, rel=1e-7)
    assert result.status == "satisfied"


@pytest.mark.parametrize("order", [1, 2, 3])
def test_independent_model_diagnostic_at_every_valid_order(order: int) -> None:
    """Cover L=1 and L=d-1, where the coverage support includes every coordinate."""
    problem = SumIntersectionRule(4, order, -0.5 * np.ones(4), np.eye(4))

    result = problem.check_efficiency_condition()

    assert result.minimum_region_rate == pytest.approx(1.0, rel=1e-7)
    assert result.coverage_bound == pytest.approx(2.0 + 1.0 / order, rel=1e-7)
    assert result.margin == pytest.approx(1.0 / order, rel=1e-7)
    assert result.status == "satisfied"


@pytest.mark.parametrize("mean", [np.zeros(3), np.array([-0.5, -0.5, 0.1])])
def test_condition_rejects_models_outside_negative_drift_assumption(mean: np.ndarray) -> None:
    problem = SumIntersectionRule(3, 2, mean, np.eye(3))

    with pytest.raises(ValueError, match="strictly negative"):
        problem.check_efficiency_condition()


@pytest.mark.parametrize("name", ["rtol", "atol"])
@pytest.mark.parametrize("value", [-1.0, np.nan, np.inf, True, "0.1"])
def test_invalid_comparison_tolerances_fail_before_optimisation(name: str, value: Any) -> None:
    problem = SumIntersectionRule(3, 2, -0.5 * np.ones(3), np.eye(3))
    with patch.object(problem.optimizer, "solve_kkt_system") as solver:
        with pytest.raises(ValueError, match=f"{name} must be a finite nonnegative"):
            problem.check_efficiency_condition(**{name: value})
    solver.assert_not_called()


@pytest.mark.parametrize(
    "method",
    ["solve_kkt_system", "solve_sum_intersection_auxiliary", "solve_sum_intersection_coverage"],
)
def test_failed_optimisation_does_not_return_a_diagnostic(method: str) -> None:
    problem = SumIntersectionRule(3, 2, -0.5 * np.ones(3), np.eye(3))
    with patch.object(problem.optimizer, method, side_effect=RuntimeError("invalid candidate")):
        with pytest.raises(RuntimeError, match="invalid candidate"):
            problem.check_efficiency_condition()


@pytest.mark.parametrize(
    "method",
    ["solve_kkt_system", "solve_sum_intersection_auxiliary", "solve_sum_intersection_coverage"],
)
def test_nonfinite_optimal_values_cannot_produce_a_positive_result(method: str) -> None:
    problem = SumIntersectionRule(3, 2, -0.5 * np.ones(3), np.eye(3))
    with patch.object(problem.optimizer, method, return_value=(np.zeros(3), np.nan)):
        with pytest.raises(RuntimeError, match="finite positive optimal values"):
            problem.check_efficiency_condition()


@pytest.mark.parametrize("order", [0, 4, 1.5, True])
def test_coverage_rejects_invalid_orders(order: Any) -> None:
    optimizer = RegionOptimizer(CumulantFunction(-np.ones(4), np.eye(4)))
    with pytest.raises(ValueError, match="L must be an integer"):
        optimizer.solve_sum_intersection_coverage([0, 1, 2], order)


@pytest.mark.parametrize("indices", [[], [0, 0, 1], [0, 1, 4]])
def test_coverage_rejects_invalid_indices(indices: list[int]) -> None:
    optimizer = RegionOptimizer(CumulantFunction(-np.ones(4), np.eye(4)))
    with pytest.raises(ValueError, match="distinct valid coordinate indices"):
        optimizer.solve_sum_intersection_coverage(indices, 2)


def test_coverage_rejects_wrong_support_size() -> None:
    optimizer = RegionOptimizer(CumulantFunction(-np.ones(4), np.eye(4)))
    with pytest.raises(ValueError, match="exactly L\\+1"):
        optimizer.solve_sum_intersection_coverage([0, 1], 2)
