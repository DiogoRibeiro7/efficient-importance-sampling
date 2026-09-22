"""Analytical checks for the complete sum-intersection proposal family."""

import itertools
import math
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import pytest

from efficient_importance_sampling import (
    CumulantFunction,
    MultidimensionalSiegmund,
    RegionOptimizer,
    SumIntersectionRule,
)


@pytest.mark.parametrize("order", [1, 2, 3])
@pytest.mark.parametrize("rho", [0.0, 0.4])
def test_exchangeable_auxiliary_matches_equation_43(order: int, rho: float) -> None:
    """Equal selected tilts have the known Gaussian root on each supported subset."""
    cgf = CumulantFunction(-0.5 * np.ones(5), (1.0 - rho) * np.eye(5) + rho * np.ones((5, 5)))
    indices = [0, 2, 4][:order]
    expected = np.zeros(5)
    expected[indices] = 1.0 / (1.0 + (order - 1) * rho)

    theta, value = RegionOptimizer(cgf).solve_sum_intersection_auxiliary(indices)

    np.testing.assert_allclose(theta, expected, atol=1e-7)
    np.testing.assert_array_equal(theta[[k for k in range(5) if k not in indices]], 0.0)
    assert value == pytest.approx(float(np.min(expected[indices])), rel=1e-7)
    assert cgf.Lambda(theta) == pytest.approx(0.0, abs=1e-10)


def test_auxiliary_coordinates_need_not_be_equal() -> None:
    """Unequal drifts can require a tilt above the minimum in one selected coordinate."""
    cgf = CumulantFunction(np.array([-5.0, -0.5, -0.5, -0.7]), np.eye(4))
    expected_minimum = (1.0 + np.sqrt(51.0)) / 2.0

    theta, value = RegionOptimizer(cgf).solve_sum_intersection_auxiliary([0, 1, 2])

    # The optimum has gradient zero in coordinate 0 and positive in coordinates 1 and 2.
    np.testing.assert_allclose(theta, [5.0, expected_minimum, expected_minimum, 0.0], atol=1e-5)
    assert value == pytest.approx(expected_minimum, rel=1e-7)
    assert value > 4.0  # The largest feasible equal-coordinate ray has minimum 4.
    assert cgf.Lambda(theta) == pytest.approx(0.0, abs=1e-10)


def test_auxiliary_depends_only_on_its_principal_gaussian_submodel() -> None:
    """Outside drift and cross-covariances vanish when the outside tilt is exactly zero."""
    covariance = np.array(
        [[2.0, 0.0, 0.25, 0.0], [0.0, 1.0, 0.0, 0.0], [0.25, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )
    changed_covariance = covariance.copy()
    changed_covariance[0, 1] = changed_covariance[1, 0] = 0.1
    changed_covariance[2, 3] = changed_covariance[3, 2] = -0.2
    original = RegionOptimizer(CumulantFunction(-0.5 * np.ones(4), covariance))
    changed = RegionOptimizer(
        CumulantFunction(np.array([-0.5, 100.0, -0.5, -200.0]), changed_covariance)
    )

    expected_tilt, expected_value = original.solve_sum_intersection_auxiliary([0, 2])
    theta, value = changed.solve_sum_intersection_auxiliary([2, 0])

    np.testing.assert_allclose(theta, expected_tilt, atol=1e-7)
    np.testing.assert_array_equal(theta[[1, 3]], 0.0)
    assert value == pytest.approx(expected_value, rel=1e-7)


@pytest.mark.parametrize("indices", [[0], [0, 1]])
def test_auxiliary_returns_zero_if_no_nonzero_supported_tilt_is_feasible(
    indices: list[int],
) -> None:
    """Nonnegative selected drifts make the origin the only feasible supported tilt."""
    optimizer = RegionOptimizer(CumulantFunction(np.array([0.5, 0.0, -0.5]), np.eye(3)))

    theta, value = optimizer.solve_sum_intersection_auxiliary(indices)

    np.testing.assert_array_equal(theta, np.zeros(3))
    assert value == 0.0


def test_mixture_contains_region_family_then_supported_auxiliary_family() -> None:
    """Both families must have the paper's values and lexicographic subset ordering."""
    rho = 0.3
    problem = SumIntersectionRule(
        4, 2, -0.5 * np.ones(4), (1.0 - rho) * np.eye(4) + rho * np.ones((4, 4))
    )
    subsets = list(itertools.combinations(range(4), 2))
    offset = 0.5 / (1.0 + 3.0 * rho)
    contrast = 0.5 / np.sqrt((1.0 + 3.0 * rho) * (1.0 - rho))

    tilts, weights = problem.compute_feasible_mixture()

    assert len(tilts) == len(weights) == 12
    np.testing.assert_allclose(weights, np.full(12, 1.0 / 12.0))
    assert sum(weights) == pytest.approx(1.0)
    for position, subset in enumerate(subsets):
        indices = list(subset)
        expected_region = np.full(4, offset - contrast)
        expected_region[indices] = offset + contrast
        expected_auxiliary = np.zeros(4)
        expected_auxiliary[indices] = 1.0 / (1.0 + rho)
        np.testing.assert_allclose(tilts[position], expected_region, atol=1e-6)
        np.testing.assert_allclose(tilts[len(subsets) + position], expected_auxiliary, atol=1e-7)
    for theta in tilts:
        assert problem.cgf.Lambda(theta) == pytest.approx(0.0, abs=1e-10)


def test_auxiliary_family_is_complete_above_one_hundred_subsets() -> None:
    """A combinatorial family of 120 subsets must retain every auxiliary component."""
    problem = SumIntersectionRule(10, 3, -0.5 * np.ones(10), np.eye(10))
    count = math.comb(10, 3)

    tilts, weights = problem.compute_feasible_mixture()

    assert count == 120
    assert len(tilts) == len(weights) == 2 * count
    for position, subset in enumerate(itertools.combinations(range(10), 3)):
        expected = np.zeros(10)
        expected[list(subset)] = 1.0
        np.testing.assert_allclose(tilts[count + position], expected, atol=1e-7)
    np.testing.assert_allclose(weights, np.full(2 * count, 1.0 / (2 * count)))


def test_order_one_mixture_agrees_with_unit_boundary_siegmund() -> None:
    """At L=1, both proposal families reduce to the existing Siegmund construction."""
    mean = -0.5 * np.ones(3)
    covariance = 0.8 * np.eye(3) + 0.2 * np.ones((3, 3))
    sum_rule = SumIntersectionRule(3, 1, mean, covariance)
    siegmund = MultidimensionalSiegmund(3, 1.0, 1.0, mean, covariance)

    tilts, weights = sum_rule.compute_feasible_mixture()
    expected_tilts, expected_weights, _ = siegmund.get_feasible_mixture()

    np.testing.assert_allclose(tilts, expected_tilts, atol=1e-7)
    np.testing.assert_allclose(weights, expected_weights)


def test_auxiliary_failure_does_not_return_a_partial_mixture() -> None:
    """A failed auxiliary solve must abort construction instead of dropping a component."""
    problem = SumIntersectionRule(3, 2, -0.5 * np.ones(3), np.eye(3))

    with patch.object(
        problem.optimizer,
        "solve_sum_intersection_auxiliary",
        side_effect=RuntimeError("auxiliary failed"),
    ):
        with pytest.raises(RuntimeError, match="auxiliary failed"):
            problem.compute_feasible_mixture()


def test_auxiliary_solver_validates_the_full_order_optimum() -> None:
    """A feasible unequal two-coordinate tilt is not necessarily the max-min solution."""
    optimizer = RegionOptimizer(CumulantFunction(-0.5 * np.ones(3), np.eye(3)))
    result = SimpleNamespace(success=True, x=np.array([2.4, 1.2, 1.2, 1.2, 1.2]))

    with patch("efficient_importance_sampling.core.opt.minimize", return_value=result):
        with pytest.raises(RuntimeError, match="optimality check"):
            optimizer.solve_sum_intersection_auxiliary([0, 2])


@pytest.mark.parametrize("indices", [[], [-1], [3], [0, 0]])
def test_auxiliary_solver_rejects_invalid_subsets(indices: list[int]) -> None:
    """The public auxiliary solver must validate its coordinate support."""
    optimizer = RegionOptimizer(CumulantFunction(-np.ones(3), np.eye(3)))

    with pytest.raises(ValueError, match="distinct valid coordinate indices"):
        optimizer.solve_sum_intersection_auxiliary(indices)
