"""Analytical regressions for balanced Gaussian gap-region tilts."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from efficient_importance_sampling import CumulantFunction, GapRule, RegionOptimizer


@pytest.mark.parametrize("drift_scale", [1e-6, 1.0, 1e6])
def test_two_coordinate_region_matches_difference_root(drift_scale: float) -> None:
    """A two-coordinate gap reduces to the CGF of their Gaussian difference."""
    cgf = CumulantFunction(drift_scale * np.array([0.5, -0.5]), np.array([[2.0, 0.5], [0.5, 1.0]]))

    theta, rate = RegionOptimizer(cgf).solve_kkt_system([1], {"gap": True})

    # Difference variance is 2 + 1 - 2*0.5 = 2, so the positive root is drift_scale.
    np.testing.assert_allclose(theta, drift_scale * np.array([-1.0, 1.0]), rtol=1e-7, atol=0.0)
    assert rate == pytest.approx(drift_scale, rel=1e-7)
    assert np.sum(theta) == pytest.approx(0.0, abs=1e-10)


@pytest.mark.parametrize("rho", [0.0, 0.3, 0.8])
def test_exchangeable_gap_mixture_has_known_region_components(rho: float) -> None:
    """In the exchangeable model, each region tilt equals its signal-noise pair tilt."""
    problem = GapRule(
        4, 2, np.array([0.5, 0.5, -0.5, -0.5]), (1.0 - rho) * np.eye(4) + rho * np.ones((4, 4))
    )
    expected = np.array(
        [[-1.0, 0.0, 1.0, 0.0], [-1.0, 0.0, 0.0, 1.0], [0.0, -1.0, 1.0, 0.0], [0.0, -1.0, 0.0, 1.0]]
    ) / (1.0 - rho)

    tilts, weights = problem.compute_feasible_mixture()

    assert len(tilts) == len(weights) == 8
    np.testing.assert_allclose(tilts[:4], expected, atol=1e-6)
    np.testing.assert_allclose(tilts[4:], expected, atol=1e-12)
    np.testing.assert_allclose(weights, np.full(8, 0.125))
    for theta in tilts:
        assert np.sum(theta) == pytest.approx(0.0, abs=1e-10)
        assert problem.cgf.Lambda(theta) == pytest.approx(0.0, abs=1e-10)


@pytest.mark.parametrize(
    ("common_drift", "common_variance"), [(0.0, 0.0), (100.0, 0.0), (-100.0, 3.0), (0.0, 20.0)]
)
def test_gap_region_is_invariant_to_common_drift_and_noise(
    common_drift: float, common_variance: float
) -> None:
    """Common increments cancel from all gaps and must not change the optimal tilt."""
    cgf = CumulantFunction(
        np.array([1.0, 0.0, -1.0]) + common_drift,
        np.eye(3) + common_variance * np.ones((3, 3)),
    )
    expected = np.array([-1.0 - 2.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0), 1.0 + 1.0 / np.sqrt(3.0)])

    theta, rate = RegionOptimizer(cgf).solve_kkt_system([1, 2], {"gap": True})

    # The optimum is the Gaussian support point on the zero-sum subspace.
    np.testing.assert_allclose(theta, expected, atol=1e-6)
    assert rate == pytest.approx(1.0 + 2.0 / np.sqrt(3.0), rel=1e-7)
    assert cgf.Lambda(theta) == pytest.approx(0.0, abs=1e-10)


def test_gap_region_respects_coordinate_permutations() -> None:
    """Nonconsecutive selected coordinates must produce the permuted analytical tilt."""
    permutation = np.array([2, 0, 1])
    cgf = CumulantFunction(np.array([1.0, 0.0, -1.0])[permutation], np.eye(3))
    expected = np.array([-1.0 - 2.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0), 1.0 + 1.0 / np.sqrt(3.0)])

    theta, rate = RegionOptimizer(cgf).solve_kkt_system([0, 2], {"gap": True})

    np.testing.assert_allclose(theta, expected[permutation], atol=1e-6)
    assert rate == pytest.approx(1.0 + 2.0 / np.sqrt(3.0), rel=1e-7)


@pytest.mark.parametrize("mean", [np.array([1.0, 0.0, -1.0]), np.ones(3)])
def test_nonnegative_pairwise_drifts_have_only_the_zero_tilt(mean: np.ndarray) -> None:
    """When the selected coordinate has the largest drift, no positive tilt is feasible."""
    optimizer = RegionOptimizer(CumulantFunction(mean, np.eye(3)))

    theta, rate = optimizer.solve_kkt_system([0], {"gap": True})

    np.testing.assert_array_equal(theta, np.zeros(3))
    assert rate == 0.0


def test_gap_solver_failure_keeps_its_diagnostic() -> None:
    """An unusable numerical candidate must fail instead of becoming a zero component."""
    problem = GapRule(2, 1, np.array([0.5, -0.5]), np.eye(2))
    result = SimpleNamespace(success=False, message="iteration limit", x=np.zeros(2))

    with patch("efficient_importance_sampling.core.opt.minimize", return_value=result):
        with pytest.raises(RuntimeError, match="iteration limit"):
            problem.compute_feasible_mixture()


def test_gap_optimum_can_be_certified_despite_line_search_status() -> None:
    """The candidate's mathematical properties take precedence over a status flag."""
    optimizer = RegionOptimizer(CumulantFunction(np.array([0.5, -0.5]), np.eye(2)))
    result = SimpleNamespace(success=False, message="line-search failure", x=np.ones(2))

    with patch("efficient_importance_sampling.core.opt.minimize", return_value=result):
        theta, rate = optimizer.solve_kkt_system([1], {"gap": True})

    np.testing.assert_allclose(theta, [-1.0, 1.0])
    assert rate == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("candidate", "message"),
    [
        (np.array([np.nan, 0.0, 0.0]), "invalid tilt"),
        (np.ones(2), "invalid tilt"),
        (np.array([-0.1, 0.0, 0.0]), "infeasible tilt"),
        (np.array([10.0, 5.0, 5.0]), "infeasible tilt"),
        # This satisfies the centred quadratic boundary, but not the balance.
        (np.ones(3), "infeasible tilt"),
        (np.zeros(3), "zero tilt"),
        # This is balanced and on the Gaussian boundary, but is not the optimum.
        (np.array([2.0 / 3.0, 2.0 / 3.0, 0.0]), "KKT residual"),
    ],
)
def test_gap_success_flag_does_not_bypass_constraints(candidate: np.ndarray, message: str) -> None:
    """Accepting a tilt requires balance, feasibility, and optimality together."""
    optimizer = RegionOptimizer(CumulantFunction(np.array([1.0, 0.0, -1.0]), np.eye(3)))
    result = SimpleNamespace(success=True, x=candidate)

    with patch("efficient_importance_sampling.core.opt.minimize", return_value=result):
        with pytest.raises(RuntimeError, match=message):
            optimizer.solve_kkt_system([1, 2], {"gap": True})


@pytest.mark.parametrize("indices", [[], [-1], [3], [1, 1], [0, 1, 2]])
def test_gap_region_rejects_invalid_coordinate_subsets(indices: list[int]) -> None:
    """A gap region needs distinct valid indices and a nonempty complement."""
    optimizer = RegionOptimizer(CumulantFunction(np.array([1.0, 0.0, -1.0]), np.eye(3)))

    with pytest.raises(ValueError, match="indices|proper coordinate subset"):
        optimizer.solve_kkt_system(indices, {"gap": True})
