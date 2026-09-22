# Efficient Importance Sampling for Rare Events

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Gaussian implementation of importance sampling techniques for estimating wrong exit probabilities in systems with combinatorially many rare regions, based on **Song & Fellouris (2025)**.

## Overview

This package constructs importance sampling proposals for rare event simulation in high-dimensional settings where:

- Events of interest are exponentially rare (probability ~ e^(-αb) as b → ∞)
- The number of rare regions grows combinatorially with dimension
- Traditional importance sampling becomes computationally infeasible

### Key Innovation

Selected region and auxiliary tilts reduce the number of mixture components. The paper establishes asymptotic efficiency under problem-specific conditions; constructing the proposals alone does not verify those conditions.

## Features

### 1\. **Multidimensional Siegmund Problem** (Section 4)

Estimates the probability of wrong exit when a multidimensional random walk with negative drift crosses boundaries.

**Application**: Sequential testing with multiple endpoints, change-point detection

**Complexity Reduction**: O(2^d) → O(d)

### 2\. **Gap Rule** (Section 5)

Estimates wrong selection probability in sequential multiple testing with gap-based stopping rules.

**Application**: Clinical trials, A/B testing with multiple variants

**Complexity Reduction**: O(2^d) → O(m(d-m))

### 3\. **Sum-Intersection Rule** (Section 6)

Estimates the probability of at least $L$ positive coordinates when the sum of
the $L$ smallest absolute coordinates first exceeds the stopping threshold.

**Application**: Rank-based sequential tests, order statistics

**Mixture size**: $2\binom{d}{L}$ component slots, polynomial in $d$ when $L$ is fixed.

## Mathematical Framework

The implementation includes:

- ✅ Cumulant generating functions (CGFs) and rate functions
- ✅ KKT system solvers for optimal exponential tilts
- ✅ Region and auxiliary proposal families
- ✅ Numerical feasibility and optimality checks for Gaussian tilts

The paper supplies the asymptotic efficiency results and their assumptions.
For the sum-intersection rule, call `check_efficiency_condition()` explicitly
to evaluate condition (H-SI) from Theorem 6.2 numerically.

## Installation

```bash
# Clone the repository
git clone https://github.com/diogoribeiro7/efficient-importance-sampling.git
cd efficient-importance-sampling

# Install the package and development tools
python -m pip install -e ".[dev]"
```

### Requirements

```
numpy>=1.26,<2.3
scipy>=1.11
```

## Quick Start

### Example 1: Multidimensional Siegmund Problem

```python
import numpy as np
from efficient_importance_sampling import MultidimensionalSiegmund

# Setup: 4D problem with negative drift and correlation
d = 4
mean = -0.5 * np.ones(d)
cov = np.eye(d) + 0.3 * (np.ones((d, d)) - np.eye(d))

# Initialize problem
siegmund = MultidimensionalSiegmund(
    d=d, 
    ell=1.0,  # lower boundary
    u=1.0,    # upper boundary
    mean=mean, 
    covariance=cov
)

# Estimate wrong exit probability
result = siegmund.simulate_wrong_exit_probability(
    b=3.0,              # scaling parameter
    n_samples=5000,
    use_feasible_mixture=True
)

print(f"Probability: {result.estimate:.2e} ± {result.std_error:.2e}")
print(f"Relative error: {result.relative_error:.3f}")
```

The stopping rule requires every coordinate to be strictly above `b * u` or strictly
below `-b * ell` at the same time. A coordinate exactly on a boundary is still inside.
The upper and lower boundaries may differ, as in equation (1) of
[Song and Fellouris (2025)](https://arxiv.org/html/2509.14596v1).

Each path has a step limit, which defaults to `int(10 * b) + 1000`. Set the keyword
argument `max_steps` to a positive integer to change it. Exits on the final permitted
step count normally. If any path remains inside when the limit is reached, the method
raises `RuntimeError` and returns no estimate. Increase the limit and rerun the whole
simulation with a fresh generator using the same seed. Unfinished paths are never
treated as zero contributions, which would bias the probability estimate.

### Example 2: Gap Rule

```python
from efficient_importance_sampling import GapRule

# Sequential testing: 3 signals among 6 hypotheses
gap_rule = GapRule(
    d=6,                    # total hypotheses
    m=3,                    # true signals
    mean=np.array([0.5, 0.5, 0.5, -0.5, -0.5, -0.5]),
    covariance=np.eye(6)
)

# Compute efficient mixture (polynomial complexity)
tilts, weights = gap_rule.compute_feasible_mixture()
print(f"Mixture components: {len(tilts)} (vs 2^6-1=63 naively)")
```

### Example 3: Sum-Intersection Rule

```python
import numpy as np

from efficient_importance_sampling import SumIntersectionRule

# Stop using the L smallest absolute coordinates.
sum_int = SumIntersectionRule(
    d=5,
    L=2,
    mean=-0.5 * np.ones(5),
    covariance=np.eye(5)
)

tilts, weights = sum_int.compute_feasible_mixture()
print(f"Mixture components: {len(tilts)}")  # 20 = 2 * binom(5, 2)

diagnostic = sum_int.check_efficiency_condition()
print(diagnostic.status)  # "satisfied" for this independent Gaussian model
print(f"H-SI margin: {diagnostic.margin:.6f}")  # 0.500000

result = sum_int.simulate_wrong_exit_probability(
    b=2.0,
    n_samples=5000,
    rng=np.random.default_rng(42),
    max_steps=2000,
)
print(f"Wrong exit probability: {result.estimate:.6g} (SE {result.std_error:.3g})")
print(f"Log probability estimate: {result.log_probability:.6f}")
```

For this simulator, `n_samples` must be an integer of at least two and every
coordinate drift must be strictly negative. The sample standard error uses
`ddof=1`. Proposal construction and simulation do not run the efficiency
diagnostic automatically.

### Sum-intersection stopping and likelihood

Write $|S_n|_{[1]}\leq\cdots\leq|S_n|_{[d]}$ for the absolute coordinates
in increasing order. The simulator uses

$$
T=\inf\left\{n\geq1:\sum_{j=1}^{L}|S_n|_{[j]}>b\right\},\qquad
E=\left\{\#\{k:S_{T,k}>0\}\geq L\right\}.
$$

Equality at the threshold does not stop a path. Coordinate signs are evaluated
at stopping, and a zero coordinate is not positive. One proposal component is
drawn per path and held fixed. For terminal position $S_T$, the contribution is

$$
Z=\mathbf{1}_E\left[\sum_j w_j
   \exp\left(\theta_j^\top S_T-T\Lambda(\theta_j)\right)\right]^{-1}.
$$

This uses the complete mixture density with its actual CGF values. The stopping
rule and exponential path density follow Sections 6 and 2.2 of
[Song and Fellouris (2025)](https://arxiv.org/html/2509.14596v1).

Contributions are scaled before computing their mean and sample variance. This
preserves `log_probability` and `relative_error` when a very small `estimate` or
`std_error` rounds to zero. If no simulated path produces a wrong exit, both
empirical values are zero, `log_probability` is `-inf`, and `relative_error` is
`inf`; this does not show that the event is impossible or its uncertainty is zero.

The per-path step limit defaults to `10 * int(b) + 1000`. An exit on the final
permitted step counts. If any path remains unfinished, the method raises
`RuntimeError` and returns no estimate. Increase `max_steps` and rerun the whole
experiment with a fresh generator initialised to the same seed.

## Running the Demo

```bash
python -m efficient_importance_sampling
```

This will run comprehensive examples for all three problems with:

- Performance comparisons (feasible vs. full mixture)
- Scaling analysis across dimensions
- Timing benchmarks
- Construction of region and auxiliary proposal families

These numerical examples do not verify the paper's asymptotic efficiency conditions.

## Algorithm Details

### Asymptotic Efficiency

An importance sampling estimator is **asymptotically efficient** if:

```
lim_{b→∞} [log Var(Z_b)] / [log P(E_b)] = 2
```

where `Z_b` is the importance sampling estimator and `E_b` is the rare event.

### Mixture Construction Strategy

For a rare event `E_b = ∪_{A⊆[d]} W^A_b` (union of exponentially many regions):

1. **Identify critical regions**: Compute optimal tilts β^A solving:

  ```
  minimize r_A subject to Λ(β^A) = 0, sign constraints
  ```

2. **Strategic selection**: Choose the region family for the stopping rule

3. **Additional coverage**: Add auxiliary tilts γ^k for variance control

4. **Efficiency conditions**: Check the relevant theorem's assumptions separately

### Gaussian Siegmund region tilts

For a nonempty set of upper-exit coordinates $A$, the region tilt maximises

$$
u\sum_{k\in A}\theta_k-\ell\sum_{k\notin A}\theta_k
\quad\text{subject to}\quad
\Lambda(\theta)\leq 0,\qquad
\theta_k\geq 0\ (k\in A),\quad \theta_k\leq 0\ (k\notin A).
$$

This is the convex optimisation problem in Lemma 4.1 of
[Song and Fellouris (2025)](https://arxiv.org/html/2509.14596v1). The implementation
uses SLSQP with analytical derivatives, scaled variables, and a feasible initial
point. It checks the candidate's feasibility and KKT residuals independently of
the solver's status flag. Small numerical boundary residuals are corrected along
the same ray before the optimality check.

An unacceptable candidate raises `RuntimeError` with a diagnostic instead of
silently returning a zero tilt and rate. The full region cache is populated only
after every requested region has been solved. For a one-dimensional Gaussian with
mean $-1/2$, variance $1$, and $u=1$, the region tilt and rate are both $1$.

### Gaussian gap-region tilts

For a nonempty proper subset $A$ of selected coordinates, the gap-region tilt solves

$$
\max_\theta\sum_{k\in A}\theta_k
\quad\text{subject to}\quad
\Lambda(\theta)\leq 0,\quad\sum_k\theta_k=0,\quad
\theta_k\geq 0\ (k\in A),\quad\theta_k\leq 0\ (k\notin A).
$$

The zero-sum constraint in
[Lemma 5.1 of Song and Fellouris (2025)](https://arxiv.org/html/2509.14596v1)
ensures that adding a common increment to every coordinate does not change the
gap proposal. The solver centres the Gaussian mean and covariance on this
subspace, uses analytical derivatives, and checks feasibility and KKT residuals
before returning the tilt and rate. Small balance and boundary residuals are
corrected before the optimality check. An unacceptable numerical candidate raises
`RuntimeError` with a diagnostic.

For independent unit-variance coordinates with means $(1/2,-1/2)$ and selected
set $A=\{2\}$ (one-based indexing), the tilt is $(-1,1)$ and the rate is $1$.
If all selected-coordinate drifts are at least all complementary drifts, only
the zero tilt is feasible and the rate is zero.

### Gaussian sum-intersection region tilts

The region objective in
[Lemma 6.1 of Song and Fellouris (2025)](https://arxiv.org/html/2509.14596v1) is

$$
f_L(\theta)=\min_{1\leq j\leq L}\frac{1}{j}
\sum_{i=L-j+1}^{d}|\theta|_{(i)},
$$

where the magnitudes are sorted in decreasing order. The solver maximises this
objective subject to the Gaussian CGF constraint and the region's coordinate
signs. For $L=1$, it uses the Siegmund solver with both boundaries equal to one.

For larger $L$, a compact equivalent formulation introduces magnitudes
$y=|\theta|$, a rate $t$, and capacities $z$ satisfying

$$
0\leq z_i\leq y_i,\qquad z_i\leq t,\qquad
\sum_i z_i\geq Lt,\qquad t\geq0.
$$

Eliminating $z$ gives $\sum_i\min(y_i,t)\geq Lt$, which is equivalent to the
ordered-tail inequalities defining $t\leq f_L(\theta)$. This uses $2d+1$
variables without enumerating all size-$L$ subsets inside each region solve.
Returned candidates pass feasibility and an independent dual optimality check.
Invalid candidates raise `RuntimeError`; unknown region types raise `ValueError`.

### Sum-intersection auxiliary tilts and mixture

For every size-$L$ subset $A$, equation (43) defines an auxiliary tilt by

$$
\max_\theta\min_{i\in A}\theta_i
\quad\text{subject to}\quad
\Lambda(\theta)\leq0,\qquad
\theta_i\geq0\ (i\in A),\qquad
\theta_i=0\ (i\notin A).
$$

`RegionOptimizer.solve_sum_intersection_auxiliary(A_indices)` solves this problem
on the selected coordinates' Gaussian mean and principal covariance submatrix.
The full-order region objective on that submodel is exactly the minimum selected
coordinate. Singleton subsets use the exact Gaussian ray solution. The returned
tilt has zeros outside $A$; its selected coordinates need not be equal.

`SumIntersectionRule.compute_feasible_mixture` constructs all size-$L$ region
tilts followed by all size-$L$ auxiliary tilts, each family in lexicographic
subset order. It returns exactly $2\binom{d}{L}$ equally weighted component slots,
retaining coincident tilts and imposing no subset cap. At $L=1$ the construction
agrees with the unit-boundary Siegmund mixture.

This is the proposal family in
[Theorem 6.2 of Song and Fellouris (2025)](https://arxiv.org/html/2509.14596v1).
The builder does **not** evaluate the theorem's sufficient condition (H-SI)
automatically. The diagnostic below evaluates it in a separate call.

### Numerical sum-intersection efficiency diagnostic

`SumIntersectionRule.check_efficiency_condition(rtol=1e-6, atol=1e-10)` returns a
`SumIntersectionEfficiencyResult` for models with strictly negative coordinate
drifts. It computes the margin

$$
\Delta=\min_{|A|=L,\;k\notin A}\left(z_A+s_{A\cup\{k\}}\right)
       -2\min_{|A|=L}r_A.
$$

The values $s_B$ come from the supported order-$L$ optimisation in equation (44)
on each size-$(L+1)$ subset. Each such problem is solved once, and the comparison
uses nested pairs $A\subset B$ rather than minimising $z_A$ and $s_B$ independently.
The diagnostic does not add these coverage tilts to the mixture. It performs
$2\binom{d}{L}+\binom{d}{L+1}$ solves and retains $\binom{d}{L}$ auxiliary values,
so its cost still grows combinatorially when $L$ grows with $d$.

The result includes `minimum_region_rate`, `required_bound`, `coverage_bound`,
`margin`, and `tolerance`. The comparison tolerance is
`atol + rtol * max(abs(coverage_bound), abs(required_bound))`.

| Status | Numerical comparison |
| --- | --- |
| `satisfied` | `margin > tolerance` |
| `not_satisfied` | `margin < -tolerance` |
| `borderline` | `abs(margin) <= tolerance`, including equality |

`critical_region` identifies a size-$L$ subset attaining the smallest region
rate. `weakest_subset` and `extra_coordinate` identify a nested pair attaining
the coverage bound. Indices are zero-based; tied minima return one attaining pair.
Invalid tolerances or nonnegative drifts raise `ValueError`. Failed numerical
optimisations raise `RuntimeError` without returning a partial diagnostic.

The comparison tolerance is a numerical decision band, not a rigorous error
bound. A satisfied result is numerical evidence for (H-SI), not a proof of
asymptotic efficiency. A failed sufficient condition does not establish
inefficiency, and other theorem assumptions are not checked by this method.

### Gaussian auxiliary tilts

For a direction $v$ and a nonnegative scale $t$, the Gaussian cumulant generating
function is

$$
\Lambda(tv) = t\,\mu^\top v + \tfrac12 t^2 v^\top\Sigma v.
$$

If $\mu^\top v < 0$, the largest feasible scale is
$t_* = -2\mu^\top v/(v^\top\Sigma v)$. Otherwise the feasible tilt is zero.
`CumulantFunction.optimal_ray_tilt(direction)` computes this endpoint directly and
returns zero for a zero direction.

The Siegmund auxiliary components use $v=e_k$. The gap-rule auxiliary components use
$v=e_{\ell'}-e_\ell$, whose projected variance includes the covariance term
$\Sigma_{\ell\ell}+\Sigma_{\ell'\ell'}-2\Sigma_{\ell\ell'}$. These implement the
Gaussian versions of equations (27) and (36) in
[Song and Fellouris (2025)](https://arxiv.org/html/2509.14596v1), without a bounded
scalar search. The remaining region-specific components still use numerical
optimisation. Exact auxiliary tilts alone do not establish asymptotic efficiency
of the complete mixture.

## Performance

Dimension | Components (Feasible) | Components (Full) | Speedup
--------- | --------------------- | ----------------- | -------
d=2       | 4                     | 3                 | ~1x
d=4       | 8                     | 15                | ~10x
d=6       | 12                    | 63                | ~100x
d=8       | 16                    | 255               | ~1000x


## API Reference

### Core Classes

#### `CumulantFunction`

Handles cumulant generating functions and rate functions for multivariate normal distributions.

#### `RegionOptimizer`

Computes region tilts and rates using convex constrained solvers with independent
feasibility and optimality checks.

`solve_sum_intersection_auxiliary(A_indices)` returns a full-dimensional supported
tilt and its minimum selected coordinate. Indices must be a nonempty subset of
distinct valid coordinates. Invalid input raises `ValueError`; an unvalidated
numerical candidate raises `RuntimeError`.

`solve_sum_intersection_coverage(B_indices, L)` returns the supported tilt and
value from equation (44). The subset must contain exactly `L+1` distinct valid
coordinates and `L` must be an integer satisfying `1 <= L < d`.

#### `MultidimensionalSiegmund`

Implements the multidimensional Siegmund problem with boundary crossings.

#### `GapRule`

Handles sequential multiple testing with gap-based stopping rules.

#### `SumIntersectionRule`

Implements sum-intersection stopping rules with order statistics.

`check_efficiency_condition(*, rtol=1e-6, atol=1e-10)` returns a numerical (H-SI)
diagnostic independently of `compute_feasible_mixture()`.

`simulate_wrong_exit_probability(b, n_samples=10000, rng=None, *, max_steps=None)`
returns a `SimulationResult` using the complete proposal family. It requires
at least two paths and strictly negative coordinate drifts. Invalid parameters
raise `ValueError`; an unfinished path or failed numerical computation raises
`RuntimeError` without a partial estimate.

### Data Structures

#### `SumIntersectionEfficiencyResult`

Immutable numerical diagnostic with the region rate, coverage bound, comparison
margin, tolerance, attaining subsets, and a three-way `status`. It is exported
from `efficient_importance_sampling` alongside `SumIntersectionRule`.

#### `SimulationResult`

```python
@dataclass
class SimulationResult:
    estimate: float              # Monte Carlo estimate
    std_error: float            # Standard error
    relative_error: float       # SE / estimate
    samples_used: int           # Number of samples
    computation_time: float     # Wall-clock time
    log_probability: float      # log(estimate)
```

## Theory Background

### The Problem

Estimating `P(E_b)` where:

- Event `E_b` is the union of exponentially many rare regions
- Each region `W^A` has escape rate `r_A`
- Optimal tilt `β^A` for region `W^A` satisfies `Λ(β^A) = 0`

### The Challenge

Using all `2^d - 1` optimal tilts:

- ✅ Achieves asymptotic efficiency
- ❌ Computationally infeasible for d ≥ 20

### The Solution

Carefully selected region and auxiliary tilts:

- Asymptotic efficiency when the applicable theorem's conditions hold
- $2\binom{d}{L}$ components for the sum-intersection rule, polynomial for fixed $L$
- Coverage conditions that must be checked for the chosen model

## Citation

If you use this code in your research, please cite:

```bibtex
@article{song2025efficient,
  title={Efficient Importance Sampling for Rare Events with Combinatorially Many Regions},
  author={Song, Yanglei and Fellouris, Georgios},
  journal={},
  year={2025}
}
```

## Applications

- **Clinical Trials**: Sequential monitoring of multiple endpoints
- **Finance**: Multi-asset risk assessment and portfolio optimization
- **Quality Control**: Multivariate process monitoring
- **Machine Learning**: Rare event prediction in high-dimensional spaces
- **Telecommunications**: Network reliability with multiple failure modes

## Contributing

Install the development dependencies and repository hooks before making changes:

```bash
python -m pip install -e ".[dev]"
pre-commit install
pre-commit install --hook-type pre-push
```

Commit hooks run Ruff and mypy. The pre-push hook also runs the test suite. To verify the
entire repository manually:

```bash
pre-commit run --all-files
pre-commit run --all-files --hook-stage pre-push
```

Create a focused branch from `main`, add tests for behavioural changes, and open a pull
request back to `main`. Do not commit generated environments, caches, or local results.

## License

This project is licensed under the MIT License - see the <LICENSE> file for details.

## Acknowledgments

- Based on theoretical results from Song & Fellouris (2025)
- Builds on classical work by Siegmund, Armitage, and others in sequential analysis
- Implements state-of-the-art importance sampling techniques

## Contact

For questions or issues, please open an issue on GitHub or contact [dfr@esmad.ipp.pt]

--------------------------------------------------------------------------------

**Keywords**: importance sampling, rare events, Monte Carlo, sequential analysis, asymptotic efficiency, high-dimensional statistics, exponential tilting, multiple testing
