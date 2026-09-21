"""Run the bundled scientific demonstration."""

import numpy as np

from .core import run_comprehensive_example


def main() -> None:
    """Execute the three reference importance-sampling examples."""
    run_comprehensive_example(rng=np.random.default_rng(42))


if __name__ == "__main__":
    main()
