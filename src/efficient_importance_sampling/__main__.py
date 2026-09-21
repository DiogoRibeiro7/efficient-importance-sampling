"""Run the bundled scientific demonstration."""

from .core import run_comprehensive_example


def main() -> None:
    """Execute the three reference importance-sampling examples."""
    run_comprehensive_example()


if __name__ == "__main__":
    main()
