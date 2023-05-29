"""ROAST turbulence computation functions tests."""

from roast import turbu

ETA_PRECISION = 1e-4


def test_kolmogorov_scale() -> None:
    """Test Kolmogorov scale computation.

    Function is tested upon the values of table 1 of [1].

    References
    ----------
    .. [1] Ishihara, T., Y. Kaneda, M. Yokokawa, K. Itakura, and A. Uno.
       "Small-Scale Statistics in High-Resolution Direct Numerical
       Simulation of Turbulence: Reynolds Number Dependence of One-Point
       Velocity Gradient Statistics.". Journal of Fluid Mechanics 592
       (2007): 335–66.
    """
    params = [
        (1.13, 0.203, 936),
        (1.02, 0.125, 2100),
        (1.28, 0.08927, 6710),
    ]
    values = [
        7.97e-3,
        3.95e-3,
        2.10e-3,
    ]
    for i, args in enumerate(params):
        assert abs(turbu.kolmogorov_scale(*args) - values[i]) < ETA_PRECISION
