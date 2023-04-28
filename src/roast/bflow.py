"""
Base flow module (:mod:`roast.bflow`)
=====================================

Provides functions to create base flow arrays.
"""


__all__ = [
    "vdinse_solexact",
    "VDFlow",
]

import numpy as np
from numpy.typing import NDArray

from roast.domain import Domain

VDFlow = tuple[NDArray, NDArray, NDArray, NDArray, NDArray]


def vdinse_solexact(
    domain: Domain,
    t: float,
    s: float = 2.0,
) -> VDFlow:
    """VDINSE exact solution flow fields.

    Parameters
    ----------
    domain : Domain
    t: float
    s: float, optional
        Density ratio.
    """
    x, y, z = domain.mgrid
    u = np.cos(t) * np.sin(x) * np.cos(y) * np.cos(z)
    v = np.cos(t) * np.cos(x) * np.sin(y) * np.cos(z)
    w = -2.0 * np.cos(t) * np.cos(x) * np.cos(y) * np.sin(z)
    p = np.sin(t) * np.sin(2.0 * x) * np.sin(2.0 * y) * np.sin(2.0 * z)
    rho = 1.0 + 0.5 * (s - 1.0) * (
        1.0 + np.sin(t) * np.sin(x) * np.sin(y) * np.cos(z)
    )

    return u, v, w, rho, p


def gaussian(
    domain: Domain,
) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]:
    """Gaussian functions and its derivative.

    Returns
    -------
    f : NDArray
        3D Gaussian function.
    dfdx : NDArray
        Derivative along the 1st axis.
    dfdy : NDArray
        Derivative along the 2nd axis.
    dfdz : NDArray
        Derivative along the 3rd axis.
    divf : NDArray
        Divergence of a 3-component vector field (f, f, f).
    lapf : NDArray
        Laplacian of a 3-component vector field (f, f, f).
    """
    mgrid = domain.mgrid
    f = np.exp(-(mgrid[0] ** 2 + mgrid[1] ** 2 + mgrid[2] ** 2))
    df = [-2.0 * mgrid[i] * f for i in range(3)]
    divf = sum(df)
    lapf = (4.0 * (mgrid[0] ** 2 + mgrid[1] ** 2 + mgrid[2] ** 2) - 6.0) * f
    return f, *df, divf, lapf  # type: ignore
