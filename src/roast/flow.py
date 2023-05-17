"""
Flow module (:mod:`roast.turbu`)
================================

Provides function to compute flow-related quantities.
"""

import numpy as np
from numpy.typing import NDArray

from roast.fspace import FSpace

__all__ = [
    "energy",
    "vorticity",
    "vorticity_comp",
    "pressure_fp",
]


def energy(
    u: NDArray,
    v: NDArray,
    w: NDArray,
    rho: NDArray,
) -> NDArray:
    """Kinetic energy

    Parameters
    ----------
    u : NDArray
        Velocity field 1st direction component.
    v : NDArray
        Velocity field 2nd direction component.
    w : NDArray
        Velocity field 3rd direction component.
    rho : NDArray
        Density field.

    Returns
    -------
    e : float
        Kinetic energy
    """
    return rho * (u * u + v * v + w * w)


def vorticity_comp(
    comp: int,
    u: NDArray,
    v: NDArray,
    w: NDArray,
    fs: FSpace,
) -> NDArray:
    """Vorticity vector component `comp`.

    Parameters
    ----------
    u : NDArray
        Velocity field 1st direction component.
    v : NDArray
        Velocity field 2nd direction component.
    w : NDArray
        Velocity field 3rd direction component.
    fs : FSpace
        Associated function space used for differentiation.

    Returns
    -------
    vort : NDArray
        Vorticity vector component.
    """
    if comp not in [0, 1, 2]:
        raise ValueError(f"Unknown component '{comp}'")

    if comp == 0:
        vort = fs.diff(w, 1) - fs.diff(v, 2)
    elif comp == 1:
        vort = fs.diff(u, 2) - fs.diff(w, 0)
    else:
        vort = fs.diff(v, 0) - fs.diff(u, 1)

    return vort


def vorticity(
    u: NDArray,
    v: NDArray,
    w: NDArray,
    fs: FSpace,
) -> NDArray:
    """Vorticity norm.

    Parameters
    ----------
    u : NDArray
        Velocity field 1st direction component.
    v : NDArray
        Velocity field 2nd direction component.
    w : NDArray
        Velocity field 3rd direction component.
    fs : FSpace
        Associated function space used for differentiation.

    Returns
    -------
    vort : NDArray
        Vorticity.
    """
    return np.sqrt(
        vorticity_comp(0, u, v, w, fs) ** 2
        + vorticity_comp(1, u, v, w, fs) ** 2
        + vorticity_comp(2, u, v, w, fs) ** 2
    )


def pressure_fp(
    u: NDArray,
    v: NDArray,
    w: NDArray,
    rho: NDArray,
    fs: FSpace,
    maxiter: int = 50,
    tol: float = 1e-6,
) -> NDArray:
    """Pressure field computed with a fixed point method.

    Parameters
    ----------
    u : NDArray
        Velocity field 1st direction component.
    v : NDArray
        Velocity field 2nd direction component.
    w : NDArray
        Velocity field 3rd direction component.
    rho : NDArray
        Density field.
    fs : FSpace
        Associated function space used for differentiation.

    Returns
    -------
    p : NDArray
        Pressure field.
    """
    divu = fs.div(*fs.ugradu(u, v, w))
    drhodx, drhody, drhodz = fs.grad(rho)
    p = fs.lap_inv(divu)
    res = 1.0
    k = 0
    while res > tol:
        dpdx, dpdy, dpdz = fs.grad(p)
        f = (drhodx * dpdx + drhody * dpdy + drhodz * dpdz) / rho + rho * divu
        res = np.max(np.abs(fs.lap(p) - f))
        k += 1
        if k > maxiter:
            break
        p = fs.lap_inv(f)
    return p
