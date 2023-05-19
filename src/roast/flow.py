"""
Flow module (:mod:`roast.flow`)
===============================

Provides function to compute flow-related quantities.
"""

import numpy as np
from numpy.typing import NDArray

from roast.fspace import FSpace

__all__ = [
    "acceleration_fick",
    "acceleration_vdinse_eulerian",
    "acceleration_vdinse_lagrangian",
    "acceleration_viscous",
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


def acceleration_viscous(
    u: NDArray,
    v: NDArray,
    w: NDArray,
    fs: FSpace,
    a: float | NDArray = 1.0,
) -> NDArray:
    r"""Acceleration resulting from the viscous drag.

    .. math::

       a \nabla^2 \vec{u}

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
    a : float | NDArray
        Viscous coefficient.

    Returns
    -------
    a : NDArray
        Viscous drag acceleration
    """
    return np.array([fs.lap(u), fs.lap(v), fs.lap(w)]) / a


def acceleration_fick(
    u: NDArray,
    v: NDArray,
    w: NDArray,
    rho: NDArray,
    fs: FSpace,
    a: float | NDArray = 1.0,
) -> NDArray:
    r"""Acceleration resulting from the VDINSE mass diffusion term.

    .. math::

       a \frac1{\mathrm{Re\,Sc}}
         \left(
             (\nabla\rho\cdot\nabla)\vec{u}
           + (u\cdot\nabla)\nabla\rho
         \right)

    Parameters
    ----------
    u : NDArray
        Velocity field 1st direction component.
    v : NDArray
        Velocity field 2nd direction component.
    w : NDArray
        Velocity field 3rd direction component.
    rho : NDArray
        Density field
    fs : FSpace
        Associated function space used for differentiation.
    a : float | NDArray
        Diffusion coefficient.
    """
    drho = fs.grad(rho)
    ugrdrodx, ugrdrody, ugrdrodz = fs.ugradv(u, v, w, *drho)
    grodudx, grodudy, grodudz = fs.ugradv(*drho, u, v, w)
    return np.array(
        a
        / rho
        * [
            ugrdrodx + grodudx,
            ugrdrody + grodudy,
            ugrdrodz + grodudz,
        ]
    )


def acceleration_vdinse_lagrangian(
    u: NDArray,
    v: NDArray,
    w: NDArray,
    rho: NDArray,
    p: NDArray,
    fs: FSpace,
    Re: float = 1000,
    Sc: float = 1,
) -> NDArray:
    r"""Lagrangian acceleration of a VDINSE flow.

    .. math::

       \frac{\mathrm{D}\vec{u}}{\mathrm{D}t} =
           - \frac1{\rho}\nabla{p}
           + \frac1{\rho\mathrm{Re}}\nabla^2\vec{u}
           + \frac1{\rho\mathrm{Re\,Sc}}
               \left(
                   (\nabla\rho\cdot\nabla)\vec{u}
                 + (u\cdot\nabla)\nabla\rho
               \right)


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
    p : NDArray
        Pressure field.
    fs : FSpace
        Associated function space used for differentiation.
    Re : float, optional
        Reynolds number
    Sc : float, optional
        Schmidt number

    Returns
    -------
    a : NDArray
        Lagrangian acceleration
    """
    a_pres = fs.grad(p)
    a_visc = acceleration_viscous(u, v, w, fs, a=1.0 / (Re * rho))
    a_rho = acceleration_fick(u, v, w, rho, fs, a=1.0 / (Re * Sc))
    return -a_pres + a_visc + a_rho


def acceleration_vdinse_eulerian(
    u: NDArray,
    v: NDArray,
    w: NDArray,
    rho: NDArray,
    p: NDArray,
    fs: FSpace,
    Re: float = 1000,
    Sc: float = 1,
) -> NDArray:
    r"""Eulerian acceleration of a VDINSE flow.

    .. math::

       \frac{\partial\vec{u}}{\partial{t}} =
           - (\vec{u}\cdot\nabla)\vec{u}
           - \frac1{\rho}\nabla{p}
           + \frac1{\rho\mathrm{Re}}\nabla^2\vec{u}
           + \frac1{\rho\mathrm{Re\,Sc}}
               \left(
                   (\nabla\rho\cdot\nabla)\vec{u}
                 + (u\cdot\nabla)\nabla\rho
               \right)


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
    p : NDArray
        Pressure field.
    fs : FSpace
        Associated function space used for differentiation.
    Re : float, optional
        Reynolds number
    Sc : float, optional
        Schmidt number

    Returns
    -------
    a : NDArray
        Eulerian acceleration
    """
    return acceleration_vdinse_lagrangian(
        u, v, w, rho, p, fs, Re=Re, Sc=Sc
    ) - fs.ugradu(u, v, w)


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
