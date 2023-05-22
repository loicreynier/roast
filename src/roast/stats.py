"""
Statistics computation module (:mod:`roast.stats`)
==================================================

.. currentmodule:: roast.stats

Provides function to compute stastistic quantities used in the
analysis for turbulent flows.
"""

import numpy as np
from numpy.typing import NDArray

from roast.fspace import FSpace

__all__ = [
    "pdf",
    "moment",
    "rms",
    "structure",
]


def rms(*fields: NDArray) -> float:
    r"""RMS value.

    Notes
    -----
    The RMS of a dataset :math:`x = { x_1, x_2, \cdots, x_n}` is
    defined as

    .. math::

       X = \sqrt{(x_1^2 + x_2^2 + \cdots + x_n^2) / n}


    Examples
    --------
    >>> shape = (8, 32, 16)
    >>> u = np.random.rand(shape)
    >>> v = np.random.rand(shape)
    >>> w = 2 * u - v
    >>> u_rms = rms(u, v, w)
    """
    u = np.array(fields)
    u2 = np.zeros(u[0].shape)
    for u_i in u:
        u2 += u_i * u_i
    return np.sqrt(np.sum(u2) / (u.size))


def pdf(
    u: NDArray,
    bins: int = 100,
    norm: bool = False,
) -> tuple[NDArray, NDArray]:
    """Probability density function of field `u`.

    Parameters
    ----------
    u : NDArray
    bins : int, optional
    norm : bool, optional

    Returns
    -------
    freq : NDArray
    vals : NDArray
    """
    freq, vals = np.histogram(u, bins=bins, density=True)
    vals = (vals + np.roll(u, -1))[:-1] / 2.0
    if norm:
        freq = freq / np.sqrt(np.var(freq))
    return freq, vals


def moment(u: NDArray, n: float = 3.0) -> np.floating:
    """Moment of order `n` of field `u`.

    Parameters
    ----------
    u : NDArray
    n : int

    Returns
    -------
    moment : float
    """
    return np.average(u**n) / np.average(u**2) ** (n / 2.0)


def structure(
    u: NDArray,
    fs: FSpace,
    n: int = 6,
    axis: int = 0,
) -> tuple[NDArray, NDArray]:
    """Structure function of order `n` of the field `u` along `axis`.

    Parameters
    ----------
    u : NDArray
        Input field.
    fs : :class:`roast.fspace.FSpace`
        Function space associated.
    n : int
        Structure function order.
    axis : {0, 1, 2}
        Structure function direction.

    Returns
    -------
    f : NDArray
        Structure function.
    r : NDArray
        Displacement vector.
    """
    x = fs.domain.axes[axis]
    dx = np.abs(x[0] - x[1])
    r = np.zeros((x.size // 2,))
    f = r.copy()
    for i, _ in enumerate(r):
        r[i] = i * dx
        f[i] = np.average(np.power(np.roll(u, i, axis=axis) - u, n))
    return f, r
