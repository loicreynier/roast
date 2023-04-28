"""FFT-based differentiation functions."""

__all__ = [
    "diff",
    "div",
    "curl",
    "grad",
    "lap",
    "rot",
]

import itertools

import numpy as np
from numpy.typing import NDArray

from roast.fft.typing import ROASTFFT


def diff(
    u: NDArray,
    dir_: int,
    fft: ROASTFFT,
) -> NDArray:
    """Derivative of `u` along the `dir_` direction.

    Parameters
    ----------
    u : NDArray
        Input field.
    dir_ : int
        Differentiation direction.
    fft : ROASTFFT
        FFT object used for computation.

    Returns
    -------
    deriv : NDArray
    """
    dir_spec = 2 - dir_ if fft.transposed else dir_

    u_c = fft.fft(u)
    indices = [slice(None), slice(None), slice(None)]
    for mode in range(fft.shape_spec[dir_spec]):
        indices[dir_spec] = mode  # type: ignore
        u_c[tuple(indices)] *= 1.0j * fft.modes[dir_spec][mode]
    u = fft.ifft(u_c).copy()

    return u


def grad(u: NDArray, fft: ROASTFFT) -> NDArray:
    """Gradient of `u`.

    Parameters
    ----------
    u : NDArray
        Input field.
    fft : ROASTFFT
        FFT object used for computation.

    Returns
    -------
    grad : NDArray
    """
    return np.array(
        (
            diff(u, 0, fft),
            diff(u, 1, fft),
            diff(u, 2, fft),
        )
    )


def div(u: NDArray, v: NDArray, w: NDArray, fft: ROASTFFT) -> NDArray:
    """Divergence of the vector field (`u`, `v`, `w`).

    Parameters
    ----------
    u, v, w : NDArray, NDArray, NDArray
        Input fields.
    fft : ROASTFFT
        FFT object used for computation.

    Returns
    -------
    div : NDArray
    """
    return diff(u, 0, fft) + diff(v, 1, fft) + diff(w, 2, fft)


def curl(u: NDArray, v: NDArray, w: NDArray, fft: ROASTFFT) -> NDArray:
    """Curl of the vector field (`u`, `v`, `w`).

    Parameters
    ----------
    u, v, w : NDArray, NDArray, NDArray
        Input fields.
    fft : ROASTFFT
        FFT object used for computation.

    Returns
    -------
    div : NDArray
    """
    _, dudy, dudz = grad(u, fft)
    dvdx, _, dvdz = grad(v, fft)
    dwdx, dwdy, _ = grad(w, fft)
    return np.asarray(
        (
            dwdy - dvdz,
            dudz - dwdx,
            dvdx - dudy,
        )
    )


rot = curl


def lap(u: NDArray, fft: ROASTFFT) -> NDArray:
    """Laplacian of `u`.

    Parameters
    ----------
    u : NDArray
        Input field.
    fft : ROASTFFT
        FFT object used for computation.

    Returns
    -------
    curl : NDArray
    """
    k2 = np.zeros(fft.shape_spec)
    u_c = fft.fft(u)

    for i, j, k in itertools.product(*[range(d) for d in fft.shape_spec]):
        k2[i, j, k] = (
            -fft.modes[0][i] * fft.modes[0][i]
            - fft.modes[1][j] * fft.modes[1][j]
            - fft.modes[2][k] * fft.modes[2][k]
        )
    u_c *= k2

    return fft.ifft(u_c).copy()
