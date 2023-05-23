"""FSpace tests."""

from typing import Literal

import numpy as np
import pytest
import scipy as sp

from roast import bflow
from roast.domain import Domain
from roast.fspace import FSpace

FLOAT_PRECISION: float = 1e-12
INTEGRAL_PRECISION_256: float = 1e-3
FFT_LIBS = ["numpy", "pyfftw"]
FFT_LIB = Literal["numpy", "pyfftw"]


@pytest.mark.parametrize("fft_lib", FFT_LIBS)
def test_fft_diff(
    fft_lib: FFT_LIB,
    shape: tuple[int, int, int] = (512, 256, 128),
) -> None:
    """Test FFT function space derivation on a Gaussian field."""
    dom = Domain(
        shape,
        bounds=tuple((-n * np.pi / 64, n * np.pi / 64) for n in shape),
    )
    fs = FSpace(dom, fft_lib=fft_lib)
    f, dfdx, dfdy, dfdz, _, _ = bflow.gaussian(dom)
    err = np.zeros((3,))

    for dir_, df in zip(range(3), (dfdx, dfdy, dfdz)):
        df_n = fs.diff(f, dir_)
        err[dir_] = np.linalg.norm(df - df_n.real) / np.linalg.norm(df)

    assert np.all(err < FLOAT_PRECISION)


@pytest.mark.parametrize("fft_lib", FFT_LIBS)
def test_fft_grad(
    fft_lib: FFT_LIB,
    shape: tuple[int, int, int] = (512, 256, 128),
) -> None:
    """Test FFT function space gradient computation on a Gaussian field."""
    dom = Domain(
        shape,
        bounds=tuple((-n * np.pi / 64, n * np.pi / 64) for n in shape),
    )
    fs = FSpace(dom, fft_lib=fft_lib)
    f, dfdx, dfdy, dfdz, _, _ = bflow.gaussian(dom)
    gradf = np.asarray((dfdx, dfdy, dfdz))
    gradf_n = fs.grad(f)

    err = np.asarray(
        [
            np.linalg.norm(gradf[dir_] - gradf_n[dir_].real)
            / np.linalg.norm(gradf)
            for dir_ in range(3)
        ]
    )

    assert np.all(err < FLOAT_PRECISION)


@pytest.mark.parametrize("fft_lib", FFT_LIBS)
def test_fft_div(
    fft_lib: FFT_LIB,
    shape: tuple[int, int, int] = (512, 256, 128),
) -> None:
    """Test FFT function space divergence computation on a Gaussian field."""
    dom = Domain(
        shape,
        bounds=tuple((-n * np.pi / 64, n * np.pi / 64) for n in shape),
    )
    f, _, _, _, divf, _ = bflow.gaussian(dom)
    fs = FSpace(dom, fft_lib=fft_lib)

    divf_n = fs.div(*((f,) * 3))

    assert (
        np.linalg.norm(divf - divf_n.real) / np.linalg.norm(divf)
        < FLOAT_PRECISION
    )


@pytest.mark.parametrize("fft_lib", FFT_LIBS)
def test_fft_lap(
    fft_lib: FFT_LIB,
    shape: tuple[int, int, int] = (512, 256, 128),
) -> None:
    """Test FFT function space laplacian computation on a Gaussian field."""
    dom = Domain(
        shape,
        bounds=tuple((-n * np.pi / 64, n * np.pi / 64) for n in shape),
    )
    f, _, _, _, _, lapf = bflow.gaussian(dom)
    fs = FSpace(dom, fft_lib=fft_lib)

    lapf_n = fs.lap(f)

    assert (
        np.linalg.norm(lapf - lapf_n.real) / np.linalg.norm(lapf)
        < FLOAT_PRECISION
    )


@pytest.mark.parametrize("fft_lib", FFT_LIBS)
def test_fft_lap_inv(
    fft_lib: FFT_LIB,
    shape: tuple[int, int, int] = (512, 256, 128),
) -> None:
    """Test FFT function space inverse laplacian computation.

    Test is performed on the velocity first component of the VDINSE
    pseudo-exact solution.
    """
    dom = Domain(shape)
    flow = bflow.vdinse_solexact(dom, np.pi / 4.0)
    fs = FSpace(dom, fft_lib=fft_lib)

    f = flow[0]
    f_n = fs.lap_inv(fs.lap(f))

    assert np.linalg.norm(f - f_n.real) / np.linalg.norm(f) < FLOAT_PRECISION


@pytest.mark.parametrize("fft_lib", FFT_LIBS)
def test_fft_integral(
    fft_lib: FFT_LIB,
    shape: tuple[int, int, int] = (256, 256, 256),
) -> None:
    """Test FFT function space volume integral computation."""
    dom = Domain(shape)
    fs = FSpace(dom, fft_lib=fft_lib)
    x, y, z = dom.mgrid
    f = np.sin(x**2) * np.cos(y**2) * np.cos(z**2)

    S, C = sp.special.fresnel(np.sqrt(2 * np.pi))
    i = 2 * np.sqrt(2) * np.pi ** (3 / 2) * C**2 * S
    i_n = fs.volume_integral(f)

    assert np.abs(i - i_n) < INTEGRAL_PRECISION_256
