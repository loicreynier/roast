"""FFT-based differentiation tests."""

import numpy as np
import pytest
import scipy as sp

from roast import bflow, diff
from roast.domain import Domain
from roast.fft import FFT, FFTW, RFFT, RFFTW
from roast.fft.typing import ROASTFFT

FLOAT_PRECISION: float = 1e-12
INTEGRAL_PRECISION_256: float = 1e-3


@pytest.mark.parametrize("fft_class", [FFT, FFTW, RFFT, RFFTW])
def test_fft_diff(
    fft_class: ROASTFFT, shape: tuple[int, int, int] = (512, 256, 128)
) -> None:
    """Test FFT-based derivation on a Gaussian field."""
    dom = Domain(
        shape,
        bounds=tuple((-n * np.pi / 64, n * np.pi / 64) for n in shape),
    )
    f, dfdx, dfdy, dfdz, _, _ = bflow.gaussian(dom)
    fft = fft_class(dom.axes)
    err = np.zeros((3,))

    for dir_, df in zip(range(3), (dfdx, dfdy, dfdz)):
        df_n = diff.fftdiff.diff(f, dir_, fft)
        err[dir_] = np.linalg.norm(df - df_n.real) / np.linalg.norm(df)

    assert np.all(err < FLOAT_PRECISION)


@pytest.mark.parametrize("fft_class", [FFT, FFTW, RFFT, RFFTW])
def test_fft_grad(
    fft_class: ROASTFFT, shape: tuple[int, int, int] = (512, 256, 128)
) -> None:
    """Test FFT-based gradient computation on a Gaussian field."""
    dom = Domain(
        shape,
        bounds=tuple((-n * np.pi / 64, n * np.pi / 64) for n in shape),
    )
    f, dfdx, dfdy, dfdz, _, _ = bflow.gaussian(dom)
    fft = fft_class(dom.axes)

    gradf = np.asarray((dfdx, dfdy, dfdz))
    gradf_n = diff.fftdiff.grad(f, fft)

    err = np.asarray(
        [
            np.linalg.norm(gradf[dir_] - gradf_n[dir_].real)
            / np.linalg.norm(gradf)
            for dir_ in range(3)
        ]
    )

    assert np.all(err < FLOAT_PRECISION)


@pytest.mark.parametrize("fft_class", [FFT, FFTW, RFFT, RFFTW])
def test_fft_div(
    fft_class: ROASTFFT, shape: tuple[int, int, int] = (512, 256, 128)
) -> None:
    """Test FFT-based divergence computation on a Gaussian field."""
    dom = Domain(
        shape,
        bounds=tuple((-n * np.pi / 64, n * np.pi / 64) for n in shape),
    )
    f, _, _, _, divf, _ = bflow.gaussian(dom)
    fft = fft_class(dom.axes)

    divf_n = diff.fftdiff.div(*((f,) * 3), fft)

    assert (
        np.linalg.norm(divf - divf_n.real) / np.linalg.norm(divf)
        < FLOAT_PRECISION
    )


@pytest.mark.parametrize("fft_class", [FFT, FFTW, RFFT, RFFTW])
def test_fft_lap(
    fft_class: ROASTFFT, shape: tuple[int, int, int] = (512, 256, 128)
) -> None:
    """Test FFT-based Laplacian computation on a Gaussian field."""
    dom = Domain(
        shape,
        bounds=tuple((-n * np.pi / 64, n * np.pi / 64) for n in shape),
    )
    f, _, _, _, _, lapf = bflow.gaussian(dom)
    fft = fft_class(dom.axes)

    lapf_n = diff.fftdiff.lap(f, fft)

    assert (
        np.linalg.norm(lapf - lapf_n.real) / np.linalg.norm(lapf)
        < FLOAT_PRECISION
    )


@pytest.mark.parametrize("fft_class", [RFFT, RFFTW])
def test_fft_lap_inv(
    fft_class: ROASTFFT,
    shape: tuple[int, int, int] = (256, 256, 256),
) -> None:
    """Test FFT-based inverse Laplacian computation.

    Test is performed on the velocity first component of the VDINSE
    pseudo-exact solution.
    """
    dom = Domain(shape)
    flow = bflow.vdinse_solexact(dom, np.pi / 4.0)
    fft = fft_class(dom.axes)

    f = flow[0]
    f_n = diff.fftdiff.lap_inv(diff.fftdiff.lap(f, fft), fft)

    assert np.linalg.norm(f - f_n.real) / np.linalg.norm(f) < FLOAT_PRECISION


@pytest.mark.parametrize("fft_class", [FFT, FFTW, RFFT, RFFTW])
def test_fft_integral(
    fft_class: ROASTFFT,
    shape: tuple[int, int, int] = (256, 256, 256),
) -> None:
    """Test FFT-based volume integral computation."""
    dom = Domain(shape)
    fft = fft_class(dom.axes)
    x, y, z = dom.mgrid
    f = np.sin(x**2) * np.cos(y**2) * np.cos(z**2)

    S, C = sp.special.fresnel(np.sqrt(2 * np.pi))
    i = 2 * np.sqrt(2) * np.pi ** (3 / 2) * C**2 * S
    i_n = diff.fftdiff.integral(f, dom.volume, fft)

    assert np.abs(i - i_n) < INTEGRAL_PRECISION_256
