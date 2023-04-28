"""FFT-based differentiation tests."""

import numpy as np
import pytest

from roast import bflow, diff
from roast.domain import Domain
from roast.fft import FFT, FFTW, RFFT, RFFTW
from roast.fft.typing import ROASTFFT

FLOAT_PRECISION: float = 1e-12


@pytest.mark.parametrize("fft_class", [FFT, FFTW, RFFT, RFFTW])
def test_fft_diff(
    fft_class: ROASTFFT, shape: tuple[int, int, int] = (512, 256, 128)
):
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
):
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
):
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
):
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
