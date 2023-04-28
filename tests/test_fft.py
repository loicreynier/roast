"""ROAST FFT API tests."""

import numpy as np
import pytest

from roast import bflow, domain
from roast.fft import FFT, FFTW, RFFT, RFFTW
from roast.fft.typing import ROASTFFT

FLOAT_PRECISION: float = 1e-12


@pytest.mark.parametrize("fft_class", [FFT, FFTW, RFFT, RFFTW])
def test_round_trip(
    fft_class: ROASTFFT,
    shape: tuple[int, int, int] = (512, 256, 128),
) -> None:
    """Test FFT round trip on a Gaussian field."""
    dom = domain.Domain(shape)
    fft: ROASTFFT = fft_class(dom.axes)
    fun, _, _, _, _, _ = bflow.gaussian(dom)
    fun_c = fft.fft(fun)
    fun_n = fft.ifft(fun_c)
    assert (
        np.linalg.norm(fun - fun_n.real) / np.linalg.norm(fun)
        < FLOAT_PRECISION
    )
