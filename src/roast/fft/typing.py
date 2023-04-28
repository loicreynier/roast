"""Types annotations for FFT module (:mod:`roast.typing`)."""

from typing import Union

from ._npfft import FFT, RFFT
from ._pyfftw import FFTW, RFFTW

ROASTFFT = Union[
    FFT,
    RFFT,
    FFTW,
    RFFTW,
]
