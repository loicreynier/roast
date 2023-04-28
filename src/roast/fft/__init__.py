"""
FFT module (:mod:`roast.fft`)
=============================

.. currentmodule:: roast.fft

Provides FFT interface classes :class:`FFT`, :class:`RFFT`,
:class:`FFTW` and :class:`RFFTW` for the computation of Fourier
transforms using NumPy or PyFFTW API.

.. warning::

   By default, spectral fields are transposed with respect to physical
   fields (similar fashion as in the POUSSINS code). As an example,
   the real ``FORWARD`` FFT of a field of shape ``(256, 129, 64)`` has
   a shape of ``(64, 128, 129)``.

"""

from . import typing
from ._npfft import FFT, RFFT
from ._pyfftw import FFTW, RFFTW

__all__ = [
    "FFT",
    "RFFT",
    "FFTW",
    "RFFTW",
    "typing",
]
