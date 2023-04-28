"""FFT computation interface using PyFFTW API."""

__all__ = [
    "FFTW",
    "RFFTW",
]

import pyfftw  # type: ignore
from numpy.typing import NDArray

from ._npfft import FFT, RFFT


class FFTW(FFT):
    """3D complex FFT interface using PyFFTW API.

    Attributes
    ----------
    axes : tuple[ndarray, ndarray, ndarray]
        Cartesian coordinates vectors.
    shape_phys: tuple[int, int, int]
        Field shape in the physical space.
    shape_spec: tuple[int, int, int]
        Field shape in the spectral space.
    modes : ndarray
        Cartesian Fourier modes in reverse order.
    transposed : bool
        Whether the spectral field are transposed with respect to the
        physical fields.
    """

    def __init__(self, axes: tuple[NDArray, NDArray, NDArray]) -> None:
        super().__init__(axes)
        self._in = pyfftw.empty_aligned(self.shape_phys, dtype="complex")
        self._out = pyfftw.empty_aligned(self.shape_spec, dtype="complex")
        self._fft = pyfftw.builders.fftn(self._in)
        self._ifft = pyfftw.builders.ifftn(self._out)

    def fft(self, u: NDArray) -> NDArray:
        """3D FORWARD FFT of `u`."""
        return self._fft(u).T

    def ifft(self, v: NDArray) -> NDArray:
        """3D BACKWARD FFT of `v`."""
        return self._ifft(v).T


class RFFTW(RFFT):
    """3D real FFT interface using PyFFTW API.

    Attributes
    ----------
    axes : tuple[ndarray, ndarray, ndarray]
        Cartesian coordinates vectors.
    shape_phys: tuple[int, int, int]
        Field shape in the physical space.
    shape_spec: tuple[int, int, int]
        Field shape in the spectral space.
    modes : ndarray
        Cartesian Fourier modes in reverse order.
    transposed : bool
        Whether the spectral field are transposed with respect to the
        physical fields.
    """

    def __init__(self, axes: tuple[NDArray, NDArray, NDArray]) -> None:
        super().__init__(axes)
        self._in = pyfftw.empty_aligned(self.shape_phys, dtype="float")
        self._out = pyfftw.empty_aligned(self.shape_spec, dtype="complex")
        self._fft = pyfftw.builders.rfftn(self._in, axes=(2, 1, 0))
        self._ifft = pyfftw.builders.irfftn(self._out)

    def fft(self, u: NDArray) -> NDArray:
        """3D FORWARD FFT of `u`."""
        return self._fft(u).T

    def ifft(self, v: NDArray) -> NDArray:
        """3D BACKWARD FFT of `v`."""
        return self._ifft(v).T
