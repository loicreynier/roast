"""FFT computation interface using NumPy API."""

__all__ = [
    "FFT",
    "RFFT",
    "modes_real",
    "modes_comp",
]

import numpy as np
from numpy.typing import NDArray


class FFT:
    """3D complex FFT interface using NumPy API.

    Attributes
    ----------
    axes : ndarray, ndarray, ndarray
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
        self.axes = axes
        self.shape_phys = tuple(axe.size for axe in axes)
        self.shape_spec = tuple(reversed(self.shape_phys))
        self.transposed = True
        self._init_modes()

    def __call__(
        self,
        u: NDArray,
        direction: str = "FORWARD",
    ) -> NDArray:
        """FFT of `u` along `direction`."""
        if direction == "FORWARD":
            v = self.fft(u)
        elif direction == "BACKWARD":
            v = self.ifft(u)
        else:
            raise ValueError
        return v

    def _init_modes(self) -> None:
        modes: list[NDArray] = [np.zeros([])] * 3
        for dir_ in range(3):
            if self.shape_phys[dir_] > 1:
                modes[dir_] = modes_comp(self.axes[dir_])
            else:
                modes[dir_] = np.asarray([0.0])
        self.modes: NDArray = np.array(modes[::-1], dtype=object)

    def fft(self, u: NDArray) -> NDArray:
        """3D FORWARD FFT of `u`."""
        return np.fft.fftn(u).T

    def ifft(self, v: NDArray) -> NDArray:
        """3D BACKWARD FFT of `v`."""
        return np.fft.ifftn(v).T


class RFFT:
    """3D real FFT interface using NumPy API.

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
        self.axes = axes
        self.shape_phys = tuple(axe.size for axe in axes)
        self.shape_spec = (axes[2].size, axes[1].size, axes[0].size // 2 + 1)
        self.transposed = True
        self._init_modes()

    def _init_modes(self) -> None:
        modes: list[NDArray] = [np.zeros([])] * 3
        for dir_ in range(3):
            dir_rev = abs(dir_ - 2)
            if self.shape_phys[dir_] > 1:
                modes[dir_] = (
                    modes_comp(self.axes[dir_rev])
                    if dir_ != 2  # noqa: PLR2004
                    else modes_real(self.axes[dir_rev])
                )
            else:
                modes[dir_] = np.asarray([0.0])
        self.modes: NDArray = np.array(modes, dtype=object)

    def fft(self, u: NDArray) -> NDArray:
        """3D FORWARD FFT of `u`."""
        return np.fft.rfftn(u, axes=(2, 1, 0)).T

    def ifft(self, v: NDArray) -> NDArray:
        """3D BACKWARD FFT of `v`."""
        return np.fft.irfftn(v).T


def modes_comp(axis: NDArray) -> NDArray:
    """Fourier modes along `axis`."""
    dim = axis.size
    length = axis.max() - axis.min() + axis[1] - axis[0]
    return np.concatenate(
        (
            np.arange(0, dim // 2 + 1),
            np.arange(-dim // 2 + 1, 0),
        )
    ) * (2.0 * np.pi / length)


def modes_real(axis: NDArray) -> NDArray:
    """Fourier real modes along `axis`."""
    dim = axis.size
    length = axis.max() - axis.min() + axis[1] - axis[0]
    return np.arange(0, dim // 2 + 1) * 2.0 * np.pi / length
