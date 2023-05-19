"""
Function space module (:mod:`roast.fspace`)
===========================================

.. currentmodule:: roast.fspace

Provides the :class:`FSpace` class.
"""

__all__ = [
    "FSpace",
]


from typing import Literal

from numpy.typing import NDArray

from roast import fft
from roast.diff import fftdiff
from roast.domain import Domain
from roast.simulation import Simulation


class FSpace:
    """Function space using FFT for differentiation.

    Provides an interface to perform operation on 3D flow fields
    arrays such as differentiation.

    Most of ROAST functions requires a instance of :class:`FSpace` as
    argument.

    Attributes
    ----------
    domain : :class:`roast.domain.Domain`
        Associated computation domain.
    shape : tuple[int, int, int]
        Shape of the 3D flow fields.

    Parameters
    ----------
    domain: :class:`roast.domain.Domain`
        Associated computation domain.
    fft_lib : {"numpy", "pyfftw"}
        FFT library.
    """

    def __init__(
        self,
        domain: Domain,
        fft_lib: Literal["numpy", "pyfftw"] = "numpy",
    ) -> None:
        self.domain = domain

        # FFT initialization
        if fft_lib.lower() == "numpy":
            self.fft = fft.RFFT(self.domain.axes)
        elif fft_lib.lower() == "pyfftw":
            self.fft = fft.RFFTW(self.domain.axes)
        else:
            raise ValueError(f'"{fft_lib}" FFT library unsupported')

        self.rot = self.curl

    def __call__(self, var: NDArray) -> NDArray:
        """`var` adapted to FSpace grid/box."""
        return var

    def diff(self, u: NDArray, direction: int) -> NDArray:
        """Derivation of `u` along `direction`."""
        return fftdiff.diff(u, direction, self.fft)

    def grad(self, u: NDArray) -> NDArray:
        """Gradient of u."""
        return fftdiff.grad(u, self.fft)

    def div(self, u: NDArray, v: NDArray, w: NDArray) -> NDArray:
        """Divergence of the vector field (`u`, `v`, `w`)."""
        return fftdiff.div(u, v, w, self.fft)

    def curl(self, u: NDArray, v: NDArray, w: NDArray) -> NDArray:
        """Curl of the vector field (`u`, `v`, `w`)."""
        return fftdiff.curl(u, v, w, self.fft)

    def lap(self, u: NDArray) -> NDArray:
        """Laplacian of `u`."""
        return fftdiff.lap(u, self.fft)

    def lap_inv(self, u: NDArray, tol: float = 1e-6) -> NDArray:
        """Inverse Laplacian of `ù`."""
        return fftdiff.lap_inv(u, self.fft, tol=tol)

    def ugradu(self, u: NDArray, v: NDArray, w: NDArray) -> NDArray:
        """Convection term of vector field (`u`, `v`, `w`)."""
        return fftdiff.ugradu(u, v, w, self.fft)

    def ugradv(
        self,
        u1: NDArray,
        u2: NDArray,
        u3: NDArray,
        v1: NDArray,
        v2: NDArray,
        v3: NDArray,
    ) -> NDArray:
        r"""Convection term :math:`(\vec{u}\cdot\nabla)\vec{v}`."""
        return fftdiff.ugradv(u1, u2, u3, v1, v2, v3, self.fft)

    @classmethod
    def from_simu(
        cls,
        simu: Simulation,
        fft_lib: Literal["numpy", "pyfftw"] = "numpy",
    ):
        """Construct function space from a :class:`Simulation`."""
        return cls(simu.domain, fft_lib=fft_lib)
