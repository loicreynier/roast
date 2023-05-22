"""
Domain module (:mod:`roast.domain`)
===================================
"""

__all__ = [
    "Domain",
]

import numpy as np
from numpy.typing import NDArray

Shape = tuple[int, int, int]
Bounds = tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
Axes = tuple[NDArray, NDArray, NDArray]


class Domain:
    """Base domain."""

    def __init__(
        self,
        shape: Shape,
        bounds: Bounds = ((-np.pi, np.pi),) * 3,
        periodic: tuple[bool, bool, bool] = (True, True, True),
    ) -> None:
        self.shape: Shape = shape
        self.periodic = periodic
        self.axes: tuple[NDArray, NDArray, NDArray] = tuple(  # type: ignore
            np.linspace(*bounds[i], shape[i], endpoint=not periodic[i])
            for i in range(3)
        )
        self.x, self.y, self.z = self.axes
        self.mgrid = np.meshgrid(self.x, self.y, self.z, indexing="ij")
        self.volume = np.prod([b[1] - b[0] for b in bounds])

    @classmethod
    def from_axes(
        cls,
        axes: Axes,
        periodic: tuple[bool, bool, bool],
    ):
        """Construct domain from axes coordinates arrays."""
        shape = [len(a) for a in axes]
        bounds = [(a[0], a[-1]) for a in axes]
        dom = cls(shape, bounds=bounds, periodic=(False,) * 3)  # type: ignore
        dom.periodic = periodic
        return dom
