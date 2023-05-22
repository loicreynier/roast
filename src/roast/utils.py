"""Utils module (:mod:`roast.utils`)."""

import os
import typing

from numpy.typing import NDArray

PathLike = typing.Union[str, bytes, os.PathLike]
StrPath = typing.Union[str, os.PathLike]


def rounded_float(num: float, ndigits: int = 1) -> str:
    """Return string of the float number `num` rounded at `ndigits`."""
    return f"{round(num, ndigits):0.{ndigits}f}"


def index_from_value(x: NDArray, x0: float) -> int:
    """`x` index for which `x` is close to `x0`."""
    ind = -1
    n = len(x)
    xx = abs(x - x0)
    mini = min(xx)
    for i in range(n):
        if xx[i] == mini:
            ind = i
            break
    if ind == -1:
        raise ValueError(
            f"Could not find a value close to {x0} in input data array"
        )
    return ind
