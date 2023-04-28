"""Utils module (:mod:`roast.utils`)."""

import os
import typing

PathLike = typing.Union[str, bytes, os.PathLike]
StrPath = typing.Union[str, os.PathLike]


def rounded_float(num: float, ndigits: int = 1) -> str:
    """Return string of the float number `num` rounded at `ndigits`."""
    return f"{round(num, ndigits):0.{ndigits}f}"
