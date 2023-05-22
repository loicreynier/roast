"""
Visualization module (:mod:`roast.plot`)
========================================

Provides functions to visualize 3D fields.
"""


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.contour import QuadContourSet
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from numpy.typing import NDArray

from roast import utils
from roast.fspace import FSpace
from roast.utils import StrPath

__all__ = [
    "line",
    "crossview",
    "crossview_data",
    "show_figure",
    # Legacy names
    "p1d",
    "p2d",
]

AXES = {
    "x": 0,
    "y": 1,
    "z": 2,
}
SECS = {
    "yz": 0,
    "xz": 1,
    "xy": 2,
}


def line(
    u: NDArray[np.float_],
    axis: int,
    fs: FSpace,
    p1: float = 0.0,
    p2: float = 0.0,
) -> Line2D:
    """Plot a line of `u` in the direction (̀̀ axis`, `p1`, `p2`)

    Parameters
    ----------
    u : NDArray
        Data field.
    axis : int
        Plot direction.
    fs : FSpace
        Associated function space to access coordinates axes.
    p1, p2: float, optional
        Intersection points.

    Returns
    -------
    line : Line2D
        Line representing the plotted data.
    """
    if axis == AXES["x"]:
        i = utils.index_from_value(fs.domain.axes[1], p1)
        j = utils.index_from_value(fs.domain.axes[2], p2)
        u = u[:, i, j]
    elif axis == AXES["y"]:
        i = utils.index_from_value(fs.domain.axes[0], p1)
        j = utils.index_from_value(fs.domain.axes[2], p2)
        u = u[i, :, j]
    elif axis == AXES["z"]:
        i = utils.index_from_value(fs.domain.axes[0], p1)
        j = utils.index_from_value(fs.domain.axes[1], p2)
        u = u[i, :, j]
    else:
        raise ValueError(
            f"Unsupported axis '{axis}'. Must be either 0, 1 or 2."
        )
    x = fs.domain.axes[axis]
    return plt.gca().plot(x, u)


p1d = line


def crossview_data(
    u: NDArray[np.float_],
    section: int,
    fs: FSpace,
    x0: float = 0.0,
) -> tuple[
    NDArray[np.float_],
    NDArray[np.float_],
    NDArray[np.float_],
    tuple[NDArray[np.float_], NDArray[np.float_]],
]:
    """Data used for plotting a cross-sectional view of `u` in the
    (`section`, `x0`) plane.

    Parameters
    ----------
    u : NDArray
        Data field.
    axis : int
        Plot direction.
    section : int, {0, 1, 2}
        Cross-section.
    fs : FSpace
        Associated function space to access coordinates axes.
    x0 : float, optional
        Intersection point.

    Results
    -------
    x, y : NDArray
        Coordinates meshgrids.
    u : NDArray
        Plotting data.
    ticks : tuple[NDArray, NDArray]
        Plotting ticks.
    """
    x, y, z = fs.domain.axes
    if section == SECS["yz"]:
        x1, x2 = np.meshgrid(y, z)
        ind = utils.index_from_value(x, x0)
        u = u[ind]
        ticks = (
            np.linspace(y[0], y[-1], 6),
            np.linspace(z[0], z[-1], 6),
        )
    elif section == SECS["xz"]:
        x1, x2 = np.meshgrid(x, z)
        ind = utils.index_from_value(y, x0)
        u = u[:, ind, :]
        ticks = (
            np.linspace(x[0], x[-1], 6),
            np.linspace(z[0], z[-1], 6),
        )
    elif section == SECS["xy"]:
        x1, x2 = np.meshgrid(x, y)
        ind = utils.index_from_value(z, x0)
        u = u[:, :, ind]
        ticks = (
            np.linspace(x[0], x[-1], 6),
            np.linspace(y[0], y[-1], 6),
        )
    else:
        raise ValueError(
            f"Unsupported section '{section}'. Must be either 0, 1 or 2."
        )
    return x1, x2, u.T, ticks


def crossview(
    u: NDArray[np.float_],
    section: int,
    fs: FSpace,
    x0: float = 0.0,
    cmap: str = "inferno",
    levels: int = 20,
) -> QuadContourSet:
    """Plot a cross-sectional view of `u` in the (`section`, `x0`) plane.

    Parameters
    ----------
    u : NDArray
        Data field.
    section : int, {0, 1, 2}
        Cross-section.
    fs : FSpace
        Associated function space to access coordinates axes.
    x0 : float, optional
        Intersection point.
    levels: int or array-like, optional
        Contour levels.
    cmap: str, optional
        Colormap used for the filling.

    Returns
    -------
    contour : QuadContourSet
    """
    x1, x2, u, _ = crossview_data(u, section, fs, x0=x0)
    return plt.gca().contourf(x1, x2, u, levels=levels, cmap=cmap)


p2d = crossview


def export_crossview_data(
    path: StrPath,
    u: NDArray[np.float_],
    section: int,
    fs: FSpace,
    x0: float = 0.0,
    delimiter: str = " ",
) -> None:
    """Export data used for plotting a cross-sectional view of `u` in
    the (`section`, `x0`) plane.

    This function should be used to generate data for other plotting
    utilities such as LaTeX pgfplots package.

    Parameters
    ----------
    u : NDArray
        Data field.
    section : int, {0, 1, 2}
        Cross-section.
    fs : FSpace
        Associated function space to access coordinates axes.
    x0 : float, optional
        Intersection point.
    """
    x1, x2, u, _ = crossview_data(u, section, fs, x0=x0)
    np.savetxt(
        path,
        np.array((x1.flatten(), x2.flatten(), u.flatten())).T,
        delimiter=delimiter,
    )


def show_figure(fig: Figure) -> None:
    """Display figure `fig`.

    Display a figure even if it has been closed.

    Parameters
    ----------
    fig : Figure
        Figure to display.
    """
    dummy = plt.figure()
    new_manager = dummy.canvas.manager
    new_manager.canvas.figure = fig
    fig.set_canvas(new_manager.canvas)
    plt.show()
