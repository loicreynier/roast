"""
Run module (:mod:`roast.run`)
=============================

Provides wrappers to perform calculations over snapshots of a ROAST
:class:`roast.simulation.Simulation` object.
"""

from typing import Callable

from h5py import Group  # type: ignore
from numpy.typing import DTypeLike

from roast import parallel
from roast.fspace import FSpace
from roast.parallel import MPI4PY
from roast.simulation import Simulation, SimulationSnapID, SimulationSnapRange

__all__ = [
    "compute",
]


MPI = MPI4PY()


def _snap_list(
    simu: Simulation,
    data: SimulationSnapID | SimulationSnapRange = None,
) -> list[str]:
    """Simulation snapshot list."""
    if data is not None:
        if isinstance(data, tuple):
            snaps = simu.snap_list(range=data)
        else:
            snaps = [str(data)]
    else:
        snaps = simu.snap_list()
    return snaps


def compute(  # noqa: PLR0912
    simu: Simulation,
    fun: Callable,
    *args,
    data: SimulationSnapID | SimulationSnapRange = None,
    fs: FSpace | None = None,
    store_params: dict[str, tuple[tuple, DTypeLike]] | None = None,
    **kwargs,
) -> list | None:
    """Compute `fun` on snapshots of `simu` with MPI threading.

    Parameters
    ----------
    simu: Simulation
        Simulation to run computation on.
    fun: Callable
        Computation function.
    *args: Any
        Function `fun` arguments
    data: SimulationSnapID | SimulationSnapRange, optional
        Snapshot data to run computation on. If not specified,
        computation is run over all snapshots.
    fs: FSpace, optional
        Function space associated with the simulation or computation.
        If not specified, a function space is generated from `simu`
        shape and domain.
    store_params: dict["str", tuple[tuple, DTypeLike]], optional
        Dictionary of which the keys are the names and the values are a
        a tuple of the size and the type of the dataset used to store
        the result in the ``simu.data`` HDF5 file.
        If not specified, result are not stored but returned in a list.
    **kwargs : Any, optional
        Function `fun` keyword arguments.

    Returns
    -------
    results : list
        List of the results. Only return if `store_params` hasn't be
        specified.

    Warnings
    --------
    Computation without storing (meaning with results returned) is not
    supported with MPI threading. Some kind of gathering should be
    implemented for support it, so that every thread returns all the
    results.

    Examples
    --------
    >>> df = roast.parallel.h5file("simu.hdf5")
    >>> simu = roast.poussins.Simulation.from_file(df)
    >>> roast.run.compute(
    ...     simu,
    ...     stats.rms,
    ...     "u",
    ...     "v",
    ...     "w",
    ...     store_params={"rms_u": ((), np.dtype(float))},
    ... )
    """
    res: list | None = None
    res_list: list = []

    # Snap list, will be splited among threads later
    snaps = _snap_list(simu, data=data)

    # If storing, allocate dataset first
    if store_params:
        for snap_id in snaps:
            snap: Group = simu.snaps.require_group(str(snap_id))
            for key, val in store_params.items():
                snap.require_dataset(
                    key,
                    val[0],  # Shape
                    val[1],  # DType
                    exact=True,
                )

    # Create associated function space, if necessary
    if not fs:
        fs = FSpace.from_simu(simu)

    # Split data among threads
    snaps = parallel.chunked_list(snaps)

    for snap_id in snaps:
        snap = simu.snaps.require_group(str(snap_id))

        # Filter variables which have the same shape has the simulation with
        # the function space
        fs_args = []
        for arg in args:
            if isinstance(arg, str):
                if (snap[arg].shape == simu.shape).all():
                    fs_args.append(fs(snap[arg][:]))  # type: ignore
            else:
                fs_args.append(arg)

        out = [fun(*fs_args, **kwargs)]

        if store_params:
            for i, (key, val) in enumerate(store_params.items()):
                # If scalar
                if len(val[0]) == 0:
                    snap[key][()] = out[i]
                # If array
                else:
                    snap[key][:] = out[i]
        else:
            res_list.append(out)

    # If storing, don't return
    if store_params:
        res = None
    else:
        res = res_list
    return res
