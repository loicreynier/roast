"""
Simulation module (:mod:`roast.simulation`)
===========================================

Provides the base :class:`Simulation` data interface class.

ROAST's workflow stores simulation data into HDF5 files.
:class:`Simulation` provides an object-oriented interface to these
data files allowing easier data manipulation.

:mod:`roast.poussins` provides
:class:`roast.poussins.SimulationPOUSSINS`,
a subclass of :class:`Simulation` providing functions to load POUSSINS
simulation output into a ROAST :class:`Simulation`.
Other subclasses may be implemented to provide support for other
simulation code.
"""

__all__ = [
    "Simulation",
    "SimulationConfig",
]

from h5py import File, Group  # type: ignore

from roast.domain import Domain  # type: ignore

SimulationConfig = dict[str, int | float | str]
SimulationSnapID = str | int | float
SimulationSnapRange = (
    tuple[SimulationSnapID | None, SimulationSnapID | None] | None
)


class Simulation:
    """Simulation data interface.

    This interface is a wrapper around the linked HDF5 data file
    :attr:`data` providing function to access data and simulation
    configurations.

    The HDF5 data files has two top-level groups:
      * ``snaps``
      * ``logs``

    The ``snaps`` group contains snapshots of the flow at different
    simulation times. Each snapshot is stored in a subgroup of ``snaps``
    and contains datasets such as the velocity, density and pressure
    fields::

        data.hdf5
        └── /snaps
            ├── 1.0
            │   ├── u  (64, 64, 64), float64
            │   └── p  (64, 64, 64), float64
            └── 2.0
                ├── u  (64, 64, 64), float64
                └── p  (64, 64, 64), float64

    The ``logs`` groups contains...

    Parameters
    ----------
    name : str
        Simulation name/identifier.
    bind : File | Group
        HDF5 data container.
    create_group: bool
        Whether to create a top level `name` group.

    Attributes
    ----------
    config : HDF5 AttributeManager
        Simulation configuration variables.
    data : HDF5 File | HDF5 Group
        Data container where snapshots and config are stored.
    shape : tuple[int, int, int]
        Shape of the 3D flow fields.
    snaps : HDF5 Group
        Snapshot data.
    name : str
        Simulation name/identifier.

    Examples
    --------
    >>> import h5py
    >>> df = h5py.File("data.hdf5", "a")
    >>> shape = (256,) * 3
    >>> dom = roast.domain.Domain(shape)
    >>> simu = roast.simulation.Simulation(
    ...     "simu-1", shape, dom, df, create_group=True
    ... )
    >>> print(simu.data)
    <HDF5 group "/simu-1" (1 members)>
    >>> print(simu.data.keys())
    <KeysViewHDF5 ['snaps']>
    >>> df.close()
    >>> del simu

    See Also
    --------
    roast.poussins.SimulationPOUSSINS
    """

    def __init__(
        self,
        name: str,
        shape: tuple[int, int, int],
        dom: Domain,
        bind: File | Group,
        create_group: bool = False,
    ) -> None:
        self.name = name
        self.shape = shape
        self.domain = dom

        if create_group:
            bind = bind.require_group(name)
        self.data: File | Group = bind

        self.snaps: Group = self.data.require_group("snaps")
        self.config = self.data.attrs
        self.config["name"] = name
        self.config["shape"] = shape

    def snap_list(
        self,
        range: SimulationSnapRange = None,
    ) -> list[str]:
        """Sorted list of snapshot keys.

        Parameters
        ----------
        range : tuple[int, int], optional
            Index of the first and last snapshot to extract.
        """
        snaps = sorted(list(self.snaps.keys()), key=float)

        if range is not None:
            min, max = range
            if min is not None:
                i_min = snaps.index(str(min))
                snaps = snaps[i_min:]
            if max is not None:
                i_max = snaps.index(str(max))
                snaps = snaps[: i_max + 1]

        return snaps

    @classmethod
    def from_file(
        cls,
        h5file: File,
        **kwargs,
    ):
        """Construct simulation from HDF5 file."""
        shape = h5file.attrs["shape"]
        name = h5file.attrs["name"]
        return cls(
            name,
            shape,
            Domain(shape),
            h5file,
            **kwargs,
        )
