"""
POUSSINS module (:mod:`roast.poussins`)
=======================================

Provides an API to load data from POUSSINS output and configuration
files into a ROAST :class:`roast.simulation.Simulation` object.
"""

__all__ = [
    "bin_file_name",
    "bin_file_paths_from_ids",
    "CONFIG_FILE_NAME",
    "CONFIG_FILE_NAME_OLD",
    "CONFIG_DEFAULT",
    "config",
    "data_nml_file",
    "data_ns_file",
    "data_bin_file",
    "id_list",
    "POUSSINSBinData",
    "POUSSINSFileIDs",
    "config",
    "Simulation",
    "VARS",
    "write_config_file",
    "write_data_to_bin_file",
]

import glob
import itertools
import os
import re
import struct
from collections import OrderedDict
from pathlib import Path
from typing import IO, Any

import f90nml  # type: ignore
import numpy as np
from h5py import File as H5File  # type: ignore
from h5py import Group as H5Group  # type: ignore
from numpy.typing import NDArray

from roast import simulation
from roast.domain import Domain
from roast.parallel import MPI4PY
from roast.simulation import SimulationConfig
from roast.utils import PathLike, StrPath

POUSSINSBinData = tuple[
    float,
    int,
    int,
    int,
    int,
    NDArray,
    NDArray,
    NDArray,
    NDArray,
    NDArray,
    NDArray,
    NDArray,
    NDArray,
]
POUSSINSFileIDs = tuple[int, int, int] | list[int]

MPI = MPI4PY()

CONFIG_FILE_NAME = "config.nml"
CONFIG_FILE_NAME_OLD = "ns.inp"
CONFIG_DEFAULT: SimulationConfig = (  # Default POUSSINS config file
    OrderedDict(
        [
            ("nx", 256),
            ("ny", 256),
            ("nz", 256),
            ("lx", 2.0),
            ("ly", 2.0),
            ("lz", 2.0),
            ("re", 1000.0),
            ("sc", 1.0),
            ("s", 1.0),
            ("q", 0.0),
            ("init_base_flow", 4),
            ("init_perturb", 0),
            ("restart_file", 0),
            ("tmax", 10000),
            ("tprint", 1000),
            ("cfl", 0.25),
            ("dt0", 0.001),
            ("dt_var", False),
            ("int_scheme", 1),
            ("impl_visc_solving", False),
            ("press_scheme", 1),
            ("rho_filtering", False),
            ("p_rho_filter", 16.0),
            ("sigma_rho_filter", 18.0),
            ("k_rho_filter", 0.825),
            ("forcing_term", False),
            ("forcing_term_method", 2),
            ("forcing_term_init", 0.25),
            ("forcing_term_reinit", False),
            ("fringe_region", False),
            ("fringe_size", 0.25),
            ("xstart_fringe", -0.5),
            ("deltarise_fringe", 0.6),
            ("deltafall_fringe", 0.1),
            ("init_rho_turbiso", 1),
            ("d_ball_turbiso", 10.0),
            ("reinit_rho_turbiso", False),
            ("k_ravier", 1.0),
            ("n_ravier", 1.0),
            ("eps_ravier", 1e-06),
            ("mode_ravier", 1.0),
            ("d_tanh", 0.1),
            ("a_tanh", 0.001),
            ("k_tanh", 0.5),
            ("mode_tanh", 1.0),
        ]
    )
)

VARS = ["u", "v", "w", "rho", "p"]

LOG_FILES = [  # Log files we care about for post-processing
    "divu",
    "ener" "plt",
    "calcp",
    "viscovarsc",
    "iso",
    "forcing",
]

# Regexes for Parsing old configuration format
_NS_INT_REGEX = r"\d*"
_NS_FLT_REGEX = r"\d*\.*\d*"


class Simulation(simulation.Simulation):
    """POUSSINS simulation data container.

    Velocity, density and pressure fields are stored in the snapshots.
    Simulation time is also stored as a scalar dataset in the snapshots.

    Notes
    -----
    Although the time could have been stored in snapshot attributes,
    we preferred to store it as a scalar dataset so that it is more
    easily accessible by MPI threads.
    """

    def __init__(  # noqa: PLR0913
        self,
        name: str,
        bind: H5File,
        path: StrPath | None = None,
        data: POUSSINSFileIDs | None = None,
    ) -> None:
        # Check if `bind` already contains simulation data
        if "_loaded" in bind.attrs:
            new = False
        else:
            new = True

        PathMissingError = ValueError(
            "Bind file data `path` keyword argument missing."
        )

        if new and not path:
            raise PathMissingError

        cfg = {}
        if new:
            if path:
                cfg = config(path)
                shape: tuple[int, int, int] = (
                    cfg["nx"],
                    cfg["ny"],
                    cfg["nz"],
                )
                cfg["shape"] = shape
                box_sizes = cfg["lx"], cfg["ly"], cfg["lz"]
            else:
                raise PathMissingError
        else:
            shape = bind.attrs["shape"]
            box_sizes = bind.attrs["lx"], bind.attrs["ly"], bind.attrs["lz"]

        super().__init__(
            name,
            shape,
            Domain(
                shape,
                bounds=tuple(  # type:ignore
                    (-np.pi * n / 2.0, np.pi * n / 2.0) for n in box_sizes
                ),
            ),
            bind,
        )
        self.logs: H5Group = self.data.require_group("logs")
        if new:
            if path:
                self.load_logs(path)
                for key, val in cfg.items():
                    self.config[key] = val
                self.config["_loaded"] = True
            else:
                raise PathMissingError

        if data:
            if path:
                self.load_data(path, data)
            else:
                raise PathMissingError

    @classmethod
    def from_file(
        cls,
        h5file: H5File,
        **kwargs,
    ):
        """Construct simulation from HDF5 file."""
        return cls(
            h5file.attrs["name"],
            h5file,
            **kwargs,
        )

    # Does not work with multiple MPI threads
    # def __del__(self) -> None:
    #     """Close associated data file."""
    #     self.data.file.close()

    def load_data(self, path: StrPath, ids: POUSSINSFileIDs) -> None:
        """Load data from `ids` in :attr:`path`.

        Data is loaded into the HDF5 file :attr:`data`.

        Parameters
        ----------
        ids: list[int] | tuple[int, int, int]
            File IDs list or tuple of the first file ID, last ID and
            the step between each file ID.
        """
        path = Path(path)
        ids = id_list(ids)
        paths = bin_file_paths_from_ids(ids, path=path)

        # Remove non-existent paths
        for i, path in enumerate(paths):
            if not path.exists():
                print(f"'{path}' does not exist'")
                paths.pop(i)
                ids.pop(i)

        # All groups and datasets must be first created by all the MPI threads
        for id_, path in zip(ids, paths):
            snap = self.snaps.require_group(str(id_))
            snap.attrs["file"] = id_
            snap.attrs["iter"] = id_ * self.config["tprint"]
            snap.require_dataset("t", shape=(), dtype=np.dtype(float))
            for var in VARS:
                snap.require_dataset(
                    var,
                    shape=self.shape,
                    dtype=np.dtype(float),
                )

        # Split the file IDs & paths lists for MPI threads
        if MPI.rank == 0:
            chunks: list[list[Any]] = [[] for _ in range(MPI.size)]
            for i, (id_, path) in enumerate(zip(ids, paths)):
                chunks[i % MPI.size].append((id_, path))
        else:
            chunks = []
        files: list[tuple[int, Path]] = MPI.comm.scatter(chunks, root=0)
        # print(MPI.rank, files)

        # Load the data, for real
        for id_, path in files:
            data = data_bin_file(path)
            snap = self.snaps[str(id_)]
            snap["t"][()] = data[0]
            for var, array in zip(VARS, data[-5:]):
                snap[var][:] = array

        MPI.comm.barrier()

        # Update snap keys
        # (done after because moving should be done with all threads ?)
        # -> Not working with large number of threads
        # for id_ in ids:
        #     snap = self.snaps[str(id_)]
        #     self.snaps.move(
        #         str(id_),
        #         rounded_float(snap["t"][()], ndigits=3),
        #     )

    def delete_data(self, snap_keys: list[int]) -> None:
        """Delete data from snapshots."""
        for key in snap_keys:
            del self.snaps[key]

    def load_logs(self, path: StrPath) -> None:
        """Load log files from simulation directory."""
        path = Path(path)
        for file in glob.glob(str(path) + "/log_*.txt"):
            if (
                # Get basename, remove extension and remove "log_"
                name := os.path.splitext(os.path.basename(file))[0][4:]
            ) in LOG_FILES:
                data = np.loadtxt(file, unpack=True)
                # Only add if not empty
                # (forcing file may be empty depending of the forcing method)
                if data.size != 0:
                    if name in self.logs:
                        del self.logs[name]
                    self.logs.require_dataset(
                        name,
                        shape=data.shape,
                        dtype=np.dtype(float),
                        data=data,
                    )
            else:
                pass


def _float_from_bin_file(bfile: IO) -> float:
    """Float loaded from binary file `bfile`."""
    _ = struct.unpack("i", bfile.read(4))
    n: float = struct.unpack("d", bfile.read(8))[0]
    _ = struct.unpack("i", bfile.read(4))
    return n


def _array_from_bin_file(bfile: IO, shape: tuple[int, ...]) -> NDArray:
    """Array of shape `shape` loaded from binary file `bfile`."""
    a = np.zeros(shape)
    for index in itertools.product(*[range(s) for s in shape]):
        # `index` is reversed since POUSSINS (Fortran) writes in z-y-x order
        a[index[::-1]] = _float_from_bin_file(bfile)
    return a


def _write_float_to_bin_file(bfile: IO, n: float) -> None:
    """Write `n` to binary file `bfile`.

    Examples
    --------
    >>> t = 2.4
    >>> with open("/tmp/test.bin", "wb") as bf:
    ...     roast.poussins._write_float_to_bin_file(t)
    """
    bfile.write(struct.pack("i", 0))
    bfile.write(struct.pack("d", n))
    bfile.write(struct.pack("i", 0))


def data_bin_file(path: PathLike) -> POUSSINSBinData:
    """Simulation data from POUSSINS binary output file `path`.

    Parameters
    ----------
    path : PathLike
        Binary file path.

    Returns
    -------
    t : float
        Non-dimensional physical time
    i : int
        Simulation iteration number
    n_x : int
        Number of mesh points in the 1st direction.
    n_y : int
        Number of mesh points in the 2nd direction.
    n_z : int
        Number of mesh points in the 3rd direction.
    x : ndarray
        1st direction coordinate vector.
    y : ndarray
        2nd direction coordinate vector.
    z : ndarray
        3rd direction coordinate vector.
    u : ndarray
        Velocity field 1st direction component.
    v : ndarray
        Velocity field 2nd direction component.
    w : ndarray
        Velocity field 3rd direction component.
    rho : ndarray
        Density field.
    p : ndarray
        Pressure field.

    Examples
    --------
    >>> data = roast.poussins.data_bin_file("./plt.bin")

    See Also
    --------
    write_data_to_bin_file
    """
    with open(path, "rb") as bfile:
        t = _float_from_bin_file(bfile)
        i = int(_float_from_bin_file(bfile))
        n_x = int(_float_from_bin_file(bfile))
        n_y = int(_float_from_bin_file(bfile))
        n_z = int(_float_from_bin_file(bfile))

        x = _array_from_bin_file(bfile, (n_x,))
        y = _array_from_bin_file(bfile, (n_y,))
        z = _array_from_bin_file(bfile, (n_z,))

        shape_3d = (n_z, n_y, n_x)
        u = _array_from_bin_file(bfile, shape_3d)
        v = _array_from_bin_file(bfile, shape_3d)
        w = _array_from_bin_file(bfile, shape_3d)
        rho = _array_from_bin_file(bfile, shape_3d)
        p = _array_from_bin_file(bfile, shape_3d)

    return t, i, n_x, n_y, n_z, x, y, z, u, v, w, rho, p


def write_data_to_bin_file(path: PathLike, data: POUSSINSBinData) -> None:
    """Write data to binary file in `path`.

    Data is a tuple containing:
      * ``t`` - time (``float``)
      * ``i`` - iteration number (``int``)
      * ``n_x`` - number of mesh points in the 1st direction (``int``)
      * ``n_y`` - number of mesh points in the 2nd direction (``int``)
      * ``n_z`` - number of mesh points in the 3rd direction (``int``)
      * ``x`` - 1st direction coordinate vector (``ndarray``)
      * ``y`` - 2nd direction coordinate vector (``ndarray``)
      * ``z`` - 3rd direction coordinate vector (``ndarray``)
      * ``u`` - Velocity field 1st direction component (``ndarray``)
      * ``v`` - Velocity field 2nd direction component (``ndarray``)
      * ``w`` - Velocity field 3rd direction component (``ndarray``)
      * ``rho`` - Density field (``ndarray``)
      * ``p`` - Pressure field (``ndarray``)

    Examples
    --------
    >>> t = np.pi / 4.0
    >>> shape = (32,) * 3
    >>> dom = roast.domain.Domain(shape)
    >>> flow = roast.bflow.vdinse_solexact(dom, t, s=4.0)
    >>> data = (t, i, *dom.shape, *dom.axes, *flow)
    >>> roast.poussins.write_data_to_bin_file("./plt.bin", data)

    See Also
    --------
    data_bin_file
    """
    t, i, n_x, n_y, n_z, x, y, z, u, v, w, rho, p = data
    with open(path, "wb") as bfile:
        for var in [t, i, n_x, n_y, n_z]:
            _write_float_to_bin_file(bfile, float(var))

        for i in range(n_x):
            _write_float_to_bin_file(bfile, x[i])
        for j in range(n_y):
            _write_float_to_bin_file(bfile, y[j])
        for k in range(n_z):
            _write_float_to_bin_file(bfile, z[k])

        # `ranges` are reversed since POUSSINS (Fortran) writes in z-y-x order
        for var, k, j, i in itertools.product(
            [u, v, w, rho, p], range(n_z), range(n_y), range(n_x)
        ):
            _write_float_to_bin_file(bfile, var[i, j, k])  # type: ignore


def bin_file_name(number: int) -> str:
    """Path of `number` POUSSINS binary output file.

    Examples
    --------
    >>> number = 4
    >>> print(roast.poussins.bin_file_name(number))
    plt_0004.bin
    """
    return f"plt_{number:04d}.bin"


def bin_file_paths_from_ids(
    ids: list[int],
    path: Path = Path("./"),
) -> list[Path]:
    """List of binary file paths corresponding to file `ids`.

    Parameters
    ----------
    ids : list
        File IDS list.
    path :
        Path where to look for files.

    Examples
    --------
    >>> roast.poussins.bin_file_paths_from_ids([0, 100])
    [PosixPath('plt_000.bin'), PosixPath('plt_100.bin')]
    """
    return [path / bin_file_name(i) for i in ids]


def id_list(ids: POUSSINSFileIDs) -> list[int]:
    """List of file IDS.

    Parameters
    ----------
    ids: list or tuple
        File IDs list or tuple of the first file ID, last ID and
        the step between each file ID.

    Examples
    --------
    >>> roast.poussins.id_list((0, 4, 1))
    [0, 1, 2, 3]
    """
    # Construct list if `ids` is a tuple for `range`
    if isinstance(ids, tuple) and len(ids) == 3:  # noqa: PLR2004
        ids = [i for i in list(range(*ids))]
    elif isinstance(ids, list):
        pass
    else:
        raise ValueError
    return ids


def data_nml_file(path: PathLike) -> OrderedDict[str, SimulationConfig]:
    """Configuration loaded from Fortran namelist file `path`.

    Notes
    -----
    The actual configuration is located in the ``"config"`` key of the
    returned data dict (see examples).

    Examples
    --------
    >>> cfg = roast.poussins.data_nml_file("config.nml")["config"]

    See Also
    --------
    data_ns_file
    """
    return f90nml.read(path).todict()


def data_ns_file(path: PathLike) -> SimulationConfig:
    """Configuration loaded from `path`.

    Notes
    -----
    This function loads data from the old POUSSINS configuration file
    ``ns.inp``. Most recent version of POUSSINS use the ``config.nml``
    configuration file which contains the same configuration written
    in a Fortran namelist.

    See Also
    --------
    data_nml_file
    """
    data: SimulationConfig = {}
    try:
        with open(path, "r", encoding="utf-8") as nsfile:
            lines = nsfile.readlines()
            data["nx"] = int(re.findall(_NS_INT_REGEX, lines[8])[0])
            data["ny"] = int(re.findall(_NS_INT_REGEX, lines[9])[0])
            data["nz"] = int(re.findall(_NS_INT_REGEX, lines[10])[0])
            data["lx"] = int(re.findall(_NS_INT_REGEX, lines[29])[0])
            data["ly"] = int(re.findall(_NS_INT_REGEX, lines[30])[0])
            data["lz"] = int(re.findall(_NS_INT_REGEX, lines[31])[0])
            data["Re"] = float(re.findall(_NS_FLT_REGEX, lines[33])[0])
            data["Sc"] = float(re.findall(_NS_FLT_REGEX, lines[34])[0])
            data["q"] = float(re.findall(_NS_FLT_REGEX, lines[35])[0])
            data["s"] = float(re.findall(_NS_FLT_REGEX, lines[36])[0])
            data["tf"] = int(re.findall(_NS_INT_REGEX, lines[12])[0])
            data["tp"] = int(re.findall(_NS_INT_REGEX, lines[13])[0])
    except IndexError as error:
        raise SyntaxError(f"'{path}' invalid") from error  # type: ignore
    return data


def config(path: StrPath) -> dict:
    """Configuration loaded from POUSSINS simulation directory `path`.

    Examples
    --------
    >>> cfg = roast.poussins.config("./")

    See Also
    --------
    data_nml_file
    data_ns_file
    write_config_file
    """
    path = Path(path)
    if os.path.isfile(path := path / CONFIG_FILE_NAME):
        cfg_dict = data_nml_file(path)["config"]
    elif os.path.isfile(path := path / CONFIG_FILE_NAME_OLD):
        cfg_dict = data_ns_file(path)
    else:
        raise OSError("no configuration file found")
    return cfg_dict


def write_config_file(path: PathLike, cfg: SimulationConfig) -> None:
    """Write config `cfg` into Fortran namelist file `path`.

    Examples
    --------
    >>> roast.poussins.write_config_file(
    ...     "config.nml",
    ...     roast.poussins.CONFIG_DEFAULT,
    ... )

    See Also
    --------
    data_nml_file
    """
    f90nml.write(OrderedDict([("config", cfg)]), path)
