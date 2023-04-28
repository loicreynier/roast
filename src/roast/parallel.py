"""Parallel (MPI) module (:mod:`roast.parallel`)."""

from h5py import File as H5File  # type: ignore
from mpi4py import MPI

from roast.utils import PathLike


class MPI4PY:
    """MPI4PY interface."""

    def __init__(self) -> None:
        self.comm = MPI.COMM_WORLD
        self.rank: int = self.comm.Get_rank()
        self.size: int = self.comm.Get_size()
        self.finalize = MPI.Finalize


def h5file(path: PathLike, mode="a") -> H5File:
    """HDF5 like with MPI driver."""
    return H5File(path, mode, driver="mpio", comm=MPI4PY().comm)


def chunked_list(data: list) -> list:
    """Divided list for MPI threading."""
    comm = MPI4PY().comm
    if comm.rank == 0:
        chunks: list[list] = [[] for _ in range(comm.size)]
        for i, chunk in enumerate(data):
            chunks[i % comm.size].append(chunk)
    else:
        data = []
        chunks = []
    data = comm.scatter(chunks, root=0)
    return data
