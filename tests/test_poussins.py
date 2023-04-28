"""ROAST POUSSINS API tests."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from roast import bflow, domain, parallel, poussins


def test_read_write_bin_file(shape: tuple[int, int, int] = (8, 8, 8)) -> None:
    """Test reading and writing of POUSSINS output binary files."""
    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "plt.bin"
        t = np.pi / 4.0
        dom = domain.Domain(shape)
        flow = bflow.vdinse_solexact(dom, t)
        data_1 = (t, 0, *dom.shape, *dom.axes, *flow)
        poussins.write_data_to_bin_file(path, data_1)
        data_2 = poussins.data_bin_file(path)
        for i in range(len(data_1)):
            assert np.allclose(data_1[i], data_2[i])


def test_read_write_config_file() -> None:
    """Test reading and writing of POUSSINS config file."""
    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "config.nml"
        poussins.write_config_file(path, poussins.CONFIG_DEFAULT)
        cfg = poussins.config(path.parent)
        assert cfg == poussins.CONFIG_DEFAULT


def test_create_simulation(shape: tuple[int, int, int] = (8, 8, 8)) -> None:
    """Test the creation of a POUSSINS simulation object."""
    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)
        dom = domain.Domain(shape)
        t = np.linspace(0, np.pi / 4.0, 10)

        # Generate files to load
        snaps = [bflow.vdinse_solexact(dom, t_i) for t_i in t]
        for i, snap in enumerate(snaps):
            poussins.write_data_to_bin_file(
                path / f"plt_{i:04d}.bin",
                (t[i], i, *dom.shape, *dom.axes, *snap),
            )
        cfg = poussins.CONFIG_DEFAULT
        cfg["nx"] = shape[0]
        cfg["ny"] = shape[1]
        cfg["nz"] = shape[2]
        cfg["tprint"] = 1
        cfg["name"] = "test"
        poussins.write_config_file(path / "config.nml", cfg)

        # Create the simulation object
        df = parallel.h5file(path / "data.hdf5")
        simu = poussins.Simulation(
            "test",
            df,
            path=path,
            data=(0, 10, 1),
        )

        # Compare data
        for i, snap_id in enumerate(simu.snap_list()):
            snap = simu.snaps[snap_id]
            assert snap.attrs["file"] == i
            assert snap.attrs["iter"] == i
            for var, array in zip(poussins.VARS, snaps[i][-5:]):
                assert np.allclose(snap[var][:], array)
        for key in sorted(list(simu.config.keys())):
            if key in ["_loaded", "shape"]:
                continue
            assert simu.config[key] == cfg[key]

        df.close()
