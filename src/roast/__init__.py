"""
==============================================================
ROAST - Post-processing toolbox for turbulence DNS simulations
==============================================================

Provides functions/classes to load simulation data from the DNS code
POUSSINS used at the LMFA laboratory (Lyon, France) and functions to
post-process this data in the context of turbulence study.

Documentation
-------------

Documentation is available in two forms:

    1. docstrings provided within the code
    2. user guide online at https://loicreynier.github.io/roast

Use the built-in `help` function to view a function, class or module's
docstring::

    >>> help(roast.analysis.Analysis)
    ... # doctest: +SKIP

Examples within the docstrings assume that `roast` and `numpy` have been
imported

    >>> import roast
    >>> import numpy as np

Subpackages
-----------

ROAST provides the following modules/packages:

    bflow       Base flow related functions
    diff        Differentiations functions
    domain      Computation domain related classes and functions
    fft         FFT computation interfaces
    fspace      API for calculation on fields
    parallel    Utilities to run parallel computations with MPI
    poussins    API to load POUSSINS simulation data
    run         API to run over simulation snapshots
    stats       Statistics related functions
    simulation  Simulation data container related classes and functions
    utils       Miscellaneous utilities

Use `help(roast.module)` to know more about `module`.

All the aforementioned subpackages are imported while importing `roast`.
"""

from . import (  # noqa: 401
    bflow,
    domain,
    fspace,
    parallel,
    poussins,
    run,
    simulation,
    stats,
    turbu,
    utils,
)
