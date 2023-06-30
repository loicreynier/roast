"""
Turbulence module (:mod:`roast.turbu`)
======================================

Provides function to compute turbulence-related quantities.
"""

import itertools

import numpy as np
import scipy.stats
from numpy.typing import NDArray

from roast.fspace import FSpace

__all__ = [
    "energy_spectrum",
    "integral_scale",
    "isotropy",
    "kolmogorov_scale",
    "spectrum",
    "taylor_scale",
    "taylor_scale_iso",
    "tke_spectrum",
    "turnover_time",
]


def spectrum(
    u: NDArray,
    v: NDArray,
    w: NDArray,
    fs: FSpace,
):
    r"""
    Spectral density of energy of the vector field (`u`, `v`, `w`).

    Parameters
    ----------
    u : NDArray
        Velocity field 1st direction component.
    v : NDArray
        Velocity field 2nd direction component.
    w : NDArray
        Velocity field 3rd direction component.
    fs : FSpace
        Associated function space used for FFT computation.

    Returns
    -------
    E : NDArray[float]
        One-dimensional spetral density of energy.
    k : NDArray[float]
        One-dimensional Fourier modes array

    Notes
    -----
    Computation based on the derivation of section 3.2 of [1].

    The turbulent kinetic energy is defined as

    .. math::

       \text{TKE}(t) = \int_0^\infty E(k, t) \,\mathrm{d}k

    where the spectral density of energy is defined as a function
    of the spectral tensor :math:`\phi_{ii}(k, t)`

    .. math::

        E(k,t) = 2 \pi k^2 \phi_{ii}(k, t).

    References
    ----------
    .. [1] Cazalbou, Jean-Bernard, and Jérôme Fontane, 2021,
       "Physique de La Turbulence", Formation Ingénieur ISAE-SUPAERO,
       Filière Dynamique des Fluides.
    .. [2] user4557934, Nov 21th, 2015, "Computing turbulent energy
       spectrum from isotropic turbulence flow field in a box",
       Computational Science StackOverflow,
       https://scicomp.stackexchange.com/questions/21360
       (accessed September 07, 2021)
    .. [3] Lalylulelo, Nov 28th, 2016, answer Victor Piria on
       "Ways to compute the turbulent velocity kinetic energy spectrum"
       https://physics.stackexchange.com/questions/293638
       (accessed September 07, 2021)
    """
    k = np.zeros(fs.fft.shape_spec)
    E = np.copy(k)

    u_c = fs.fft.fft(u) / u.size
    v_c = fs.fft.fft(v) / v.size
    w_c = fs.fft.fft(w) / w.size

    for i in itertools.product(*[range(d) for d in fs.fft.shape_spec]):
        k[i] = np.sqrt(
            fs.fft.modes[0][i[0]] ** 2
            + fs.fft.modes[1][i[1]] ** 2
            + fs.fft.modes[2][i[2]] ** 2
        )
        E[i] = np.abs(u_c[i] ** 2 + v_c[i] ** 2 + w_c[i] ** 2)
    E = 2.0 * np.pi * E.reshape(-1)
    k = k.reshape(-1)

    stats, bin_edges, _ = scipy.stats.binned_statistic(
        k,
        E * k**2,
        statistic="mean",  # We don't want to sum `k^2`
        # statistic="sum",
        bins=max(u.shape + v.shape + w.shape),
    )
    bin_middles = np.zeros(*stats.shape)
    for i in range(*bin_middles.shape):
        bin_middles[i] = (bin_edges[i] + bin_edges[i + 1]) / 2.0

    E = stats[:]
    k = bin_middles[:]

    return E, k


def tke_spectrum(
    u: NDArray,
    v: NDArray,
    w: NDArray,
    fs: FSpace,
) -> tuple[NDArray, NDArray]:
    """Turbulent kinetic energy spectrum.

    Parameters
    ----------
    u : NDArray
        Velocity field 1st direction component.
    v : NDArray
        Velocity field 2nd direction component.
    w : NDArray
        Velocity field 3rd direction component.
    fs : FSpace
        Associated function space used for FFT computation.

    Returns
    -------
    E : NDArray
        One-dimensional energy spectrum
    k : NDArray
        One-dimensional Fourier modes array

    See Also
    --------
    energy_spectrum
    """
    return spectrum(u, v, w, fs)


def energy_spectrum(
    u: NDArray,
    v: NDArray,
    w: NDArray,
    rho: NDArray,
    fs: FSpace,
) -> tuple[NDArray, NDArray]:
    """Kinetic energy (Favrian energy) spectrum.

    Parameters
    ----------
    u : NDArray
        Velocity field 1st direction component.
    v : NDArray
        Velocity field 2nd direction component.
    w : NDArray
        Velocity field 3rd direction component.
    fs : FSpace
        Associated function space used for FFT computation.

    Returns
    -------
    E : NDArray
        One-dimensional energy spectrum
    k : NDArray
        One-dimensional Fourier modes array

    See Also
    --------
    tke_spectrum
    """
    r = np.sqrt(np.abs(rho))
    return spectrum(r * u, r * v, r * w, fs)


def isotropy(
    u: NDArray,
    v: NDArray,
    w: NDArray,
    fs: FSpace,
) -> float:
    """Isotropy degree.

    Parameters
    ----------
    u : NDArray
        Velocity field 1st direction component.
    v : NDArray
        Velocity field 2nd direction component.
    w : NDArray
        Velocity field 3rd direction component.
    fs : FSpace
        Associated function space used for FFT computation.

    Notes
    -----
    Isotropy degree is computed from the development
    section 5.3 page 28 of [1].

    References
    ----------
    .. [1] Curry, James H., Jackson. R. Herring, Josip Loncaric, and
       Steven A. Orszag. 1984. “Order and Disorder in Two- and
       Three-Dimensional Bénard Convection.” Journal of Fluid Mechanics
       147 (October): 1–38.
    """
    kk = np.array(np.meshgrid(*fs.fft.modes, indexing="ij"))
    e_1 = kk.copy()
    e_2 = kk.copy()
    v_c = np.zeros((3, *fs.fft.shape_spec), dtype="complex")
    psi_1 = np.zeros(fs.fft.shape_spec)
    psi_2 = psi_1.copy()

    for i, v_i in enumerate((w, v, u)):
        v_c[i] = fs.fft.fft(v_i)

    for k, j, i in itertools.product(
        *[range(1, n - 1) for n in fs.fft.shape_spec]
    ):
        m = (slice(None), k, j, i)
        n = m[1:]

        e_1[m] = np.cross(np.array([1.0, 0.0, 0.0]), kk[m])
        e_1[m] /= np.linalg.norm(e_1[m]) + 1e-10
        e_2[m] = np.cross(e_1[m], kk[m])
        e_2[m] /= np.linalg.norm(e_2[m]) + 1e-10

        psi_1[n] = np.abs(np.vdot(e_1[m], v_c[m])) ** 2
        psi_2[n] = np.abs(np.vdot(e_2[m], v_c[m])) ** 2

    return np.sqrt(np.sum(psi_1) / np.sum(psi_2))


def taylor_scale(E: NDArray, k: NDArray) -> float:
    """Taylor microscale computed from TKE spectrum.

    Parameters
    ----------
    E : NDArray
        One-dimensional energy spectrum.
    k : NDArray
        One-dimensional Fourier modes array.

    Returns
    -------
    lambda : float
        Taylor microscale.

    See Also
    --------
    taylor_scale_iso
    integral_scale
    kolmogorov_scale

    Notes
    -----
    Taylor scale is computed from equation 3.4.26 page 59
    of [1]. Integrals are computed with the trapezoidal rule.

    References
    ----------
    .. [1] Batchelor, George Keith. 1982. The Theory of Homogeneous
       Turbulence. Cambridge New York: Cambridge University Press.
    """
    return np.sqrt(5.0 * np.trapz(E, x=k) / np.trapz(E * k**2, x=k))


def taylor_scale_iso(u: NDArray, dudx: NDArray) -> float:
    """Isotropic flow Taylor microscale computed from velocity field.

    Parameters
    ----------
    u : NDArray
        Velocity field  component.
    dudx : NDArray
        Velocity field longitudinal derivative.

    See Also
    --------
    taylor_scale
    integral_scale
    kolmogorov_scale

    Notes
    -----
    Taylor scale is comupted from equation 9 page 34 of [1].

    References
    ----------
    .. [1] Kerr, Robert McDougall. 1985. “Higher-Order Derivative
       Correlations and the Alignment of Small-Scale Structures in
       Isotropic Numerical Turbulence.” Journal of Fluid Mechanics 153
       (April): 31–58.
    """
    return np.sqrt(np.mean(u * u) / np.mean(dudx * dudx))


def integral_scale(E: NDArray, k: NDArray) -> float:
    """Integral scale computed from TKE spectrum.

    Parameters
    ----------
    E : NDArray
        One-dimensional energy spectrum.
    k : NDArray
        One-dimensional Fourier modes array.

    Returns
    -------
    L : float
        Integral scale.

    See Also
    --------
    taylor_scale
    taylor_scale_iso
    kolmogorov_scale

    Notes
    -----
    Integral scale is computed from equation 3.2.5 page 39
    of [1]. Integrals are computed with the trapezoidal rule.

    References
    ----------
    .. [1] Batchelor, George Keith. 1982. The Theory of Homogeneous
       Turbulence. Cambridge New York: Cambridge University Press.
    """
    return 3.0 * np.pi / 4.0 * np.trapz(E / k, x=k) / np.trapz(E, x=k)


def kolmogorov_scale(L_0: float, l_t: float, Re: float = 1000) -> float:
    r"""Kolmogorov scale.

    Parameters
    ----------
    L_0 : float
        Integral scale.
    l_t : float
        Taylor microscale.
    Re : float
        Reynolds number.

    Returns
    -------
    eta : float
        Kolmogorov scale.

    Notes
    -----
    Kolmogorov scale is computed from the large scale Reynolds number as

    .. math::

       \eta = \frac{\lambda^2L^2}{15\mathrm{Re}^2}.

    This expression is obtained from the classical formula

    .. math::

       \eta = \left(\frac{\nu^3}{\epsilon}\right)^{\tfrac14}

    where the kinetic viscosity and the energy dissipation have been
    respectively substituted by

    .. math::

       \nu = \frac{u'L}{\mathrm{Re}}

    and

    .. math::

       \langle\epsilon\rangle = \frac{15\nu{}u'^2}{\lambda^2}.

    See Also
    --------
    taylor_scale
    taylor_scale_iso
    integral_scale
    """
    return ((L_0 * l_t) ** 2 / (15 * Re**2)) ** 0.25


def turnover_time(L_0: float, u: float) -> float:
    """Large eddy turnover time.

    Parameters
    ----------
    L_0 : float
        Integral scale.
    u : float
        Velocity field RMS.

    Returns
    -------
    t_0 : float
    """
    return L_0 / u
