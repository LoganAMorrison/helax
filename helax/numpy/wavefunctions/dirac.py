import dataclasses

import numpy as np

from helax.numpy.typing import ComplexArray, RealArray


@dataclasses.dataclass
class DiracWf:
    """Dataclass for Dirac wavefunctions.

    Attributes
    ----------
    wavefunction: complex array
        4-Component Dirac spinor.
    momentum: array
        Four-momentum of the spinor. The momentum always points in the
        direction of the fermion flow.
    direction: array
        1 for a u,v spinors and -1 for ubar,vbar spinors.

    """

    wavefunction: ComplexArray
    momentum: RealArray
    direction: int


# =============================================================================
# ---- Dirac Wavefunction -----------------------------------------------------
# =============================================================================


def __check_spin_dirac(*, spin: int):
    assert spin == 1 or spin == -1, "Spin must be 1 or -1."


def __chi(*, momentum: RealArray, spin: int) -> ComplexArray:
    """
    Compute the two-component weyl spinor.

    Parameters
    ----------
    p: array
        Array containing the 4-momentum of the wavefunction.
    s: int
        Spin of the wavefunction. Must be 1 or -1.
    """
    dtype = momentum.dtype
    eps = np.finfo(dtype.name).eps

    px = momentum[1]
    py = momentum[2]
    pz = momentum[3]

    pm = np.linalg.norm(momentum[1:], axis=0)
    mask = pm + pz <= eps
    x = np.zeros((2, momentum.shape[-1]), dtype=dtype) * 1j

    if np.any(mask):
        # x[0, mask] = 0.0j  # (spin - 1.0) / 2.0 + 0j
        x[1, mask] = spin  # (spin + 1.0) / 2.0 + 0j

    mask = ~mask
    if np.any(mask):
        px = px[mask]
        py = py[mask]
        pz = pz[mask]
        pm = pm[mask]

        den = np.sqrt(2 * pm * (pm + pz))
        x[0, mask] = (pm + pz) / den
        x[1, mask] = (spin * px + py * 1j) / den

    if spin == -1:
        return np.array([x[1], x[0]])
    return np.array([x[0], x[1]])


def __dirac_spinor(
    *, momentum: RealArray, mass: float, spin: int, anti: int
) -> ComplexArray:
    """
    Compute the dirac wavefunction.

    Parameters
    ----------
    p:
        Four-momentum of the wavefunction.
    mass:
        Mass of the wavefunction.
    s: int
        Spin of the wavefunction. Must be 1 or -1.
    anti: Int
        If anti = -1, the wavefunction represents a v-spinor.
        If 1, wavefunction represents a u-spinor.
    """
    pm = np.linalg.norm(momentum[1:], axis=0)

    wp = np.sqrt(momentum[0] + pm)
    wm = mass / wp

    if spin == -anti:
        w = np.array([anti * wp, wm])
    else:
        w = np.array([wm, anti * wp])

    x = __chi(momentum=momentum, spin=spin * anti)

    return np.array([w[0] * x[0], w[0] * x[1], w[1] * x[0], w[1] * x[1]])


def __spinor_u(*, momentum: RealArray, mass: float, spin: int) -> ComplexArray:
    return __dirac_spinor(momentum=momentum, mass=mass, spin=spin, anti=1)


def __spinor_v(*, momentum: RealArray, mass: float, spin: int) -> ComplexArray:
    return __dirac_spinor(momentum=momentum, mass=mass, spin=spin, anti=-1)


def __spinor_ubar(*, momentum: RealArray, mass: float, spin: int) -> ComplexArray:
    x = np.conj(__dirac_spinor(momentum=momentum, mass=mass, spin=spin, anti=1))
    return np.array([x[2], x[3], x[0], x[1]])


def __spinor_vbar(*, momentum: RealArray, mass: float, spin: int) -> ComplexArray:
    x = np.conj(__dirac_spinor(momentum=momentum, mass=mass, spin=spin, anti=-1))
    return np.array([x[2], x[3], x[0], x[1]])


def spinor_u(momentum: RealArray, mass: float, spin: int) -> DiracWf:
    """
    Compute a u-spinor wavefunction.

    Parameters
    ----------
    momentum: ndarray
        Array containing the four-momentum of the particle.
        Must be 1 or 2 dimensional with leading dimension of size 4.
        If 2-dimensional, 2nd dimension must be the batch dimension.
    mass: float
        Mass of the particle.
    spin: int
        Spin of the particle. Must be 1 or -1.
    """
    __check_spin_dirac(spin=spin)
    wf = __spinor_u(momentum=momentum, mass=mass, spin=spin)
    return DiracWf(
        wavefunction=wf,
        momentum=momentum,
        direction=1,
    )


def spinor_v(momentum: RealArray, mass: float, spin: int) -> DiracWf:
    """
    Compute a v-spinor wavefunction.

    Parameters
    ----------
    momentum: ndarray
        Array containing the four-momentum of the particle.
        Must be 1 or 2 dimensional with leading dimension of size 4.
    mass: float
        Mass of the particle.
    spin: int
        Spin of the particle. Must be 1 or -1.
    """
    __check_spin_dirac(spin=spin)
    wf = __spinor_v(momentum=momentum, mass=mass, spin=spin)
    return DiracWf(
        wavefunction=wf,
        momentum=momentum,
        direction=1,
    )


def spinor_ubar(momentum: RealArray, mass: float, spin: int) -> DiracWf:
    """
    Compute a ubar-spinor wavefunction.

    Parameters
    ----------
    momentum: ndarray
        Array containing the four-momentum of the particle.
        Must be 1 or 2 dimensional with leading dimension of size 4.
    mass: float
        Mass of the particle.
    spin: int
        Spin of the particle. Must be 1 or -1.
    """
    __check_spin_dirac(spin=spin)
    wf = __spinor_ubar(momentum=momentum, mass=mass, spin=spin)
    return DiracWf(
        wavefunction=wf,
        momentum=-momentum,
        direction=-1,
    )


def spinor_vbar(momentum: RealArray, mass: float, spin: int) -> DiracWf:
    """
    Compute a vbar-spinor wavefunction.

    Parameters
    ----------
    momentum: ndarray
        Array containing the four-momentum of the particle.
        Must be 1 or 2 dimensional with leading dimension of size 4.
    mass: float
        Mass of the particle.
    spin: int
        Spin of the particle. Must be 1 or -1.
    """
    __check_spin_dirac(spin=spin)
    wf = __spinor_vbar(momentum=momentum, mass=mass, spin=spin)
    return DiracWf(
        wavefunction=wf,
        momentum=-momentum,
        direction=-1,
    )


def charge_conjugate(psi: DiracWf) -> DiracWf:
    """
    Charge conjugate the input wavefunction.

    Parameters
    ----------
    psi: DiracWf
        The dirac wavefunction.

    Returns
    -------
    psi_cc: DiracWf
        Charge conjugated wavefunction.
    """
    s = psi.direction
    wf = np.array(
        [
            s * psi.wavefunction[1],
            -s * psi.wavefunction[0],
            -s * psi.wavefunction[3],
            s * psi.wavefunction[2],
        ]
    )
    p = -psi.momentum
    return DiracWf(wavefunction=wf, momentum=p, direction=-s)
