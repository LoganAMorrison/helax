import dataclasses
import enum

import numpy as np

from helax.numpy.typing import ComplexArray, RealArray


class WeylType(enum.IntFlag):
    """Enumeration of the different types of Weyl spinors.

    Members
    -------
    X:
        Incoming left-handed (1/2, 0) fermion.
    Xd:
        Outgoing left-handed (1/2, 0) fermion.
    Y:
        Outgoing right-handed (0, 1/2) fermion.
    Yd:
        Incoming right-handed (0, 1/2) fermion.
    """

    X = enum.auto()
    Xd = enum.auto()
    Y = enum.auto()
    Yd = enum.auto()


@dataclasses.dataclass
class WeylWf:
    """Dataclass for Weyl wavefunctions (2-component spinors).

    Attributes
    ----------
    wavefunction: complex array
        2-Component Dirac spinor.
    momentum: array
        Four-momentum of the spinor. The momentum always points in the
        direction of the fermion flow.
    type: WeylType
        Type of weyl spinor (x, y, x^+, or y^+).
    """

    wavefunction: ComplexArray
    momentum: RealArray
    type: WeylType


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


def __weyl_spinor(
    *, momentum: RealArray, mass: float, spin: int, type: WeylType
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

    if type == WeylType.X:
        x = __chi(momentum=momentum, spin=spin)
        w = wp if spin == -1 else wm

    elif type == WeylType.Yd:
        x = __chi(momentum=momentum, spin=spin)
        w = wm if spin == -1 else wp
        x = np.array([-x[1], x[0]])

    elif type == WeylType.Y:
        x = __chi(momentum=momentum, spin=-spin)
        w = -wp if spin == 1 else wm

    elif type == WeylType.Xd:
        x = __chi(momentum=momentum, spin=-spin)
        x = np.array([-x[1], x[0]])
        w = wm if spin == 1 else -wp

    else:
        raise ValueError(f"Invalid weyl spinor type {type}")

    return w * x


def weyl_x(momentum: RealArray, mass: float, spin: int) -> WeylWf:
    """
    Compute a x-spinor wavefunction.

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
    wf = __weyl_spinor(momentum=momentum, mass=mass, spin=spin, type=WeylType.X)
    return WeylWf(
        wavefunction=wf,
        momentum=momentum,
        type=WeylType.X,
    )


def weyl_y(momentum: RealArray, mass: float, spin: int) -> WeylWf:
    """
    Compute a y-spinor wavefunction.

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
    wf = __weyl_spinor(momentum=momentum, mass=mass, spin=spin, type=WeylType.Y)
    return WeylWf(
        wavefunction=wf,
        momentum=-momentum,
        type=WeylType.Y,
    )


def weyl_xd(momentum: RealArray, mass: float, spin: int) -> WeylWf:
    """
    Compute a x^+-spinor wavefunction.

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
    wf = __weyl_spinor(momentum=momentum, mass=mass, spin=spin, type=WeylType.Xd)
    return WeylWf(
        wavefunction=wf,
        momentum=momentum,
        type=WeylType.Xd,
    )


def weyl_yd(momentum: RealArray, mass: float, spin: int) -> WeylWf:
    """
    Compute a y^+-spinor wavefunction.

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
    wf = __weyl_spinor(momentum=momentum, mass=mass, spin=spin, type=WeylType.Yd)
    return WeylWf(
        wavefunction=wf,
        momentum=-momentum,
        type=WeylType.Yd,
    )


def weyl_raise_index(chi: WeylWf) -> WeylWf:
    """Raise the index of the weyl spinor."""

    # z^a = eps^ab w_b
    #   z^1 = eps^12 w_2 = +w_2
    #   z^2 = eps^21 w_1 = -w_1
    wf = np.array(
        [
            chi.wavefunction[1],
            -chi.wavefunction[0],
        ]
    )
    return WeylWf(wavefunction=wf, momentum=chi.momentum, type=chi.type)


def weyl_lower_index(chi: WeylWf) -> WeylWf:
    """Lower the index of the weyl spinor."""

    # z_a = eps_ab w^b
    #   z_1 = eps_12 w^2 = -w_2
    #   z_2 = eps_21 w^1 = +w_1
    wf = np.array(
        [
            -chi.wavefunction[1],
            chi.wavefunction[0],
        ]
    )
    return WeylWf(wavefunction=wf, momentum=chi.momentum, type=chi.type)
