import dataclasses

import numpy as np

from helax.numpy.typing import ComplexArray, RealArray

from .utils import check_momentum_shape


@dataclasses.dataclass
class VectorWf:
    wavefunction: ComplexArray
    momentum: RealArray
    direction: int


# =============================================================================
# ---- Vector Wavefunction ----------------------------------------------------
# =============================================================================


def __check_spin_vector(*, spin: int):
    assert spin == 1 or spin == -1 or spin == 0, "Spin must be 1, 0, or -1."


def __polvec_transverse(*, k: RealArray, spin: int, sgn: int) -> RealArray:
    """
    Compute a transverse (spin 1 or -1) vector wavefunction.

    Parameters
    ----------
    k: ndarray
        Array containing the four-momentum of the particle.
        Must be 1 or 2 dimensional with leading dimension of size 4.
    spin: int
        Spin of the particle. Must be -1, 0, or -1.
    s: int
        If 1, the returned wavefunction is outgoing. If -1, the
        returned wavefunction is incoming. Must be 1 or -1.
    """
    assert sgn == 1 or sgn == -1, "`s` value must be 1 or -1."
    check_momentum_shape(k, require_batch=True)
    kx, ky, kz = k[1:]
    kt = np.hypot(kx, ky)

    eps = np.finfo(k.dtype.name).eps
    mask = kt < eps
    polvec = np.zeros((4, kx.shape[-1]), dtype=kx.dtype) * 1j

    if np.any(mask):
        polvec[0, mask] = 0.0j
        polvec[1, mask] = -spin / np.sqrt(2) + 0.0j
        polvec[2, mask] = -np.copysign(1.0, kz[mask]) * 1.0j / np.sqrt(2)
        polvec[3, mask] = 0.0j

    mask = ~mask
    if np.any(mask):
        kx = kx[mask]
        ky = ky[mask]
        kz = kz[mask]
        km = np.sqrt(np.square(kx) + np.square(ky) + np.square(kz))

        kxt = kx / kt[mask] / np.sqrt(2)
        kyt = ky / kt[mask] / np.sqrt(2)
        kzm = kz / km
        ktm = kt[mask] / km / np.sqrt(2)

        polvec[0, mask] = 0.0 + 0.0 * 1j
        polvec[1, mask] = -spin * kxt * kzm + +sgn * kyt * 1j
        polvec[2, mask] = -spin * kyt * kzm + -sgn * kxt * 1j
        polvec[3, mask] = +spin * ktm + 0.0 * 1j

    return polvec


def __polvec_longitudinal(*, k: RealArray, mass: float) -> RealArray:
    """
    Compute a longitudinal (spin 0) vector wavefunction.

    Parameters
    ----------
    k: ndarray
        Array containing the four-momentum of the particle.
        Must be 1 or 2 dimensional with leading dimension of size 4.
    mass: float
        Mass of the particle.
    """
    check_momentum_shape(k, require_batch=True)
    polvec = np.zeros((4, k.shape[-1]), dtype=k.dtype) * 1j
    if mass == 0.0:
        return polvec

    e, kx, ky, kz = k
    km = np.linalg.norm(k[1:], axis=0)
    eps = np.finfo(k.dtype.name).eps
    mask = km < eps

    if np.any(mask):
        polvec[3, mask] = 1.0 + 0.0j

    mask = ~mask
    if np.any(mask):
        n = e[mask] / (mass * km[mask])
        polvec[0, mask] = km[mask] / mass + 0.0j
        polvec[1, mask] = n * kx[mask] + 0.0j
        polvec[2, mask] = n * ky[mask] + 0.0j
        polvec[3, mask] = n * kz[mask] + 0.0j

    return polvec


def __vector_wf(*, momentum: RealArray, mass: float, spin: int, sgn: int):
    """
    Compute a vector wavefunction.

    Parameters
    ----------
    momentum: ndarray
        Array containing the four-momentum of the particle.
        Must be 1 or 2 dimensional with leading dimension of size 4.
    mass: float
        Mass of the particle.
    spin: int
        Spin of the particle. Must be -1, 0, or -1.
    s: int
        If 1, the returned wavefunction is outgoing. If -1, the
        returned wavefunction is incoming. Must be 1 or -1.
    """
    assert sgn == 1 or sgn == -1, "`s` value must be 1 (incoming) or -1 (outgoing)."
    check_momentum_shape(momentum, require_batch=True)

    if spin == -1 or spin == 1:
        return __polvec_transverse(k=momentum, spin=spin, sgn=sgn)
    return __polvec_longitudinal(k=momentum, mass=mass)


def vector_wf(momentum: RealArray, mass: float, spin: int, out: bool) -> VectorWf:
    """
    Compute a vector wavefunction.

    Parameters
    ----------
    momentum: ndarray
        Array containing the four-momentum of the particle.
        Must be 1 or 2 dimensional with leading dimension of size 4.
    mass: float
        Mass of the particle.
    spin: int
        Spin of the particle. Must be -1, 0, or -1.
    out: bool
        If true, the returned wavefunction is outgoing.
    """
    __check_spin_vector(spin=spin)
    check_momentum_shape(momentum, require_batch=False)
    sgn = -1 if out else 1

    if len(momentum.shape) == 2:
        k = momentum
        squeeze = False
    else:
        k = np.expand_dims(momentum, -1)
        squeeze = True

    wf = __vector_wf(momentum=k, mass=mass, spin=spin, sgn=sgn)

    if squeeze:
        wf = np.squeeze(wf)

    return VectorWf(wavefunction=wf, momentum=sgn * momentum, direction=sgn)
