import dataclasses

import numpy as np

from .typing import ComplexArray, RealArray


@dataclasses.dataclass
class DiracWf:
    wavefunction: ComplexArray
    momentum: RealArray
    direction: int


@dataclasses.dataclass
class VectorWf:
    wavefunction: ComplexArray
    momentum: RealArray
    direction: int


@dataclasses.dataclass
class ScalarWf:
    wavefunction: ComplexArray
    momentum: RealArray
    direction: int


def __check_momentum_shape(momentum: RealArray, require_batch: bool = True):
    shape = momentum.shape
    assert shape[0] == 4, "Leading dimension must have length 4."
    if require_batch:
        assert len(shape) == 2, "Expected a batch dimension. Found 1-D array."


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
    __check_momentum_shape(k, require_batch=True)
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

        kxt = kx / kt / np.sqrt(2)
        kyt = ky / kt / np.sqrt(2)
        kzm = kz / km
        ktm = kt / km / np.sqrt(2)

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
    __check_momentum_shape(k, require_batch=True)
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
        n = e / (mass * km[mask])
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
    __check_momentum_shape(momentum, require_batch=True)

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
    __check_momentum_shape(momentum, require_batch=False)
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


# =============================================================================
# ---- Scalar Wavefunction ----------------------------------------------------
# =============================================================================


def scalar_wf(momentum: RealArray, out: bool) -> ScalarWf:
    """
    Compute a vector wavefunction.

    Parameters
    ----------
    momentum: ndarray
        Array containing the four-momentum of the particle.
        Must be 1 or 2 dimensional with leading dimension of size 4.
    """
    s = -1 if out else 1
    if len(momentum.shape) > 1:
        shape = (momentum.shape[-1],)
    else:
        shape = (1,)

    wf = np.ones(shape, dtype=momentum.dtype) + 0.0j
    return ScalarWf(wavefunction=wf, momentum=s * momentum, direction=s)
