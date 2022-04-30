import functools
from typing import NamedTuple, Union

import chex
import jax
import jax.numpy as jnp


class DiracWf(NamedTuple):
    wavefunction: chex.Array
    momentum: chex.Array
    direction: int


class VectorWf(NamedTuple):
    wavefunction: chex.Array
    momentum: chex.Array
    direction: int


class ScalarWf(NamedTuple):
    wavefunction: chex.Array
    momentum: chex.Array
    direction: int


# =============================================================================
# ---- Dirac Wavefunction -----------------------------------------------------
# =============================================================================


def _check_spin_dirac(s: int):
    assert s == 1 or s == -1, "Spin must be 1 or -1."


def _chi(p: chex.Array, s: int):
    """
    Compute the two-component weyl spinor.

    Parameters
    ----------
    p: array
        Array containing the 4-momentum of the wavefunction.
    s: int
        Spin of the wavefunction. Must be 1 or -1.
    """
    eps = jnp.finfo(p.dtype).eps

    px = p[1]
    py = p[2]
    pz = p[3]

    pm = jnp.linalg.norm(p[1:], axis=0)

    def _chi_aligned(s: int, *_):
        # Compute the two-component weyl spinor in the case where pz = -pm.
        # return jnp.array([(s - 1.0) / 2.0 + 0j, (s + 1.0) / 2.0 + 0j])
        return jnp.array([0.0j, s + 0.0j])

    def _chi_unaligned(s, px, py, pz, pm):
        # Compute the two-component weyl spinor in the case where pz != -pm.
        den = jnp.sqrt(2 * pm * (pm + pz))

        return jnp.array(
            [
                (pm + pz) / den,
                (s * px + py * 1j) / den,
            ]
        )

    x = jax.lax.cond(pm + pz < eps, _chi_aligned, _chi_unaligned, s, px, py, pz, pm)

    return jax.lax.switch(
        s + 1,
        [
            lambda: jnp.array([x[1], x[0]]),
            lambda: jnp.array([x[0], x[1]]),
        ],
    )


def _dirac_spinor(p: chex.Array, mass: float, s: int, anti: int):
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
    pm = jnp.linalg.norm(p[1:], axis=0)

    wp = jnp.sqrt(p[0] + pm)
    wm = mass / wp

    w = jax.lax.cond(
        s == -anti,
        lambda: jnp.array([anti * wp, wm]),
        lambda: jnp.array([wm, anti * wp]),
    )

    x = _chi(p, s * anti)

    return jnp.array([w[0] * x[0], w[0] * x[1], w[1] * x[0], w[1] * x[1]])


@functools.partial(jax.vmap, in_axes=(1, None, None), out_axes=1)
def _spinor_u(p: chex.Array, mass: float, s: int):
    return _dirac_spinor(p, mass, s, 1)


@functools.partial(jax.vmap, in_axes=(1, None, None), out_axes=1)
def _spinor_v(p: chex.Array, mass: float, s: int):
    return _dirac_spinor(p, mass, s, -1)


@functools.partial(jax.vmap, in_axes=(1, None, None), out_axes=1)
def _spinor_ubar(p: chex.Array, mass: float, s: int):
    x = jnp.conj(_dirac_spinor(p, mass, s, 1))
    return jnp.array([x[2], x[3], x[0], x[1]])


@functools.partial(jax.vmap, in_axes=(1, None, None), out_axes=1)
def _spinor_vbar(p: chex.Array, mass: float, s: int):
    x = jnp.conj(_dirac_spinor(p, mass, s, -1))
    return jnp.array([x[2], x[3], x[0], x[1]])


def spinor_u(
    momentum: Union[chex.ArrayNumpy, chex.Array], mass: float, spin: int
) -> DiracWf:
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
    _check_spin_dirac(spin)
    p: chex.Array = jnp.array(momentum)
    wf = _spinor_u(p, mass, spin)
    return DiracWf(
        wavefunction=wf,
        momentum=p,
        direction=1,
    )


def spinor_v(
    momentum: Union[chex.ArrayNumpy, chex.Array], mass: float, spin: int
) -> DiracWf:
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
    _check_spin_dirac(spin)
    p: chex.Array = jnp.array(momentum)
    wf = _spinor_v(p, mass, spin)
    return DiracWf(
        wavefunction=wf,
        momentum=p,
        direction=1,
    )


def spinor_ubar(
    momentum: Union[chex.ArrayNumpy, chex.Array], mass: float, spin: int
) -> DiracWf:
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
    _check_spin_dirac(spin)
    p: chex.Array = jnp.array(momentum)
    wf = _spinor_ubar(p, mass, spin)
    return DiracWf(
        wavefunction=wf,
        momentum=-p,
        direction=-1,
    )


def spinor_vbar(
    momentum: Union[chex.ArrayNumpy, chex.Array], mass: float, spin: int
) -> DiracWf:
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
    _check_spin_dirac(spin)
    p: chex.Array = jnp.array(momentum)
    wf = _spinor_vbar(p, mass, spin)
    return DiracWf(
        wavefunction=wf,
        momentum=-p,
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
    wf = jnp.array(
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


def _check_spin_vector(s: int) -> None:
    assert s in [-1, 0, 1], "Spin must be -1, 0, or 1."


def _polvec_transverse(k: chex.Array, spin: int, s: int) -> chex.Array:
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
    # assert s == 1 or s == -1, "`s` value must be 1 or -1."
    kx, ky, kz = k[1:]
    kt = jnp.hypot(kx, ky)

    def aligned():
        return jnp.array(
            [
                0.0j,
                -spin / jnp.sqrt(2) + 0.0j,
                -jnp.copysign(1.0, kz) * 1.0j / jnp.sqrt(2),
                0.0j,
            ]
        )

    def unaligned():
        km = jnp.sqrt(jnp.square(kx) + jnp.square(ky) + jnp.square(kz))

        kxt = kx / kt / jnp.sqrt(2)
        kyt = ky / kt / jnp.sqrt(2)
        kzm = kz / km
        ktm = kt / km / jnp.sqrt(2)

        eps_0 = 0.0 + 0.0 * 1j
        eps_x = -spin * kxt * kzm + +s * kyt * 1j
        eps_y = -spin * kyt * kzm + -s * kxt * 1j
        eps_z = +spin * ktm + 0.0 * 1j

        return jnp.array([eps_0, eps_x, eps_y, eps_z])

    return jax.lax.cond(kt == 0.0, aligned, unaligned)


def _polvec_longitudinal(k: chex.Array, mass: float) -> chex.Array:
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
    e, kx, ky, kz = k
    km = jnp.linalg.norm(k[1:], axis=0)
    n = e / (mass * km)

    def rest():
        return jnp.array([0.0j, 0.0j, 0.0j, 1.0 + 0.0j])

    def boosted():
        return jnp.array(
            [km / mass + 0.0j, n * kx + 0.0j, n * ky + 0.0j, n * kz + 0.0j]
        )

    def massive():
        return jax.lax.cond(km == 0.0, rest, boosted)

    def massless():
        return jnp.array([0.0j, 0.0j, 0.0j, 0.0j])

    return jax.lax.cond(mass == 0.0, massless, massive)


def _vector_wf(k: chex.Array, mass: float, spin: int, s: int):
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
    # assert s == 1 or s == -1, "`s` value must be 1 (incoming) or -1 (outgoing)."
    return jax.lax.switch(
        spin + 1,
        [
            lambda: _polvec_transverse(k, spin, s),
            lambda: _polvec_longitudinal(k, mass),
            lambda: _polvec_transverse(k, spin, s),
        ],
    )


@functools.partial(jax.vmap, in_axes=(1, None, None, None), out_axes=1)
def _vector_wf_vec(k: chex.Array, mass: float, spin: int, s: int):
    return _vector_wf(k, mass, spin, s)


def vector_wf(
    momentum: Union[chex.Array, chex.ArrayNumpy], mass: float, spin: int, out: bool
) -> VectorWf:
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
    _check_spin_vector(spin)
    s = jax.lax.cond(out, lambda: -1, lambda: 1)
    p = jnp.array(momentum)
    wf = _vector_wf_vec(p, mass, spin, s)
    return VectorWf(wavefunction=wf, momentum=s * p, direction=s)


# =============================================================================
# ---- Scalar Wavefunction ----------------------------------------------------
# =============================================================================


def _scalar_wf(k: chex.Array):
    """
    Compute a vector wavefunction.

    Parameters
    ----------
    momentum: ndarray
        Array containing the four-momentum of the particle.
        Must be 1 or 2 dimensional with leading dimension of size 4.
    """
    return jnp.array([1.0], dtype=k.dtype) + 0.0j


@functools.partial(jax.vmap, in_axes=1, out_axes=1)
def _scalar_wf_vec(k: chex.Array):
    return _scalar_wf(k)


def scalar_wf(momentum: Union[chex.Array, chex.ArrayNumpy], out: bool) -> ScalarWf:
    """
    Compute a vector wavefunction.

    Parameters
    ----------
    momentum: ndarray
        Array containing the four-momentum of the particle.
        Must be 1 or 2 dimensional with leading dimension of size 4.
    """
    s = jax.lax.cond(out, lambda: -1, lambda: 1)
    p = jnp.array(momentum)
    wf = _scalar_wf_vec(p)
    return ScalarWf(wavefunction=wf, momentum=s * p, direction=s)
