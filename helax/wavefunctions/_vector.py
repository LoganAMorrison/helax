import functools
from typing import Union

import chex
import jax
import jax.numpy as jnp
from flax import struct

NdArray = Union[chex.Array, chex.ArrayNumpy]


class VectorWf(struct.PyTreeNode):
    wavefunction: chex.Array
    momentum: chex.Array
    direction: int


def _check_spin(s: int) -> None:
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
    assert s == 1 or s == -1, "`s` value must be 1 or -1."
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
    assert s == 1 or s == -1, "`s` value must be 1 (incoming) or -1 (outgoing)."
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


def _dispatch(unvec_fn, vec_fn, momentum: NdArray, mass: float, spin: int, s: int):
    assert momentum.shape[0] == 4, "First dimension of `momentum` must have size 4."

    if len(momentum.shape) == 1:
        return unvec_fn(momentum, mass, spin, s)
    elif len(momentum.shape) > 1:
        return vec_fn(momentum, mass, spin, s)
    else:
        raise ValueError(
            "Invalid shape for `momentum`. `momentum` must be 1 or 2 dimensional."
        )


def vector_wf(momentum: NdArray, mass: float, spin: int, out: bool) -> VectorWf:
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
    _check_spin(spin)
    s = jax.lax.cond(out, -1, 1)
    p = jnp.array(momentum)
    wf = _dispatch(_vector_wf, _vector_wf_vec, p, mass, spin, s)
    return VectorWf(wavefunction=wf, momentum=s * p, direction=s)
