import chex
import jax
import jax.numpy as jnp
from flax import struct

from ._utils import NdArray, dispatch


class DiracWf(struct.PyTreeNode):
    wavefunction: chex.Array
    momentum: chex.Array
    direction: int


def check_spin(s: int):
    assert s == 1 or s == -1, "Spin must be 1 or -1."


def chi_aligned(
    px: chex.Array, py: chex.Array, pz: chex.Array, pm: chex.Array, s: int
) -> chex.Array:
    """
    Compute the two-component weyl spinor in the case where pz = -pm.

    Parameters
    ----------
    px, py, pz: array
        x-, y-, and z-components of momentum.
    pm: array
        Magnitude of 3-momentum.
    s: int
        Spin of the wavefunction. Must be 1 or -1.
    """
    return jnp.array([(s - 1.0) / 2.0 + 0j, (s + 1.0) / 2.0 + 0j])


def chi_unaligned(
    px: chex.Array, py: chex.Array, pz: chex.Array, pm: chex.Array, s: int
):
    """
    Compute the two-component weyl spinor in the case where pz != -pm.

    Parameters
    ----------
    px, py, pz: array
        x-, y-, and z-components of momentum.
    pm: array
        Magnitude of 3-momentum.
    s: int
        Spin of the wavefunction. Must be 1 or -1.
    """
    den = jnp.sqrt(2 * pm * (pm + pz))

    return jnp.array(
        [
            (pm + pz) / den,
            (s * px + py * 1j) / den,
        ]
    )


def chi(p: chex.Array, s: int):
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

    x = jax.lax.cond(pm + pz < eps, chi_aligned, chi_unaligned, px, py, pz, pm, s)

    return jax.lax.switch(
        s + 1,
        [
            lambda: jnp.array([x[1], x[0]]),
            lambda: jnp.array([x[0], x[1]]),
        ],
    )


def dirac_spinor(p: chex.Array, mass: float, s: int, anti: int):
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

    x = chi(p, s * anti)

    return jnp.array([w[0] * x[0], w[0] * x[1], w[1] * x[0], w[1] * x[1]])


def _spinor_u(p: chex.Array, mass: float, s: int):
    return dirac_spinor(p, mass, s, 1)


def _spinor_v(p: chex.Array, mass: float, s: int):
    return dirac_spinor(p, mass, s, -1)


def _spinor_ubar(p: chex.Array, mass: float, s: int):
    x = jnp.conj(dirac_spinor(p, mass, s, 1))
    return jnp.array([x[2], x[3], x[0], x[1]])


def _spinor_vbar(p: chex.Array, mass: float, s: int):
    x = jnp.conj(dirac_spinor(p, mass, s, -1))
    return jnp.array([x[2], x[3], x[0], x[1]])


_spinor_u_vec = jax.vmap(
    _spinor_u,
    in_axes=(1, None, None),  # type: ignore
    out_axes=1,
)
_spinor_v_vec = jax.vmap(
    _spinor_v,
    in_axes=(1, None, None),  # type: ignore
    out_axes=1,
)
_spinor_ubar_vec = jax.vmap(
    _spinor_ubar,
    in_axes=(1, None, None),  # type: ignore
    out_axes=1,
)
_spinor_vbar_vec = jax.vmap(
    _spinor_vbar,
    in_axes=(1, None, None),  # type: ignore
    out_axes=1,
)


def spinor_u(momentum: NdArray, mass: float, spin: int) -> DiracWf:
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
    check_spin(spin)
    p: chex.Array = jnp.array(momentum)
    wf = dispatch(_spinor_u, _spinor_u_vec, momentum, mass, spin)
    return DiracWf(
        wavefunction=wf,
        momentum=p,
        direction=1,
    )


def spinor_v(momentum: NdArray, mass: float, spin: int) -> DiracWf:
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
    check_spin(spin)
    p: chex.Array = jnp.array(momentum)
    wf = dispatch(_spinor_v, _spinor_v_vec, momentum, mass, spin)
    return DiracWf(
        wavefunction=wf,
        momentum=p,
        direction=1,
    )


def spinor_ubar(momentum: NdArray, mass: float, spin: int) -> DiracWf:
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
    check_spin(spin)
    p: chex.Array = jnp.array(momentum)
    wf = dispatch(_spinor_ubar, _spinor_ubar_vec, momentum, mass, spin)
    return DiracWf(
        wavefunction=wf,
        momentum=-p,
        direction=-1,
    )


def spinor_vbar(momentum: NdArray, mass: float, spin: int) -> DiracWf:
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
    check_spin(spin)
    p: chex.Array = jnp.array(momentum)
    wf = dispatch(_spinor_vbar, _spinor_vbar_vec, p, mass, spin)
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
