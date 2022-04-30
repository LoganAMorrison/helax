import functools
from typing import Union

import chex
import jax
import jax.numpy as jnp
from flax import struct

from ._utils import dispatch

NdArray = Union[chex.Array, chex.ArrayNumpy]


class ScalarWf(struct.PyTreeNode):
    wavefunction: chex.Array
    momentum: chex.Array
    direction: int


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


def scalar_wf(momentum: NdArray, out: bool) -> ScalarWf:
    """
    Compute a vector wavefunction.

    Parameters
    ----------
    momentum: ndarray
        Array containing the four-momentum of the particle.
        Must be 1 or 2 dimensional with leading dimension of size 4.
    """
    s = jax.lax.cond(out, -1, 1)
    p = jnp.array(momentum)
    wf = dispatch(_scalar_wf, _scalar_wf_vec, p)
    return ScalarWf(wavefunction=wf, momentum=s * p, direction=s)
