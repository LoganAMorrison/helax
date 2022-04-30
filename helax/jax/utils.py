import jax
import jax.numpy as jnp


def kallen_lambda(a, b, c):
    return a**2 + b**2 + c**2 - 2 * a * b - 2 * a * c - 2 * b * c


@jax.jit
def abs2(arr: jnp.ndarray):
    return jnp.real(arr) ** 2 + jnp.imag(arr) ** 2
