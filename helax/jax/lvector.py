from typing import Union

import chex
import jax.numpy as jnp

NdArray = Union[chex.Array, chex.ArrayNumpy]


def ldot(lv1: NdArray, lv2: NdArray):
    return lv1[0] * lv2[0] - lv1[1] * lv2[1] - lv1[2] * lv2[2] - lv1[3] * lv2[3]


def lnorm_sqr(lv: NdArray):
    return ldot(lv, lv)


def lnorm(lv: NdArray):
    return jnp.sqrt(lnorm_sqr(lv))


def lnorm3_sqr(lv: NdArray):
    return jnp.square(lv[1:])


def lnorm3(lv: NdArray):
    return jnp.square(lnorm3_sqr(lv))
