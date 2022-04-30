from typing import Union
import jax.numpy as jnp
import chex

# import jax
# from flax import struct


# class LVector(struct.PyTreeNode):
#     x0: chex.Array
#     x1: chex.Array
#     x2: chex.Array
#     x3: chex.Array

#     def __getitem__(self, i: int) -> chex.Array:
#         return jax.lax.switch(
#             i, [lambda: self.x0, lambda: self.x1, lambda: self.x2, lambda: self.x3]
#         )

#     def norm_sqr(self):
#         return (
#             self.x0 * self.x0
#             - self.x1 * self.x1
#             - self.x2 * self.x2
#             - self.x3 * self.x3
#         )

#     def norm(self):
#         return jnp.sqrt(self.norm_sqr())

#     def norm3_sqr(self):
#         return self.x1 * self.x1 + self.x2 * self.x2 + self.x3 * self.x3

#     def norm3(self):
#         return jnp.sqrt(self.norm3_sqr())


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
