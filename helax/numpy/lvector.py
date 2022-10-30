"""Utilities for working with Lorentz vectors."""


from typing import Union

import numpy as np
import numpy.typing as npt

Numeric = Union[np.uint, np.float_, np.complex_, np.int_, np.uint]
NdArray = npt.NDArray[Numeric]


def ldot(lv1: NdArray, lv2: NdArray):
    """Compute the Lorenzian inner-product between two Lorentz vectors."""
    return lv1[0] * lv2[0] - lv1[1] * lv2[1] - lv1[2] * lv2[2] - lv1[3] * lv2[3]


def lnorm_sqr(lvector: NdArray):
    """Compute the squared Lorenzian norm of a Lorentz vector."""
    return ldot(lvector, lvector)


def lnorm(lvector: NdArray):
    """Compute the Lorenzian norm of a Lorentz vector."""
    return np.sqrt(lnorm_sqr(lvector))


def lnorm3_sqr(lvector: NdArray):
    """Compute the norm of the space-like components of the Lorentz vector."""
    return np.square(lvector[1:])


def lnorm3(lvector: NdArray):
    """Compute the norm of the space-like components of the Lorentz vector."""
    return np.sqrt(lnorm3_sqr(lvector))
