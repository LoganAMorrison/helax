from typing import Union

import numpy as np
import numpy.typing as npt

Numeric = Union[np.uint, np.float_, np.complex_, np.int_, np.uint]
NdArray = npt.NDArray[Numeric]


def ldot(lv1: NdArray, lv2: NdArray):
    return lv1[0] * lv2[0] - lv1[1] * lv2[1] - lv1[2] * lv2[2] - lv1[3] * lv2[3]


def lnorm_sqr(lv: NdArray):
    return ldot(lv, lv)


def lnorm(lv: NdArray):
    return np.sqrt(lnorm_sqr(lv))


def lnorm3_sqr(lv: NdArray):
    return np.square(lv[1:])


def lnorm3(lv: NdArray):
    return np.square(lnorm3_sqr(lv))
