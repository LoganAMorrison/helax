import numpy as np

from .typing import NumericArray


def kallen_lambda(a, b, c):
    return a**2 + b**2 + c**2 - 2 * a * b - 2 * a * c - 2 * b * c


def abs2(arr: NumericArray):
    return np.real(arr) ** 2 + np.imag(arr) ** 2
