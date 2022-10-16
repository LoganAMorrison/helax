"""Module implementing Dirac matrices."""

import numpy as np

Dirac1 = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.complex128
)
DiracG0 = np.array(
    [[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.complex128
)
DiracG1 = np.array(
    [[0, 0, 0, 1], [0, 0, 1, 0], [0, -1, 0, 0], [-1, 0, 0, 0]], dtype=np.complex128
)

DiracG2 = np.array(
    [[0, 0, 0, -1j], [0, 0, 1j, 0], [0, 1j, 0, 0], [-1j, 0, 0, 0]], dtype=np.complex128
)
DiracG3 = np.array(
    [[0, 0, 1, 0], [0, 0, 0, -1], [-1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.complex128
)
DiracG5 = np.array(
    [[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.complex128
)

ProjL = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.complex128
)
ProjR = np.array(
    [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.complex128
)
ChargeConj = np.array(
    [[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, -1], [0, 0, 1, 0]], dtype=np.complex128
)

# Shape: (4, 4, 4)
DiracG = np.array([DiracG0, DiracG1, DiracG2, DiracG3])
# NP broadcasting is from right to left, so this works
DiracGL = DiracG @ ProjL
DiracGR = DiracG @ ProjR


WeylS0 = np.array([[1, 0], [0, 1]], dtype=np.complex128)
WeylS1 = np.array([[0, 1], [1, 0]], dtype=np.complex128)
WeylS2 = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
WeylS3 = np.array([[1, 0], [0, -1]], dtype=np.complex128)

WeylSBar0 = np.array([[1, 0], [0, 1]], dtype=np.complex128)
WeylSBar1 = np.array([[0, -1], [-1, 0]], dtype=np.complex128)
WeylSBar2 = np.array([[0, 1j], [-1j, 0]], dtype=np.complex128)
WeylSBar3 = np.array([[-1, 0], [0, 1]], dtype=np.complex128)
