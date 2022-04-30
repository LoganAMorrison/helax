from typing import Union
from flax import struct

RealOrComplex = Union[float, complex]


class VertexFFS(struct.PyTreeNode):
    """
    Vertex representing fermion-fermion-scalar vertex corresponding to the
    interaction: S * fbar * (gL * PL + gR * PR) * f

    Parameters
    ----------
    left: float
        Coupling coefficient of PL
    right: float
        Coupling coefficient of PR
    """

    left: RealOrComplex
    right: RealOrComplex


class VertexFFSDeriv(struct.PyTreeNode):
    """
    Vertex representing fermion-fermion-scalar vertex corresponding to the
    interaction: d[S,mu] * fbar * gamma[mu] * (gL * PL + gR * PR) * f

    Parameters
    ----------
    left: float
        Coupling coefficient of gamma[mu] * PL
    right: float
        Coupling coefficient of gamma[mu] * PR
    """

    left: RealOrComplex
    right: RealOrComplex


class VertexFFV(struct.PyTreeNode):
    """
    Vertex representing fermion-fermion-vector vertex corresponding to the
    interaction: V[mu] * fbar * gamma[mu] * (gL * PL + gR * PR) * f

    Parameters
    ----------
    left: float
        Coupling coefficient of gamma[mu] * PL
    right: float
        Coupling coefficient of gamma[mu] * PR
    """

    left: RealOrComplex
    right: RealOrComplex
