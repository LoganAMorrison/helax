from typing import NamedTuple, Union

RealOrComplex = Union[float, complex]


# =============================================================================
# ---- Renormalizable Vertices ------------------------------------------------
# =============================================================================


class VertexFFS(NamedTuple):
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


class VertexFFV(NamedTuple):
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


class VertexWWS(NamedTuple):
    """
    Vertex representing weyl-weyl-scalar vertex corresponding to the
    interaction: g * S * chi * chi

    Parameters
    ----------
    g: complex
        Coupling coefficient.
    """

    g: RealOrComplex


class VertexWWV(NamedTuple):
    """
    Vertex representing weyl-weyl-vector vertex corresponding to the
    interaction: g * V[mu] * chi^+ * SigmaBar[mu] * eta

    Parameters
    ----------
    g: float
        Coupling coefficient.
    """

    g: RealOrComplex


class VertexVVVV(NamedTuple):
    """
    Vertex representing four-vector vertex corresponding to the
    interaction: V1[mu1] V2[mu2] V3[mu3] V4[mu4]

    Parameters
    ----------
    g1,g2,g3: float or complex
        Coupling coefficients of:
            g1: g[mu1,mu2] * g[mu3, mu4]
            g2: g[mu1,mu3] * g[mu2, mu4]
            g3: g[mu1,mu4] * g[mu2, mu3]
    """

    g1: RealOrComplex
    g2: RealOrComplex
    g3: RealOrComplex


class VertexVVV(NamedTuple):
    """
    Vertex representing three-vector vertex.

    The interaction term is:
          g[mu,nu] V1[mu] V2[nu] del[V3[rho], rho]
        + g[mu,nu] V2[mu] V3[nu] del[V1[rho], rho]
        + g[mu,nu] V3[mu] V1[nu] del[V2[rho], rho]
    with the vertex rule:
        g * (
              g[mu,nu] * (k2[rho] - k1[rho])
            + g[nu,rho] * (k3[mu] - k2[mu])
            + g[rho,mu] * (k1[nu] - k3[nu])
        )

    Parameters
    ----------
    g: float or complex
        Coupling coefficient
    """

    g: RealOrComplex


class VertexSSSS(NamedTuple):
    """
    Vertex representing a four-scalar vertex.
    """

    g: RealOrComplex


class VertexSSS(NamedTuple):
    """
    Vertex representing a four-scalar vertex.
    """

    g: RealOrComplex


class VertexSSVV(NamedTuple):
    """
    Vertex representing a scalar-scalar-vector-vector vertex.
    """

    g: RealOrComplex


class VertexSSV(NamedTuple):
    """
    Vertex representing a scalar-scalar-vector vertex.
    """

    g: RealOrComplex


class VertexSVV(NamedTuple):
    """
    Vertex representing a scalar-vector-vector vertex.
    """

    g: RealOrComplex


class VertexUUV(NamedTuple):
    """
    Vertex representing a ghost-ghost-vector vertex.
    """

    g1: RealOrComplex
    g2: RealOrComplex


class VertexUUS(NamedTuple):
    """
    Vertex representing a ghost-ghost-scalar vertex.
    """

    g: RealOrComplex


# =============================================================================
# ---- Effective Vertices -----------------------------------------------------
# =============================================================================


class VertexFFSDeriv(NamedTuple):
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
