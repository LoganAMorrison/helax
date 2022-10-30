"""Module for computing currents and amplitudes for two fermions and a vector.
"""

# pylint: disable=invalid-name


# Notes on signs

# All momentum are considered incoming, except for fermions. For fermions, the
# momentum always points along the direction of the fermion flow.

# [v,s](p) + [v,s](q) -> [v,s](k)
# -----------------------
# k = p + q

#     p ->     k=p+q ->
#     -------x------
#           /
#     -----
#     q ->


# f(p) + [v,s](q) -> f(k)
# -----------------------
# k = p + q

#     p ->     ->  k=p+q
#     --->---x--->---
#           /
#     -----
#     q ->


# fbar(p) + [v,s](q) -> fbar(k)
# -----------------------------
# k = p - q

#     p <-     <- k=p-q
#     ---<---x---<---
#           /
#     -----
#     q ->


# f(p) + fbar(q) -> [v,s](k)
# --------------------------
# k = p - q

#     p ->     -> k=p-q
#     --->---x------
#           /
#     --<--
#     q <-


# w(p) + w(q) -> s(k)
# -------------------
# k = p + q

#     p ->     -> k=p+q
#     --->---x------
#           /
#     -->--
#     q ->


# w^+(p) + w^+(q) -> s(k)
# ---------------------------
# k = -p - q

#     p <-     -> k=-p-q
#     ---<---x------
#           /
#     --<--
#     q <-

from typing import Union

import numpy as np

from helax.numpy.typing import ComplexArray
from helax.numpy.wavefunctions import DiracWf, VectorWf
from helax.vertices import VertexFFV

from .propagators import attach_dirac, attach_vector

IM = 1.0j


def fermion_line_vector(
    psi_out: DiracWf,
    psi_in: DiracWf,
    left: Union[float, complex],
    right: Union[float, complex],
):
    """Compute a fermion line with a single gamma-matrix.

    Notes
    -----
    This computes the following inner-product:
        psi_out.ga[mu].(left * PL + right * PR).psi_in

    Parameters
    ----------
    psi_out: DiracWf
        Flow-out Dirac fermion.
    psi_in: DiracWf
        Flow-in Dirac fermion.
    left: float or complex
        Left-handed coupling.
    right: float or complex
        Right-handed coupling.

    Returns
    -------
    line: complex array
        The resulting Lorentz vector of the inner-product.
    """
    wf_in = psi_in.wavefunction
    wf_out = psi_out.wavefunction

    return np.array(
        [
            right * (wf_in[2] * wf_out[0] + wf_in[3] * wf_out[1])
            + left * (wf_in[0] * wf_out[2] + wf_in[1] * wf_out[3]),
            right * (wf_in[3] * wf_out[0] + wf_in[2] * wf_out[1])
            + left * (-wf_in[1] * wf_out[2] - wf_in[0] * wf_out[3]),
            IM * right * (-wf_in[3] * wf_out[0] + wf_in[2] * wf_out[1])
            + IM * left * (wf_in[1] * wf_out[2] - wf_in[0] * wf_out[3]),
            right * (wf_in[2] * wf_out[0] - wf_in[3] * wf_out[1])
            + left * (-wf_in[0] * wf_out[2] + wf_in[1] * wf_out[3]),
        ]
    )


def lorentz_slash(
    psi: DiracWf,
    lvec: ComplexArray,
    left: Union[float, complex],
    right: Union[float, complex],
):
    """Compute a Lorentz vector contracted with gamma-matrix."""

    f = psi.wavefunction
    flow = psi.direction

    g1 = left if flow == 1 else right
    g2 = right if flow == 1 else left

    k1 = lvec[0] + flow * lvec[3]
    k2 = lvec[0] - flow * lvec[3]
    k3 = lvec[1] + flow * IM * lvec[2]
    k4 = lvec[1] - flow * IM * lvec[2]

    return np.array(
        [
            g1 * (k1 * f[2] - k3 * f[3]),
            g1 * (-k4 * f[2] + k2 * f[3]),
            g2 * (k2 * f[0] + k3 * f[1]),
            g2 * (k4 * f[0] + k1 * f[1]),
        ]
    )


# =============================================================================
# ---- Currents ---------------------------------------------------------------
# =============================================================================


def current_fv_to_f(
    vertex: VertexFFV, mass: float, width: float, psi: DiracWf, polvec: VectorWf
) -> DiracWf:
    """Fuse a fermion and vector into and off-shell fermion.

    The off-shell fermion will have the same fermion flow as `psi`.

    Parameters
    ----------
    vertex : VertexFFV
        Feynman rule for the F-F-V vertex.
    mass : float
        Mass of the produced fermion.
    width : float
        Width of the produced fermion.
    psi : DiracWf
        Fermion to fuse with vector.
    polvec : VectorWf
        Vector to fuse with fermion.

    Returns
    -------
    chi: DiracWf
        Off-shell generated fermion.
    """
    flow = psi.direction
    wavefunction = lorentz_slash(psi, polvec.wavefunction, vertex.left, vertex.right)

    psi = DiracWf(
        wavefunction=wavefunction,
        momentum=psi.momentum + flow * polvec.momentum,
        direction=flow,
    )
    return attach_dirac(psi, mass, width)


def current_ff_to_v(
    vertex: VertexFFV, mass: float, width: float, psi_out: DiracWf, psi_in: DiracWf
) -> VectorWf:
    """Fuse two fermions into a vector boson.

    Parameters
    ----------
    vertex : VertexFFV
        Feynman rule for the F-F-V vertex.
    mass : float
        Mass of the resulting vector.
    width : float
        Width of the resulting vector.
    psi_out : DiracWf
        Flow-out fermion.
    psi_in : DiracWf
        Flow-in fermion.

    Returns
    -------
    eps: VectorWf
        Resulting vector boson wavefuction.
    """
    momentum = psi_in.momentum - psi_out.momentum
    wavefunction = fermion_line_vector(psi_out, psi_in, vertex.left, vertex.right)

    return attach_vector(
        VectorWf(wavefunction=wavefunction, momentum=momentum, direction=1), mass, width
    )


# =============================================================================
# ---- Amplitudes -------------------------------------------------------------
# =============================================================================


def amplitude_ffv(
    vertex: VertexFFV, psi_out: DiracWf, psi_in: DiracWf, polvec: VectorWf
):
    """Compute the scattering amplitude for a Fermion-Fermion-Vector vertex.

    Parameters
    ----------
    vertex : VertexFFV
        Feynman rule for the F-F-V vertex.
    psi_out : DiracWf
        Flow-out fermion wavefunction.
    psi_in : DiracWf
        Flow-in fermion wavefunction.
    polvec : VectorWf
        Vector wavefunction.

    Returns
    -------
    amp: complex
        Scattering amplitude.
    """
    assert psi_out.direction == -1, "`psi_out` must be have flow out."
    assert psi_in.direction == 1, "`psi_in` must be have flow in."

    wf_in = psi_in.wavefunction
    wf_out = psi_out.wavefunction
    eps = polvec.wavefunction
    vl = vertex.left
    vr = vertex.right

    k1 = eps[0] + eps[3]
    k2 = eps[0] - eps[3]
    k3 = eps[1] + IM * eps[2]
    k4 = eps[1] - IM * eps[2]

    return vr * (
        wf_out[0] * (k2 * wf_in[2] - k4 * wf_in[3])
        + (-k3 * wf_in[2] + k1 * wf_in[3]) * wf_out[1]
    ) + vl * (
        wf_out[2] * (k1 * wf_in[0] + k4 * wf_in[1])
        + (k3 * wf_in[0] + k2 * wf_in[1]) * wf_out[3]
    )
