"""Module for computing currents and amplitudes for two fermions and a vector.
"""


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


import numpy as np

from helax.numpy.wavefunctions import DiracWf, VectorWf
from helax.vertices import VertexFFV

from .propagators import attach_dirac, attach_vector

IM = 1.0j

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
    eps = polvec.wavefunction
    f = psi.wavefunction
    vl = vertex.left
    vr = vertex.right

    momentum = psi.momentum * psi.direction * polvec.momentum

    if psi.direction == -1:
        wavefunction = np.array(
            [
                vl * (f[3] * (eps[1] + IM * eps[2]) + f[2] * (eps[0] + eps[3])),
                vl * (f[2] * (eps[1] - IM * eps[2]) + f[3] * (eps[0] - eps[3])),
                vr * (-f[1] * (eps[1] + IM * eps[2]) + f[0] * (eps[0] - eps[3])),
                vr * (-f[0] * (eps[1] - IM * eps[2]) + f[1] * (eps[0] + eps[3])),
            ]
        )
    else:
        wavefunction = np.array(
            [
                vr * (-f[3] * (eps[1] - IM * eps[2]) + f[2] * (eps[0] - eps[3])),
                vr * (-f[2] * (eps[1] + IM * eps[2]) + f[3] * (eps[0] + eps[3])),
                vl * (f[1] * (eps[1] - IM * eps[2]) + f[0] * (eps[0] + eps[3])),
                vl * (f[0] * (eps[1] + IM * eps[2]) + f[1] * (eps[0] - eps[3])),
            ]
        )

    psi = DiracWf(wavefunction=wavefunction, momentum=momentum, direction=psi.direction)
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
    wf_in = psi_in.wavefunction
    wf_out = psi_out.wavefunction
    vl = vertex.left
    vr = vertex.right

    momentum = psi_in.momentum - psi_out.momentum

    wavefunction = np.array(
        [
            vr * (wf_in[2] * wf_out[0] + wf_in[3] * wf_out[1])
            + vl * (wf_in[0] * wf_out[2] + wf_in[1] * wf_out[3]),
            vr * (wf_in[3] * wf_out[0] + wf_in[2] * wf_out[1])
            - vl * (wf_in[1] * wf_out[2] + wf_in[0] * wf_out[3]),
            IM * vr * (wf_in[2] * wf_out[1] - wf_in[3] * wf_out[0])
            + IM * vl * (wf_in[1] * wf_out[2] - wf_in[0] * wf_out[3]),
            vr * (wf_in[2] * wf_out[0] - wf_in[3] * wf_out[1])
            + vl * (wf_in[1] * wf_out[3] - wf_in[0] * wf_out[2]),
        ]
    )

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
    im = 1.0j

    eps0p3 = eps[0] + eps[3]
    eps0m3 = eps[0] - eps[3]

    eps1p2 = eps[1] + im * eps[2]
    eps1m2 = eps[1] - im * eps[2]

    return vr * (
        wf_in[2] * (-wf_out[1] * eps1p2 + wf_out[0] * eps0m3)
        + wf_in[3] * (-wf_out[0] * eps1m2 + wf_out[1] * eps0p3)
    ) + vl * (
        wf_in[1] * (wf_out[2] * eps1m2 + wf_out[3] * eps0m3)
        + wf_in[0] * (wf_out[3] * eps1p2 + wf_out[2] * eps0p3)
    )
