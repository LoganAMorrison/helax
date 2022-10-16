"""Module for computing currents and amplitudes for two fermions and a scalar.
"""


import numpy as np

# from .dirac import WeylS0, WeylS1, WeylS2, WeylS3
from helax.numpy.wavefunctions import DiracWf, ScalarWf
from helax.vertices import VertexFFS

from .propagators import attach_dirac, attach_scalar

IM = 1.0j


# =============================================================================
# ---- Currents ---------------------------------------------------------------
# =============================================================================


def current_fs_to_f(
    vertex: VertexFFS, mass: float, width: float, psi: DiracWf, phi: ScalarWf
) -> DiracWf:
    """Fuse a fermion and scalar into an off-shell fermion.

    Parameters
    ----------
    vertex : VertexFFS
        Feynman rule for the F-F-S vertex.
    mass : float
        Mass of the resulting fermion.
    width : float
        Width of the resulting fermion
    psi : DiracWf
        Fermion to fuse with scalar.
    phi : ScalarWf
        Scalar to fuse with fermion.

    Returns
    -------
    chi: DiracWf
        Resulting fermion.
    """
    phi_wf = phi.wavefunction
    fi = psi.wavefunction
    vl = vertex.left
    vr = vertex.right

    momentum = psi.momentum * psi.direction + phi.momentum

    wavefunction = np.array(
        [
            vl * phi_wf * fi[0],
            vl * phi_wf * fi[1],
            vr * phi_wf * fi[2],
            vr * phi_wf * fi[3],
        ]
    )

    psi = DiracWf(wavefunction=wavefunction, momentum=momentum, direction=psi.direction)
    return attach_dirac(psi, mass, width)


def current_ff_to_s(
    vertex: VertexFFS, mass: float, width: float, psi_out: DiracWf, psi_in: DiracWf
) -> ScalarWf:
    """Fuse two fermions into a scalar.

    Parameters
    ----------
    vertex : VertexFFS
        Feynman rule for the F-F-S vertex.
    mass : float
        Mass of the resulting scalar.
    width : float
        Width of the resulting scalar.
    psi_out : DiracWf
        Flow-out fermion.
    psi_in : DiracWf
        Flow-in fermion.

    Returns
    -------
    phi: ScalarWf
        Resulting scalar wavefuction.
    """
    wf_in = psi_in.wavefunction
    wf_out = psi_out.wavefunction
    vl = vertex.left
    vr = vertex.right

    momentum = psi_in.momentum - psi_out.momentum
    wavefunction = vl * (wf_in[0] * wf_out[0] + wf_in[1] * wf_out[1]) + vr * (
        wf_in[2] * wf_out[2] + wf_in[3] * wf_out[3]
    )

    phi = ScalarWf(wavefunction=wavefunction, momentum=momentum, direction=1)

    return attach_scalar(phi, mass, width)


# =============================================================================
# ---- Amplitudes -------------------------------------------------------------
# =============================================================================


def amplitude_ffs(vertex: VertexFFS, psi_out: DiracWf, psi_in: DiracWf, phi: ScalarWf):
    """Compute the scattering amplitude for a Fermion-Fermion-Scalar vertex.

    Parameters
    ----------
    vertex : VertexFFS
        Feynman rule for the F-F-S vertex.
    psi_out : DiracWf
        Flow-out fermion wavefunction.
    psi_in : DiracWf
        Flow-in fermion wavefunction.
    phi : ScalarWf
        Scalar wavefunction.

    Returns
    -------
    amp: complex
        Scattering amplitude.
    """
    assert psi_out.direction == -1, "`psi_out` must be have flow out."
    assert psi_in.direction == 1, "`psi_in` must be have flow in."

    wf_in = psi_in.wavefunction
    wf_out = psi_out.wavefunction
    vl = vertex.left
    vr = vertex.right
    return phi.wavefunction * (
        vl * (wf_in[0] * wf_out[0] + wf_in[1] * wf_out[1])
        + vr * (wf_in[2] * wf_out[2] + wf_in[3] * wf_out[3])
    )
