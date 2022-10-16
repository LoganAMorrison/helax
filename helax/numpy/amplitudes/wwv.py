"""Module for computing currents and amplitudes."""


import numpy as np

from helax.numpy.wavefunctions import VectorWf
from helax.numpy.wavefunctions.weyl import WeylType, WeylWf
from helax.vertices import VertexWWV

from .propagators import attach_vector

IM = 1.0j
DAGGER_TYPE = WeylType.Xd | WeylType.Yd


# =============================================================================
# ---- Currents ---------------------------------------------------------------
# =============================================================================


def current_ww_to_v(
    vertex: VertexWWV, mass: float, width: float, chi: WeylWf, eta: WeylWf
) -> VectorWf:
    """Fuse two weyl spinors into a vector boson.

    Parameters
    ----------
    vertex : VertexWWV
        Feynman rule for the W-W-V vertex.
    mass : float
        Mass of the resulting vector.
    width : float
        Width of the resulting vector.
    chi, eta : WeylWf
        Weyl spinors. One must be daggered and the other un-daggered.

    Returns
    -------
    eps: VectorWf
        Resulting vector boson wavefuction.
    """
    dagger_type = WeylType.Xd | WeylType.Yd
    is_chi_dagger = chi.type & dagger_type
    is_eta_dagger = eta.type & dagger_type
    assert is_chi_dagger != is_eta_dagger, (
        "Invalid input spinors."
        + " One spinor must be daggered and the other un-daggered."
    )

    if is_chi_dagger:
        wl = chi.wavefunction
        pout = chi.momentum
        wr = eta.wavefunction
        pin = eta.momentum
    else:
        wl = eta.wavefunction
        pout = eta.momentum
        wr = chi.wavefunction
        pin = chi.momentum

    momentum = pin - pout

    wavefunction = vertex.g * np.array(
        [
            # sigma[0]
            wl[0] * wr[0] + wl[1] * wr[1],
            # sigma[1]
            -(wl[0] * wr[1] + wl[1] * wr[0]),
            # sigma[2]
            -IM * (-wl[0] * wr[1] + wl[1] * wr[0]),
            # sigma[3]
            -(wl[0] * wr[0] - wl[1] * wr[1]),
        ]
    )

    return attach_vector(
        VectorWf(wavefunction=wavefunction, momentum=momentum, direction=1), mass, width
    )
