import functools

import jax

from helax.wavefunctions import DiracWf, ScalarWf, VectorWf

from ._vertices import VertexFFS, VertexFFV

D_IN_AXES = DiracWf(wavefunction=1, momentum=1, direction=None)
V_IN_AXES = VectorWf(wavefunction=1, momentum=1, direction=None)
S_IN_AXES = ScalarWf(wavefunction=1, momentum=1, direction=None)


@functools.partial(jax.vmap, in_axes=(None, D_IN_AXES, D_IN_AXES, V_IN_AXES))
def amplitude_ffv(
    vertex: VertexFFV, psi_out: DiracWf, psi_in: DiracWf, polvec: VectorWf
):
    # assert psi_out.direction == -1, "`psi_out` must be have flow out."
    # assert psi_in.direction == 1, "`psi_in` must be have flow in."

    fi = psi_in.wavefunction
    fo = psi_out.wavefunction
    eps = polvec.wavefunction
    vl = vertex.left
    vr = vertex.right
    im = 1.0j

    eps0p3 = eps[0] + eps[3]
    eps0m3 = eps[0] - eps[3]

    eps1p2 = eps[1] + im * eps[2]
    eps1m2 = eps[1] - im * eps[2]

    return vr * (
        fi[2] * (-fo[1] * eps1p2 + fo[0] * eps0m3)
        + fi[3] * (-fo[0] * eps1m2 + fo[1] * eps0p3)
    ) + vl * (
        fi[1] * (fo[2] * eps1m2 + fo[3] * eps0m3)
        + fi[0] * (fo[3] * eps1p2 + fo[2] * eps0p3)
    )


# @functools.partial(jax.vmap, in_axes=(None, D_IN_AXES, D_IN_AXES, None))
def amplitude_ffs(vertex: VertexFFS, psi_out: DiracWf, psi_in: DiracWf, phi: ScalarWf):
    # assert psi_out.direction == -1, "`psi_out` must be have flow out."
    # assert psi_in.direction == 1, "`psi_in` must be have flow in."

    fi = psi_in.wavefunction
    fo = psi_out.wavefunction
    vl = vertex.left
    vr = vertex.right
    return phi.wavefunction * (
        vl * (fi[0] * fo[0] + fi[1] * fo[1]) + vr * (fi[2] * fo[2] + fi[3] * fo[3])
    )
