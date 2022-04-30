import functools

import jax
import jax.numpy as jnp

from helax.wavefunctions import DiracWf, ScalarWf, VectorWf

from ._propagators import attach_dirac, attach_scalar, attach_vector
from ._vertices import VertexFFS, VertexFFV

im = 1.0j

D_AXES = DiracWf(wavefunction=1, momentum=1, direction=None)
V_AXES = VectorWf(wavefunction=1, momentum=1, direction=None)
S_AXES = ScalarWf(wavefunction=0, momentum=1, direction=None)


def current_vmap(func, *, in1, in2, out):
    return jax.vmap(
        func,
        in_axes=(None, None, None, in1, in2),  # type: ignore
        out_axes=out,
    )


# @functools.partial(current_vmap, in1=D_AXES, in2=S_AXES, out=D_AXES)
def current_fs_to_f(
    vertex: VertexFFS, mass: float, width: float, psi: DiracWf, phi: ScalarWf
) -> DiracWf:
    phi_wf = phi.wavefunction
    fi = psi.wavefunction
    vl = vertex.left
    vr = vertex.right

    momentum = psi.momentum + psi.direction + phi.momentum

    wavefunction = jnp.array(
        [
            vl * phi_wf * fi[0],
            vl * phi_wf * fi[1],
            vr * phi_wf * fi[2],
            vr * phi_wf * fi[3],
        ]
    )

    psi = DiracWf(wavefunction=wavefunction, momentum=momentum, direction=psi.direction)
    return attach_dirac(psi, mass, width)


@functools.partial(current_vmap, in1=D_AXES, in2=D_AXES, out=S_AXES)
def current_ff_to_s(
    vertex: VertexFFS, mass: float, width: float, psi_out: DiracWf, psi_in: DiracWf
) -> ScalarWf:
    fi = psi_in.wavefunction
    fo = psi_out.wavefunction
    vl = vertex.left
    vr = vertex.right

    momentum = psi_in.momentum - psi_out.momentum
    wavefunction = vl * (fi[0] * fo[0] + fi[1] * fo[1]) + vr * (
        fi[2] * fo[2] + fi[3] * fo[3]
    )

    phi = ScalarWf(wavefunction=wavefunction, momentum=momentum, direction=1)

    return attach_scalar(phi, mass, width)


@functools.partial(current_vmap, in1=D_AXES, in2=V_AXES, out=D_AXES)
def current_fv_to_f(
    vertex: VertexFFV, mass: float, width: float, psi: DiracWf, polvec: VectorWf
) -> DiracWf:
    eps = polvec.wavefunction
    f = psi.wavefunction
    vl = vertex.left
    vr = vertex.right

    momentum = psi.momentum + psi.direction * polvec.momentum

    wavefunction = jax.lax.switch(
        (psi.direction + 1) // 2,
        # phi.(gl g[mu].PL + gr g[mu].PR) eps[mu]
        jnp.array(
            [
                vl * (f[3] * (eps[1] + im * eps[2]) + f[2] * (eps[0] + eps[3])),
                vl * (f[2] * (eps[1] - im * eps[2]) + f[3] * (eps[0] - eps[3])),
                vr * (-f[1] * (eps[1] + im * eps[2]) + f[0] * (eps[0] - eps[3])),
                vr * (-f[0] * (eps[1] - im * eps[2]) + f[1] * (eps[0] + eps[3])),
            ]
        ),
        # (gl g[mu].PL + gr g[mu].PR).psi eps[mu]
        jnp.array(
            [
                vr * (-f[3] * (eps[1] - im * eps[2]) + f[2] * (eps[0] - eps[3])),
                vr * (-f[2] * (eps[1] + im * eps[2]) + f[3] * (eps[0] + eps[3])),
                vl * (f[1] * (eps[1] - im * eps[2]) + f[0] * (eps[0] + eps[3])),
                vl * (f[0] * (eps[1] + im * eps[2]) + f[1] * (eps[0] - eps[3])),
            ]
        ),
    )

    psi = DiracWf(wavefunction=wavefunction, momentum=momentum, direction=psi.direction)
    return attach_dirac(psi, mass, width)


@functools.partial(current_vmap, in1=D_AXES, in2=D_AXES, out=V_AXES)
def current_ff_to_v(
    vertex: VertexFFV, mass: float, width: float, psi_out: DiracWf, psi_in: DiracWf
) -> VectorWf:
    fi = psi_in.wavefunction
    fo = psi_out.wavefunction
    vl = vertex.left
    vr = vertex.right

    momentum = psi_in.momentum - psi_out.momentum

    wavefunction = jnp.array(
        [
            vr * (fi[2] * fo[0] + fi[3] * fo[1]) + vl * (fi[0] * fo[2] + fi[1] * fo[3]),
            vr * (fi[3] * fo[0] + fi[2] * fo[1]) - vl * (fi[1] * fo[2] + fi[0] * fo[3]),
            im * vr * (fi[2] * fo[1] - fi[3] * fo[0])
            + im * vl * (fi[1] * fo[2] - fi[0] * fo[3]),
            vr * (fi[2] * fo[0] - fi[3] * fo[1]) + vl * (fi[1] * fo[3] - fi[0] * fo[2]),
        ]
    )

    return attach_vector(
        VectorWf(wavefunction=wavefunction, momentum=momentum, direction=1), mass, width
    )
