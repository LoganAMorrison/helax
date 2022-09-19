import chex
import jax
import jax.numpy as jnp

from helax.vertices import VertexFFS, VertexFFV

from .lvector import ldot, lnorm_sqr
from .wavefunctions import DiracWf, ScalarWf, VectorWf

im = 1.0j


# =============================================================================
# ---- Propagators ------------------------------------------------------------
# =============================================================================


def complex_mass_sqr(mass: float, width: float) -> complex:
    return mass**2 - im * mass * width


def propagator_den(momentum: chex.Array, mass: float, width: float) -> chex.Array:
    return im / (lnorm_sqr(momentum) - complex_mass_sqr(mass, width))


def attach_dirac(psi: DiracWf, mass: float, width: float) -> DiracWf:
    p = psi.momentum
    f = psi.wavefunction

    den = propagator_den(psi.momentum, mass, width)

    p1p2 = p[1] + im * p[2]
    p1m2 = p[1] - im * p[2]
    p0p3 = p[0] + p[3]
    p0m3 = p[0] - p[3]

    wavefunction = jax.lax.switch(
        (psi.direction + 1) // 2,
        # (p_slash + m).psi
        (
            [
                (mass * f[0] - f[3] * p1m2 + f[2] * p0m3) * den,
                (mass * f[1] - f[2] * p1p2 + f[3] * p0p3) * den,
                (mass * f[2] + f[1] * p1m2 + f[0] * p0p3) * den,
                (mass * f[3] + f[0] * p1p2 + f[1] * p0m3) * den,
            ]
        ),
        # psi.(p_slash + m)
        (
            [
                (mass * f[0] + f[3] * p1p2 + f[2] * p0p3) * den,
                (mass * f[1] + f[2] * p1m2 + f[3] * p0m3) * den,
                (mass * f[2] - f[1] * p1p2 + f[0] * p0m3) * den,
                (mass * f[3] - f[0] * p1m2 + f[1] * p0p3) * den,
            ]
        ),
    )

    return DiracWf(
        wavefunction=wavefunction, momentum=psi.momentum, direction=psi.direction
    )


def attach_vector(eps: VectorWf, mass: float, width: float) -> VectorWf:
    wf = eps.wavefunction
    k = eps.momentum
    den = propagator_den(k, mass, width)

    invcm2 = jax.lax.cond(
        mass == 0.0, lambda: 0.0j, lambda: 1.0 / complex_mass_sqr(mass, width)
    )

    wf = (-wf + k * ldot(k, wf) * invcm2) * den

    return VectorWf(wavefunction=wf, momentum=k, direction=eps.direction)


def attach_scalar(phi: ScalarWf, mass: float, width: float) -> ScalarWf:
    wf = phi.wavefunction * propagator_den(phi.momentum, mass, width)
    return ScalarWf(wavefunction=wf, momentum=phi.momentum, direction=phi.direction)


# =============================================================================
# ---- Currents ---------------------------------------------------------------
# =============================================================================


def current_fs_to_f(
    vertex: VertexFFS, mass: float, width: float, psi: DiracWf, phi: ScalarWf
) -> DiracWf:
    phi_wf = phi.wavefunction
    fi = psi.wavefunction
    vl = vertex.left
    vr = vertex.right

    momentum = psi.momentum * psi.direction + phi.momentum

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


def current_fv_to_f(
    vertex: VertexFFV, mass: float, width: float, psi: DiracWf, polvec: VectorWf
) -> DiracWf:
    eps = polvec.wavefunction
    f = psi.wavefunction
    vl = vertex.left
    vr = vertex.right

    momentum = psi.momentum * psi.direction * polvec.momentum

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


# =============================================================================
# ---- Amplitudes -------------------------------------------------------------
# =============================================================================


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
