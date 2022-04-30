import chex
import jax

from helax.lvector import ldot, lnorm_sqr
from helax.wavefunctions import DiracWf, ScalarWf, VectorWf

im = 1.0j


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
