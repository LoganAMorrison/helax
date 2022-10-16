"""Module for computing currents and amplitudes.


Notes on signs
==============

All momentum are considered incoming, except for fermions. For fermions, the
momentum always points along the direction of the fermion flow.

[v,s](p) + [v,s](q) -> [v,s](k)
-----------------------
k = p + q

    p ->     k=p+q ->
    -------x------
          /
    -----
    q ->


f(p) + [v,s](q) -> f(k)
-----------------------
k = p + q

    p ->     ->  k=p+q
    --->---x--->---
          /
    -----
    q ->


fbar(p) + [v,s](q) -> fbar(k)
-----------------------------
k = p - q

    p <-     <- k=p-q
    ---<---x---<---
          /
    -----
    q ->


f(p) + fbar(q) -> [v,s](k)
--------------------------
k = p - q

    p ->     -> k=p-q
    --->---x------
          /
    --<--
    q <-


w(p) + w(q) -> s(k)
-------------------
k = p + q

    p ->     -> k=p+q
    --->---x------
          /
    -->--
    q ->


w^+(p) + w^+(q) -> s(k)
---------------------------
k = -p - q

    p <-     -> k=-p-q
    ---<---x------
          /
    --<--
    q <-
"""


import numpy as np

from helax.numpy.dirac import WeylS0, WeylS1, WeylS2, WeylS3
from helax.numpy.lvector import ldot, lnorm_sqr
from helax.numpy.typing import RealArray
from helax.numpy.wavefunctions import DiracWf, ScalarWf, VectorWf
from helax.numpy.wavefunctions.weyl import WeylType, WeylWf

IM = 1.0j
DAGGER_TYPE = WeylType.Xd | WeylType.Yd


def complex_mass_sqr(mass: float, width: float) -> complex:
    """Returns the complexified mass given the particle's width."""
    return mass * (mass - 1j * width)


def propagator_den(momentum: RealArray, mass: float, width: float) -> RealArray:
    """Returns the scalar component of the propagator."""
    return IM / (lnorm_sqr(momentum) - complex_mass_sqr(mass, width))


def attach_dirac(psi: DiracWf, mass: float, width: float) -> DiracWf:
    """Attach a propagator to a Dirac wavefunction."""
    p = psi.momentum
    f = psi.wavefunction

    den = propagator_den(psi.momentum, mass, width)

    p1p2 = p[1] + IM * p[2]
    p1m2 = p[1] - IM * p[2]
    p0p3 = p[0] + p[3]
    p0m3 = p[0] - p[3]

    if psi.direction == -1:
        wavefunction = np.array(
            [
                (mass * f[0] - f[3] * p1m2 + f[2] * p0m3) * den,
                (mass * f[1] - f[2] * p1p2 + f[3] * p0p3) * den,
                (mass * f[2] + f[1] * p1m2 + f[0] * p0p3) * den,
                (mass * f[3] + f[0] * p1p2 + f[1] * p0m3) * den,
            ]
        )
    else:
        wavefunction = np.array(
            [
                (mass * f[0] + f[3] * p1p2 + f[2] * p0p3) * den,
                (mass * f[1] + f[2] * p1m2 + f[3] * p0m3) * den,
                (mass * f[2] - f[1] * p1p2 + f[0] * p0m3) * den,
                (mass * f[3] - f[0] * p1m2 + f[1] * p0p3) * den,
            ]
        )

    return DiracWf(
        wavefunction=wavefunction, momentum=psi.momentum, direction=psi.direction
    )


def attach_vector(eps: VectorWf, mass: float, width: float) -> VectorWf:
    wf = eps.wavefunction
    k = eps.momentum
    den = propagator_den(k, mass, width)

    if mass == 0:
        invcm2 = 0.0j
    else:
        invcm2 = 1.0 / complex_mass_sqr(mass, width)

    wf = (-wf + k * ldot(k, wf) * invcm2) * den

    return VectorWf(wavefunction=wf, momentum=k, direction=eps.direction)


def attach_scalar(phi: ScalarWf, mass: float, width: float) -> ScalarWf:
    wf = phi.wavefunction * propagator_den(phi.momentum, mass, width)
    return ScalarWf(wavefunction=wf, momentum=phi.momentum, direction=phi.direction)


def attach_weyl(chi: WeylWf, mass: float, width: float) -> WeylWf:
    """Attach a propagator to a Weyl wavefunction."""
    p = chi.momentum
    chi1 = chi.wavefunction[0]
    chi2 = chi.wavefunction[1]

    den = propagator_den(p, mass, width)

    if chi.type & DAGGER_TYPE:
        p_dot_sigma = p[0] * WeylS0 + p[1] * WeylS1 + p[2] * WeylS2 + p[3] * WeylS3
        # If the spinor is a daggered spinor, then we want to compute
        # z^b = x_a sigma^ab. This will yield a spinor with a raised index.
        # To lower the index, we contract with the spinor metric tensor:
        #   z_a = eps_ab * z^b
        #     => z_1 = eps_12 * z^2 = -z^2
        #     => z_2 = eps_21 * z^1 = +z^1
        eta1 = IM * (p_dot_sigma[0, 0] * chi1 + p_dot_sigma[0, 1] * chi2) * den
        eta2 = IM * (p_dot_sigma[1, 0] * chi1 + p_dot_sigma[1, 1] * chi2) * den

        eta1, eta2 = -eta2, eta1

    else:
        p_dot_sigma = p[0] * WeylS0 - p[1] * WeylS1 - p[2] * WeylS2 - p[3] * WeylS3
        # If the spinor isn't a daggered spinor, then we want to compute
        # z_b = x^a sigma_ab. But our spinor is x_a. Need to raise the index,
        # yielding: x^a = eps^ab x_b
        #    => x^1 = eps^12 x_2 = +x_2
        #    => x^2 = eps^21 x_1 = -x_1
        chi1, chi2 = chi2, -chi1

        eta1 = IM * (p_dot_sigma[0, 0] * chi1 + p_dot_sigma[0, 1] * chi2) * den
        eta2 = IM * (p_dot_sigma[1, 0] * chi1 + p_dot_sigma[1, 1] * chi2) * den

    wavefunction = np.array([eta1, eta2])

    if chi.type == WeylType.X:
        new_type = WeylType.Xd
    elif chi.type == WeylType.Xd:
        new_type = WeylType.X
    elif chi.type == WeylType.Y:
        new_type = WeylType.Yd
    elif chi.type == WeylType.Yd:
        new_type = WeylType.Y
    else:
        raise ValueError(f"Invalid weyl type '{chi.type}'.")

    return WeylWf(wavefunction=wavefunction, momentum=p, type=new_type)


def attach_weyl_mass(chi: WeylWf, mass: float, width: float) -> WeylWf:
    """Attach a propagator to a Dirac wavefunction."""
    p = chi.momentum
    wf = chi.wavefunction

    den = propagator_den(p, mass, width)
    prop = IM * mass * den
    wavefunction = np.array([prop * wf[0], prop * wf[1]])

    return WeylWf(wavefunction=wavefunction, momentum=p, type=chi.type)
