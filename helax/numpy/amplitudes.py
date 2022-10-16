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

from helax.vertices import VertexFFS, VertexFFV, VertexWWS, VertexWWV

from .dirac import WeylS0, WeylS1, WeylS2, WeylS3
from .lvector import ldot, lnorm_sqr
from .typing import RealArray
from .wavefunctions import DiracWf, ScalarWf, VectorWf, WeylType, WeylWf

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
    fi = psi_in.wavefunction
    fo = psi_out.wavefunction
    vl = vertex.left
    vr = vertex.right

    momentum = psi_in.momentum - psi_out.momentum

    wavefunction = np.array(
        [
            vr * (fi[2] * fo[0] + fi[3] * fo[1]) + vl * (fi[0] * fo[2] + fi[1] * fo[3]),
            vr * (fi[3] * fo[0] + fi[2] * fo[1]) - vl * (fi[1] * fo[2] + fi[0] * fo[3]),
            IM * vr * (fi[2] * fo[1] - fi[3] * fo[0])
            + IM * vl * (fi[1] * fo[2] - fi[0] * fo[3]),
            vr * (fi[2] * fo[0] - fi[3] * fo[1]) + vl * (fi[1] * fo[3] - fi[0] * fo[2]),
        ]
    )

    return attach_vector(
        VectorWf(wavefunction=wavefunction, momentum=momentum, direction=1), mass, width
    )


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


def current_ws_to_w(
    vertex: VertexWWS, mass: float, width: float, chi: WeylWf, phi: ScalarWf
) -> WeylWf:
    """Fuse two weyl spinors into a scalar boson.

    Parameters
    ----------
    vertex : VertexWWS
        Feynman rule for the W-W-S vertex.
    mass : float
        Mass of the resulting vector.
    width : float
        Width of the resulting vector.
    chi, eta : WeylWf
        Weyl spinors. Both spinors must be daggered or un-daggered.

    Returns
    -------
    phi: ScalarWf
        Resulting scalar boson wavefuction.
    """
    is_chi_dagger = chi.type & DAGGER_TYPE

    if is_chi_dagger:
        sgn = -1.0
    else:
        sgn = 1.0

    momentum = chi.momentum + sgn * phi.momentum
    wavefunction = vertex.g * chi.wavefunction

    return attach_weyl(
        WeylWf(wavefunction=wavefunction, momentum=momentum, type=chi.type), mass, width
    )


def current_ww_to_s(
    vertex: VertexWWS, mass: float, width: float, chi: WeylWf, eta: WeylWf
) -> ScalarWf:
    """Fuse two weyl spinors into a scalar boson.

    Parameters
    ----------
    vertex : VertexWWS
        Feynman rule for the W-W-S vertex.
    mass : float
        Mass of the resulting vector.
    width : float
        Width of the resulting vector.
    chi, eta : WeylWf
        Weyl spinors. Both spinors must be daggered or un-daggered.

    Returns
    -------
    phi: ScalarWf
        Resulting scalar boson wavefuction.
    """
    dagger_type = WeylType.Xd | WeylType.Yd
    is_chi_dagger = chi.type & dagger_type
    is_eta_dagger = eta.type & dagger_type
    assert is_chi_dagger == is_eta_dagger, (
        "Invalid input spinors."
        + " Both spinors must be daggered or both must be un-daggered."
    )

    if is_chi_dagger:
        wl = chi.wavefunction
        wr = eta.wavefunction
        sgn = -1
    else:
        wl = eta.wavefunction
        wr = chi.wavefunction
        sgn = 1.0

    momentum = sgn * (chi.momentum + eta.momentum)
    wavefunction = vertex.g * (wl[0] * wr[0] - wl[1] * wr[1])

    return attach_scalar(
        ScalarWf(wavefunction=wavefunction, momentum=momentum, direction=1), mass, width
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

    fi = psi_in.wavefunction
    fo = psi_out.wavefunction
    vl = vertex.left
    vr = vertex.right
    return phi.wavefunction * (
        vl * (fi[0] * fo[0] + fi[1] * fo[1]) + vr * (fi[2] * fo[2] + fi[3] * fo[3])
    )
