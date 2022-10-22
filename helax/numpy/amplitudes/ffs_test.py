from typing import Tuple

import numpy as np
import pytest

from helax.numpy import amplitudes, wavefunctions
from helax.numpy.phase_space import PhaseSpace
from helax.numpy.utils import abs2, kallen_lambda
from helax.numpy.wavefunctions import DiracWf, ScalarWf
from helax.vertices import VertexFFS

MUON_MASS = 1.056584e-01
G_FERMI = 1.166379e-05
EL = 3.028620e-01
SW = 4.808530e-01
CW = np.sqrt(1.0 - SW**2)
MASS_W = 8.038500e01
WIDTH_W = 2.085000e00
MASS_B = 4.18
MASS_T = 172.9
MASS_MU = 105.6583745e-3
MASS_E = 0.5109989461e-3
MASS_Z = 91.1876

MASS_H = 125.00
WIDTH_H = 0.00374
VEV_H = 246.21965

SEED = 1234


def spinor_u(momenta, mass: float) -> Tuple[DiracWf, DiracWf]:
    return (
        wavefunctions.spinor_u(momenta, mass, -1),
        wavefunctions.spinor_u(momenta, mass, 1),
    )


def spinor_v(momenta, mass: float) -> Tuple[DiracWf, DiracWf]:
    return (
        wavefunctions.spinor_v(momenta, mass, -1),
        wavefunctions.spinor_v(momenta, mass, 1),
    )


def spinor_ubar(momenta, mass: float) -> Tuple[DiracWf, DiracWf]:
    return (
        wavefunctions.spinor_ubar(momenta, mass, -1),
        wavefunctions.spinor_ubar(momenta, mass, 1),
    )


def spinor_vbar(momenta, mass: float) -> Tuple[DiracWf, DiracWf]:
    return (
        wavefunctions.spinor_vbar(momenta, mass, -1),
        wavefunctions.spinor_vbar(momenta, mass, 1),
    )


def scalar_wf(momenta, out: bool) -> ScalarWf:
    return wavefunctions.scalar_wf(momenta, out)


def current_ff_to_s(vertex, mass, width, psi_out: DiracWf, psi_in: DiracWf):
    return amplitudes.current_ff_to_s(vertex, mass, width, psi_out, psi_in)


def amplitude_ffs(vertex, psi_out: DiracWf, psi_in: DiracWf, phi: ScalarWf):
    return amplitudes.amplitude_ffs(vertex, psi_out, psi_in, phi)


def wavefunction_iter(*wavefuncs):
    idx_list = []
    for wf in wavefuncs:
        if isinstance(wf, wavefunctions.ScalarWf):
            idx_list.append([0])
        elif isinstance(wf, wavefunctions.DiracWf):
            idx_list.append([0, 1])
        elif isinstance(wf, wavefunctions.VectorWf):
            idx_list.append([0, 1, 2])
        else:
            raise ValueError(f"Invalid wf {type(wf)}")

    idxs = np.array(np.meshgrid(*idx_list)).T.reshape(-1, len(idx_list))
    for idx in idxs:
        yield tuple([wf[i] for wf, i in zip(wavefuncs, idx)])


# ============================================================================
# ---- H -> f + f ------------------------------------------------------------
# ============================================================================


def widths_h_to_f_f(mf, ncf):
    yf = mf / VEV_H
    vertex = VertexFFS(
        left=yf / np.sqrt(2),
        right=yf / np.sqrt(2),
    )

    def msqrd_h_to_f_f(momenta):
        p1 = momenta[:, 0]
        p2 = momenta[:, 1]
        ph = np.sum(momenta, axis=1)

        wf1s = spinor_ubar(p1, mf)
        wf2s = spinor_v(p2, mf)
        h_wf = scalar_wf(ph, False)

        def msqrd(i1, i2):
            return abs2(amplitude_ffs(vertex, wf1s[i2], wf2s[i1], h_wf))

        idxs = np.array(np.meshgrid([0, 1], [0, 1])).T.reshape(-1, 2)
        res = sum(msqrd(i1, i2) for (i1, i2) in idxs)

        return ncf * res

    mu = mf / MASS_H
    analytic = ncf * yf**2 * MASS_H * (1 - 4 * mu**2) ** 1.5 / (16 * np.pi)

    phase_space = PhaseSpace(MASS_H, [mf, mf], msqrd=msqrd_h_to_f_f)
    width = phase_space.decay_width(1000, seed=SEED)[0]

    return width, analytic


def test_width_h_to_b_b():
    width, analytic = widths_h_to_f_f(mf=MASS_B, ncf=3.0)
    assert float(width) == pytest.approx(analytic, rel=1e-3, abs=0.0)


# ============================================================================
# ---- chi + chi -> S -> f + f -----------------------------------------------
# ============================================================================


def test_dark_matter_annihilation_scalar_mediator():
    gsxx = 1.0
    gsff = 1.0
    mx = 10.0
    mf = 1.0
    cme = 3 * mx
    mass_s = 1.0
    width_s = 1.0

    vxx = VertexFFS(left=gsxx, right=gsxx)
    vff = VertexFFS(left=gsff, right=gsff)

    ex = cme / 2.0
    px = np.sqrt(ex**2 - mx**2)
    p1 = np.expand_dims(np.array([ex, 0.0, 0.0, px]), -1)
    p2 = np.expand_dims(np.array([ex, 0.0, 0.0, -px]), -1)

    wf1s = spinor_u(p1, mx)
    wf2s = spinor_vbar(p2, mx)

    def msqrd_x_x_to_f_f(momenta):
        p3 = momenta[:, 0]
        p4 = momenta[:, 1]

        wf3s = spinor_ubar(p3, mf)
        wf4s = spinor_v(p4, mf)

        def msqrd(i1, i2, i3, i4):
            swf = current_ff_to_s(vxx, mass_s, width_s, wf2s[i2], wf1s[i1])
            return abs2(amplitude_ffs(vff, wf3s[i3], wf4s[i4], swf))

        idxs = np.array(np.meshgrid([0, 1], [0, 1], [0, 1], [0, 1])).T.reshape(-1, 4)
        res = sum(msqrd(i1, i2, i3, i4) for (i1, i2, i3, i4) in idxs)

        return res / 4.0

    phase_space = PhaseSpace(cme, [mf, mf], msqrd=msqrd_x_x_to_f_f)
    cs = phase_space.cross_section(mx, mx, 10_000, seed=SEED)[0]

    s = cme**2
    analytic = (
        gsff**2
        * gsxx**2
        * (s - 4 * mf**2)
        * np.sqrt((s - 4 * mf**2) * (s - 4 * mx**2))
    ) / (16 * np.pi * s * ((s - mass_s**2) ** 2 + (mass_s * width_s) ** 2))

    assert float(cs) == pytest.approx(analytic, rel=1e-2, abs=0.0)


# ============================================================================
# ---- Ni  -> Nj + phi -------------------------------------------------------
# ============================================================================


def widths_ni_to_nj_phi(mi, mj, mphi, y):

    vertex = VertexFFS(left=y, right=y)

    def msqrd_ni_to_nj_phi(momenta):
        pj = momenta[:, 0]
        pphi = momenta[:, 1]
        pi = np.sum(momenta, axis=1)

        wfi_u = spinor_u(pi, mi)
        wfj_ubar = spinor_ubar(pj, mj)

        wfphi = scalar_wf(pphi, mphi)

        def msqrd(ii, ij):
            amp = amplitude_ffs(vertex, wfj_ubar[ij], wfi_u[ii], wfphi)
            return abs2(amp)

        idxs = np.array(np.meshgrid([0, 1], [0, 1])).T.reshape(-1, 2)
        res = sum(msqrd(ii, ij) for (ii, ij) in idxs)

        return res / 2.0

    rj = mj**2 / mi**2
    rphi = mphi**2 / mi**2

    analytic = (
        mi
        / (16 * np.pi)
        * np.sqrt(kallen_lambda(1, rphi, rj))
        * (np.abs(y) ** 2 * (1 + rj - rphi) + 2 * np.sqrt(rj) * np.real(y**2))
    )

    phase_space = PhaseSpace(mi, [mj, mphi], msqrd=msqrd_ni_to_nj_phi)
    width = phase_space.decay_width(10_000, seed=SEED)[0]

    return width, analytic


def test_width_ni_to_nj_phi():
    mi = 100.0
    mj = 25.0
    mphi = 3.0
    y = 1.0

    width, analytic = widths_ni_to_nj_phi(mi=mi, mj=mj, mphi=mphi, y=y)
    assert float(width / analytic) == pytest.approx(1, rel=1e-3, abs=0.0)
