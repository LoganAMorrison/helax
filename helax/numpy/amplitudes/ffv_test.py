from typing import Tuple

import numpy as np
import pytest

from helax.numpy import amplitudes, wavefunctions
from helax.numpy.lvector import lnorm_sqr
from helax.numpy.phase_space import PhaseSpace
from helax.numpy.utils import abs2, kallen_lambda
from helax.numpy.wavefunctions import DiracWf, VectorWf
from helax.vertices import VertexFFV

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


def polvec(momenta, mass: float, out: bool) -> Tuple[VectorWf, VectorWf, VectorWf]:
    return (
        wavefunctions.vector_wf(momenta, mass, -1, out),
        wavefunctions.vector_wf(momenta, mass, 0, out),
        wavefunctions.vector_wf(momenta, mass, 1, out),
    )


def current_ff_to_v(
    vertex, mass: float, width: float, psi_out: DiracWf, psi_in: DiracWf
):
    return amplitudes.current_ff_to_v(vertex, mass, width, psi_out, psi_in)


def amplitude_ffv(vertex, psi_out: DiracWf, psi_in: DiracWf, eps: VectorWf):
    return amplitudes.amplitude_ffv(vertex, psi_out, psi_in, eps)


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


def test_width_mu_to_e_nu_nu():
    def msqrd_mu_to_e_nu_nu_analytic(momenta):
        t = lnorm_sqr(momenta[:, 0] + momenta[:, 2])
        s = lnorm_sqr(momenta[:, 1] + momenta[:, 1])

        return (
            -0.125
            * (
                EL**4
                * (
                    MASS_E**4 * MASS_MU**2 * (MASS_MU**2 - s - t)
                    + 4 * MASS_W**4 * t * (-(MASS_MU**2) + t)
                    + MASS_E**2
                    * (
                        -4 * MASS_W**4 * t
                        - MASS_MU**4 * (s + t)
                        + MASS_MU**2
                        * (4 * MASS_W**4 + 4 * MASS_W**2 * s + (s + t) ** 2)
                    )
                )
            )
            / (
                MASS_W**4
                * SW**4
                * (
                    (-(MASS_E**2) - MASS_MU**2 + MASS_W**2 + s + t) ** 2
                    + MASS_W**2 * WIDTH_W**2
                )
            )
        )

    def msqrd_mu_to_e_nu_nu_hel(momenta):
        pe = momenta[:, 0]
        pve = momenta[:, 1]
        pvm = momenta[:, 2]
        pmu = np.sum(momenta, axis=1)

        v_wll = VertexFFV(left=EL / (np.sqrt(2) * SW), right=0.0)

        mu_wfs = spinor_u(pmu, MUON_MASS)
        e_wfs = spinor_ubar(pe, 0.0)
        ve_wfs = spinor_v(pve, 0.0)
        vm_wfs = spinor_ubar(pvm, 0.0)

        def msqrd(imu, ie, ive, ivm):
            w_wf = current_ff_to_v(v_wll, MASS_W, WIDTH_W, vm_wfs[ivm], mu_wfs[imu])
            amp = amplitude_ffv(v_wll, e_wfs[ie], ve_wfs[ive], w_wf)
            return abs2(amp)

        idxs = np.array(np.meshgrid([0, 1], [0, 1], [0, 1], [0, 1])).T.reshape(-1, 4)
        res = sum(msqrd(i_n, i_v1, i_v2, i_v3) for (i_n, i_v1, i_v2, i_v3) in idxs)

        return res / 2.0

    phase_space = PhaseSpace(MASS_MU, [MASS_E, 0.0, 0.0], msqrd=msqrd_mu_to_e_nu_nu_hel)
    width = phase_space.decay_width(10_000, seed=SEED)[0]

    phase_space = PhaseSpace(
        MASS_MU, [MASS_E, 0.0, 0.0], msqrd=msqrd_mu_to_e_nu_nu_analytic
    )
    width2 = phase_space.decay_width(10_000, seed=SEED)[0]

    assert float(width) == pytest.approx(width2, abs=0.0, rel=0.01)


def test_width_t_to_b_w():
    def msqrd_t_to_b_w(momenta):
        pb = momenta[:, 0]
        pw = momenta[:, 1]
        pt = np.sum(momenta, axis=1)

        t_wfs = spinor_u(pt, MASS_T)
        b_wfs = spinor_ubar(pb, MASS_B)
        w_wfs = polvec(pw, MASS_W, True)

        v_tbw = VertexFFV(left=EL / (np.sqrt(2.0) * SW), right=0.0)

        def msqrd(it, ib, iw):
            amp = amplitude_ffv(v_tbw, b_wfs[ib], t_wfs[it], w_wfs[iw])
            return abs2(amp)

        idxs = np.array(np.meshgrid([0, 1], [0, 1], [0, 1, 2])).T.reshape(-1, 3)
        res = sum(msqrd(it, ib, iw) for (it, ib, iw) in idxs)

        return res / 2.0

    analytic = (
        EL**2
        * (
            (MASS_T**2 + 2 * MASS_W**2) * (MASS_T**2 - MASS_W**2)
            + MASS_B**2 * (MASS_W**2 - 2 * MASS_T**2)
            + MASS_B**4
        )
        * np.sqrt(kallen_lambda(MASS_B**2, MASS_T**2, MASS_W**2))
    ) / (64 * MASS_T**3 * MASS_W**2 * np.pi * SW**2)

    phase_space = PhaseSpace(MASS_T, [MASS_B, MASS_W], msqrd=msqrd_t_to_b_w)
    width = phase_space.decay_width(10_000, seed=SEED)[0]

    assert float(width) == pytest.approx(analytic, rel=1e-3, abs=0.0)


# ============================================================================
# ---- chi + chi -> V -> f + f -----------------------------------------------
# ============================================================================


def test_dark_matter_annihilation_vector_mediator():
    gvxx = 1.0
    gvff = 1.0
    mx = 10.0
    mf = 1.0
    cme = 3 * mx
    mv = 1.0
    widthv = 1.0

    vxx = VertexFFV(left=gvxx, right=gvxx)
    vff = VertexFFV(left=gvff, right=gvff)

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
            vwf = current_ff_to_v(vxx, mv, widthv, wf2s[i2], wf1s[i1])
            return abs2(amplitude_ffv(vff, wf3s[i3], wf4s[i4], vwf))

        idxs = np.array(np.meshgrid([0, 1], [0, 1], [0, 1], [0, 1])).T.reshape(-1, 4)
        res = sum(msqrd(i1, i2, i3, i4) for (i1, i2, i3, i4) in idxs)

        return res / 4.0

    phase_space = PhaseSpace(cme, [mf, mf], msqrd=msqrd_x_x_to_f_f)
    cs = phase_space.cross_section(mx, mx, 10_000, seed=SEED)[0]

    s = cme**2
    analytic = (
        gvff**2
        * gvxx**2
        * np.sqrt(s * (-4 * mf**2 + s))
        * (2 * mf**2 + s)
        * (2 * mx**2 + s)
    ) / (
        12.0
        * np.pi
        * s
        * np.sqrt(s * (-4 * mx**2 + s))
        * (mv**4 + s**2 + mv**2 * (-2 * s + widthv**2))
    )

    assert float(cs) == pytest.approx(analytic, rel=1e-2, abs=0.0)


# ============================================================================
# ---- Z -> f + f ------------------------------------------------------------
# ============================================================================


def widths_z_to_f_f(mf, ncf, t3f, qf):
    pre = EL / (SW * CW)
    left = pre * (t3f - qf * SW**2)
    right = -pre * qf * SW**2
    vertex = VertexFFV(left=left, right=right)

    def msqrd_z_to_f_f(momenta):
        pf1 = momenta[:, 0]
        pf2 = momenta[:, 1]
        pz = np.sum(momenta, axis=1)

        z_wfs = polvec(pz, MASS_Z, False)
        ubar = spinor_ubar(pf1, mf)
        v = spinor_v(pf2, mf)

        def msqrd(iz, i1, i2):
            amp = amplitude_ffv(vertex, ubar[i1], v[i2], z_wfs[iz])
            return abs2(amp)

        idxs = np.array(np.meshgrid([0, 1, 2], [0, 1], [0, 1])).T.reshape(-1, 3)
        res = sum(msqrd(iz, i1, i2) for (iz, i1, i2) in idxs)

        return ncf * res / 3.0

    mr2 = (mf / MASS_Z) ** 2
    af = t3f - qf * SW**2
    bf = -qf * SW**2

    analytic = (
        ncf
        * EL**2
        * MASS_Z
        / (24 * np.pi * CW**2 * SW**2)
        * np.sqrt(1 - 4 * mr2)
        * ((af**2 + bf**2) * (1 - mr2) + 6 * af * bf * mr2)
    )

    phase_space = PhaseSpace(MASS_Z, [mf, mf], msqrd=msqrd_z_to_f_f)
    width = phase_space.decay_width(10_000, seed=SEED)[0]

    return width, analytic


def test_width_z_to_b_b():
    mf = MASS_B
    ncf = 3
    t3f = -0.5
    qf = -1.0 / 3.0

    width, analytic = widths_z_to_f_f(mf=mf, ncf=ncf, t3f=t3f, qf=qf)
    assert float(width / analytic) == pytest.approx(1, rel=1e-3, abs=0.0)


def test_width_z_to_e_e():
    mf = MASS_E
    ncf = 1
    t3f = -0.5
    qf = -1.0

    width, analytic = widths_z_to_f_f(mf=mf, ncf=ncf, t3f=t3f, qf=qf)
    assert float(width / analytic) == pytest.approx(1.0, rel=1e-3, abs=0.0)
