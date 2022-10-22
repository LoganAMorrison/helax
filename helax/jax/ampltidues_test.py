import functools
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from helax.jax import amplitudes, phase_space, wavefunctions
from helax.jax.lvector import lnorm_sqr
from helax.jax.utils import abs2, kallen_lambda
from helax.jax.wavefunctions import DiracWf, ScalarWf, VectorWf

MUON_MASS = 1.056584e-01
G_FERMI = 1.166379e-05
EL = 3.028620e-01
SW = 4.808530e-01
MASS_W = 8.038500e01
WIDTH_W = 2.085000e00
MASS_B = 4.18
MASS_T = 172.9
MASS_MU = 105.6583745e-3
MASS_E = 0.5109989461e-3

MASS_H = 125.00
WIDTH_H = 0.00374
VEV_H = 246.21965


@jax.jit
def spinor_u(momenta, mass: float) -> Tuple[DiracWf, DiracWf]:
    return (
        wavefunctions.spinor_u(momenta, mass, -1),
        wavefunctions.spinor_u(momenta, mass, 1),
    )


@jax.jit
def spinor_v(momenta, mass: float) -> Tuple[DiracWf, DiracWf]:
    return (
        wavefunctions.spinor_v(momenta, mass, -1),
        wavefunctions.spinor_v(momenta, mass, 1),
    )


@jax.jit
def spinor_ubar(momenta, mass: float) -> Tuple[DiracWf, DiracWf]:
    return (
        wavefunctions.spinor_ubar(momenta, mass, -1),
        wavefunctions.spinor_ubar(momenta, mass, 1),
    )


@jax.jit
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


def scalar_wf(momenta, out: bool) -> ScalarWf:
    return wavefunctions.scalar_wf(momenta, out)


@functools.partial(jax.jit, static_argnums=(0, 1, 2))
def current_ff_to_v(
    vertex, mass: float, width: float, psi_out: DiracWf, psi_in: DiracWf
):
    return amplitudes.current_ff_to_v(vertex, mass, width, psi_out, psi_in)


@functools.partial(jax.jit, static_argnums=(0, 1, 2))
def current_ff_to_s(vertex, mass, width, psi_out: DiracWf, psi_in: DiracWf):
    return amplitudes.current_ff_to_s(vertex, mass, width, psi_out, psi_in)


@functools.partial(jax.jit, static_argnums=(0,))
def amplitude_ffv(vertex, psi_out: DiracWf, psi_in: DiracWf, eps: VectorWf):
    return amplitudes.amplitude_ffv(vertex, psi_out, psi_in, eps)


@functools.partial(jax.jit, static_argnums=(0,))
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
        pmu = jnp.sum(momenta, axis=1)

        v_wll = amplitudes.VertexFFV(left=EL / (np.sqrt(2) * SW), right=0.0)

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

    key = jax.random.PRNGKey(1234)
    width = phase_space.decay_width(
        MASS_MU, np.array([MASS_E, 0.0, 0.0]), key, msqrd=msqrd_mu_to_e_nu_nu_hel
    )[0]

    width2 = phase_space.decay_width(
        MASS_MU, np.array([MASS_E, 0.0, 0.0]), key, msqrd=msqrd_mu_to_e_nu_nu_analytic
    )[0]

    assert float(width) == pytest.approx(width2, abs=0.0, rel=0.01)


def test_width_t_to_b_w():
    def msqrd_t_to_b_w(momenta):
        pb = momenta[:, 0]
        pw = momenta[:, 1]
        pt = jnp.sum(momenta, axis=1)

        t_wfs = spinor_u(pt, MASS_T)
        b_wfs = spinor_ubar(pb, MASS_B)
        w_wfs = polvec(pw, MASS_W, True)

        v_tbw = amplitudes.VertexFFV(left=EL / (np.sqrt(2.0) * SW), right=0.0)

        def msqrd(it, ib, iw):
            amp = amplitude_ffv(v_tbw, b_wfs[ib], t_wfs[it], w_wfs[iw])
            return abs2(amp)

        idxs = np.array(np.meshgrid([0, 1], [0, 1], [0, 1, 2])).T.reshape(-1, 3)
        res = sum(msqrd(it, ib, iw) for (it, ib, iw) in idxs)

        return res / 2.0

    analytic = (
        EL**2
        * (
            MASS_B**4
            + MASS_T**4
            + MASS_T**2 * MASS_W**2
            - 2 * MASS_W**4
            + MASS_B**2 * (-2 * MASS_T**2 + MASS_W**2)
        )
        * np.sqrt(kallen_lambda(MASS_B**2, MASS_T**2, MASS_W**2))
    ) / (64 * MASS_T**3 * MASS_W**2 * np.pi * SW**2)

    key = jax.random.PRNGKey(1234)
    width = phase_space.decay_width(
        MASS_T, np.array([MASS_B, MASS_W]), key, msqrd=msqrd_t_to_b_w
    )[0]

    assert float(width) == pytest.approx(analytic, rel=1e-3, abs=0.0)


def test_width_h_to_b_b():
    def msqrd_h_to_b_b(momenta):
        p1 = momenta[:, 0]
        p2 = momenta[:, 1]
        ph = jnp.sum(momenta, axis=1)

        wf1s = spinor_ubar(p1, MASS_B)
        wf2s = spinor_v(p2, MASS_B)
        h_wf = scalar_wf(ph, False)

        v_hbb = amplitudes.VertexFFS(
            left=EL * MASS_B / (2 * MASS_W * SW), right=EL * MASS_B / (2 * MASS_W * SW)
        )

        def msqrd(i1, i2):
            return abs2(amplitude_ffs(v_hbb, wf1s[i2], wf2s[i1], h_wf))

        idxs = np.array(np.meshgrid([0, 1], [0, 1])).T.reshape(-1, 2)
        res = sum(msqrd(i1, i2) for (i1, i2) in idxs)

        return res

    mub = MASS_B / MASS_H
    analytic = (
        EL**2
        * MASS_H**3
        * mub**2
        * (1 - 4 * mub**2) ** 1.5
        / (32 * MASS_W**2 * np.pi * SW**2)
    )

    key = jax.random.PRNGKey(1234)
    width = phase_space.decay_width(
        MASS_H, np.array([MASS_B, MASS_B]), key, msqrd=msqrd_h_to_b_b
    )[0]

    assert float(width) == pytest.approx(analytic, rel=1e-3, abs=0.0)


def test_dark_matter_annihilation_scalar_mediator():
    gsxx = 1.0
    gsff = 1.0
    mx = 10.0
    mf = 1.0
    cme = 3 * mx
    mass_s = 1.0
    width_s = 1.0

    vxx = amplitudes.VertexFFS(left=gsxx, right=gsxx)
    vff = amplitudes.VertexFFS(left=gsff, right=gsff)

    ex = cme / 2.0
    px = np.sqrt(ex**2 - mx**2)
    p1 = jnp.expand_dims(jnp.array([ex, 0.0, 0.0, px]), -1)
    p2 = jnp.expand_dims(jnp.array([ex, 0.0, 0.0, -px]), -1)

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

    key = jax.random.PRNGKey(4)
    cs = phase_space.cross_section(
        cme, mx, mx, np.array([mf, mf]), key, msqrd=msqrd_x_x_to_f_f
    )[0]

    s = cme**2
    analytic = (
        gsff**2
        * gsxx**2
        * (s - 4 * mf**2)
        * np.sqrt((s - 4 * mf**2) * (s - 4 * mx**2))
    ) / (16 * np.pi * s * ((s - mass_s**2) ** 2 + (mass_s * width_s) ** 2))

    assert float(cs) == pytest.approx(analytic, rel=1e-2, abs=0.0)


def test_dark_matter_annihilation_vector_mediator():
    gvxx = 1.0
    gvff = 1.0
    mx = 10.0
    mf = 1.0
    cme = 3 * mx
    mv = 1.0
    widthv = 1.0

    vxx = amplitudes.VertexFFV(left=gvxx, right=gvxx)
    vff = amplitudes.VertexFFV(left=gvff, right=gvff)

    ex = cme / 2.0
    px = np.sqrt(ex**2 - mx**2)
    p1 = jnp.expand_dims(jnp.array([ex, 0.0, 0.0, px]), -1)
    p2 = jnp.expand_dims(jnp.array([ex, 0.0, 0.0, -px]), -1)

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

    key = jax.random.PRNGKey(4)
    cs = phase_space.cross_section(
        cme, mx, mx, np.array([mf, mf]), key, msqrd=msqrd_x_x_to_f_f
    )[0]

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
