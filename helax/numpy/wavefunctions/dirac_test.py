"""Tests for dirac wavefunctions."""

# pylint: disable=invalid-name

import pathlib
import unittest
from typing import Callable

import numpy as np
import pytest

from helax.numpy.dirac import (ChargeConj, ChargeConjInv, Dirac1, DiracG0,
                               DiracG1, DiracG2, DiracG3)
from helax.numpy.wavefunctions import (DiracWf, spinor_u, spinor_ubar,
                                       spinor_v, spinor_vbar)

# pylint: disable=too-many-locals

TEST_DATA = np.load(
    pathlib.Path(__file__).parent.joinpath("testdata").joinpath("spinor_data.npz")
)


def _dirac_spinor(spinor_type: str, momenta: np.ndarray, mass: float, spin: int):
    if spinor_type == "u":
        return spinor_u(momenta, mass, spin)
    if spinor_type == "ubar":
        return spinor_ubar(momenta, mass, spin)
    if spinor_type == "v":
        return spinor_v(momenta, mass, spin)
    if spinor_type == "vbar":
        return spinor_vbar(momenta, mass, spin)

    raise ValueError(f"Invalid spin type {spinor_type}.")


def _sigma_momentum(momentum):
    return (
        momentum[0] * DiracG0
        - momentum[1] * DiracG1
        - momentum[2] * DiracG2
        - momentum[3] * DiracG3
    )


def _conjugated_spinor_type(spinor_type):
    if spinor_type == "u":
        return "vbar"
    if spinor_type == "v":
        return "ubar"
    if spinor_type == "ubar":
        return "v"
    if spinor_type == "vbar":
        return "u"

    raise ValueError(f"Invalid spinor type {spinor_type}")


def _conjugate_spinor(spinor_type, spinor):
    if spinor_type in ["u", "v"]:
        return -spinor @ ChargeConjInv
    if spinor_type in ["ubar", "vbar"]:
        return ChargeConj @ spinor

    raise ValueError(f"Invalid spinor type {spinor_type}")


@pytest.mark.parametrize(
    "spinor_types",
    [
        ("u", "ubar"),
        ("v", "vbar"),
        ("u", "v"),
        ("ubar", "vbar"),
        ("vbar", "ubar"),
        ("v", "u"),
    ],
)
@pytest.mark.parametrize("mass", [0.0, 3.0])
def test_dirac_wf_completeness(spinor_types: tuple[str, str], mass: float):
    """Test the completeness relation for dirac wavefunctions.

    Recall that the completeness relations for dirac wavefunctions are:
        sum_s u(p,s) ubar(p,s) = DiracG.p + m
        sum_s v(p,s) vbar(p,s) = DiracG.p - m
    """
    spins = (-1, 1)
    if mass == 0:
        momenta = np.array(TEST_DATA["u_massless_momenta"])
    else:
        momenta = np.array(TEST_DATA["u_massive_momenta"])

    in_type, out_type = spinor_types

    flow_in: np.ndarray = np.transpose(
        np.array(
            [
                _dirac_spinor(in_type, momenta.T, mass, spin).wavefunction
                for spin in spins
            ]
        ),
        # (spin, lor, p) => (p, spin, lor)
        (2, 0, 1),
    )
    flow_out: np.ndarray = np.transpose(
        np.array(
            [
                _dirac_spinor(out_type, momenta.T, mass, spin).wavefunction
                for spin in spins
            ]
        ),
        # (spin, lor, p) => (p, spin, lor)
        (2, 0, 1),
    )

    for p, psi_in, psi_out in zip(momenta, flow_in, flow_out):
        pslash = _sigma_momentum(p)
        mmat = mass * Dirac1
        if in_type == "u" and out_type == "ubar":
            # gamma.p + m
            expected = pslash + mmat

        elif in_type == "v" and out_type == "vbar":
            # gamma.p - m
            expected = pslash - mmat

        elif in_type == "u" and out_type == "v":
            # (gamma.p + m) C^T
            expected = (pslash + mmat) @ ChargeConj.T

        elif in_type == "ubar" and out_type == "vbar":
            # C^-1 (gamma.p - m)
            expected = ChargeConjInv @ (pslash - mmat)

        elif in_type == "vbar" and out_type == "ubar":
            # C^-1 (gamma.p + m)
            expected = ChargeConjInv @ (pslash + mmat)

        elif in_type == "v" and out_type == "u":
            # (gamma.p - m) C^T
            expected = (pslash - mmat) @ ChargeConj.T

        else:
            raise ValueError(f"Invalid in,out types {in_type}, {out_type}")

        actual = np.zeros_like(expected, dtype=np.complex128)
        for incoming, outgoing in zip(psi_in, psi_out):
            actual += np.einsum("i,j->ij", incoming, outgoing)

        residual = np.max(np.abs(expected - actual))

        assert residual < 1e-5


@pytest.mark.parametrize("spinor_type", ["u", "v", "ubar", "vbar"])
@pytest.mark.parametrize("mass", [0.0, 3.0])
def test_dirac_wf_dirac_equation(spinor_type: str, mass: float):
    """Test that the spinors satisfy the dirac equation."""
    spins = (-1, 1)
    if mass == 0:
        momenta = np.array(TEST_DATA[f"{spinor_type}_massless_momenta"])
    else:
        momenta = np.array(TEST_DATA[f"{spinor_type}_massive_momenta"])

    wavefuncs: np.ndarray = np.transpose(
        np.array(
            [
                _dirac_spinor(spinor_type, momenta.T, mass, spin).wavefunction
                for spin in spins
            ]
        ),
        # (spin, lor, p) => (p, spin, lor)
        (2, 0, 1),
    )

    for p, psis in zip(momenta, wavefuncs):
        pslash = _sigma_momentum(p)
        mmat = mass * Dirac1
        for psi in psis:
            if spinor_type == "u":
                # gamma.p + m
                residual = (pslash - mmat) @ psi
            elif spinor_type == "v":
                # gamma.p - m
                residual = (pslash + mmat) @ psi
            elif spinor_type == "ubar":
                # (gamma.p + m) C^T
                residual = psi @ (pslash - mmat)
            elif spinor_type == "vbar":
                residual = psi @ (pslash + mmat)
            else:
                raise ValueError(f"Invalid spinor type {spinor_type}")

            assert np.max(np.abs(residual)) < 1e-4


@pytest.mark.parametrize("spinor_type", ["u", "v", "ubar", "vbar"])
@pytest.mark.parametrize("mass", [0.0, 3.0])
@pytest.mark.parametrize("spin", [-1, 1])
def test_dirac_wf_conjugation(spinor_type: str, mass: float, spin: int):
    """Test that the spinors satisfy the dirac equation."""
    if mass == 0:
        momenta = np.array(TEST_DATA[f"{spinor_type}_massless_momenta"])
    else:
        momenta = np.array(TEST_DATA[f"{spinor_type}_massive_momenta"])

    psi_wfs: np.ndarray = _dirac_spinor(
        spinor_type, momenta.T, mass, spin
    ).wavefunction.T
    psi_cc_wfs = np.array([_conjugate_spinor(spinor_type, psi) for psi in psi_wfs])
    conj_spinor_type = _conjugated_spinor_type(spinor_type)
    expected: np.ndarray = _dirac_spinor(
        conj_spinor_type, momenta.T, mass, spin
    ).wavefunction

    for psi_cc, exp in zip(psi_cc_wfs, expected.T):
        assert np.max(np.abs(psi_cc - exp)) < 1e-4


class TestDiracCompleteness(unittest.TestCase):
    def setUp(self) -> None:
        self.mass = 4.0
        self.momenta = np.transpose(
            np.array(
                [
                    [5.0, 0.0, 0.0, 3.0],
                    [5.0, 0.0, 0.0, -3.0],
                ]
            )
        )

    def test_completeness_spinor_u(self):
        # shapes: (num_spins, 2, num_momenta)
        wf_u = np.squeeze(
            np.array(
                [spinor_u(self.momenta, self.mass, s).wavefunction for s in (-1, 1)]
            )
        )
        wf_ubar = np.squeeze(
            np.array(
                [spinor_ubar(self.momenta, self.mass, s).wavefunction for s in (-1, 1)]
            )
        )

        for i in range(self.momenta.shape[-1]):
            # shapes: (num_spins, 4)
            u = wf_u[..., i]
            ubar = wf_ubar[..., i]

            spin_sum = np.einsum("ij,ik", u, ubar)
            expect = _sigma_momentum(self.momenta[..., i]) + self.mass * Dirac1
            self.assertLess(np.max(np.abs(spin_sum - expect)), 1e-10)


def run_spinor_tests(
    fn: Callable[[np.ndarray, float, int], DiracWf], ty: str, massive: bool
):
    assert ty in ["u", "v", "ubar", "vbar"], "Invalid string passed to test runner."
    if massive:
        prefix = ty + "_massive_"
        mass = 3.0
    else:
        prefix = ty + "_massless_"
        mass = 0.0

    momenta = TEST_DATA[prefix + "momenta"]
    spin_up: np.ndarray = TEST_DATA[prefix + "up"]
    spin_down: np.ndarray = TEST_DATA[prefix + "down"]

    helax_spin_up: np.ndarray = np.transpose(fn(momenta.T, mass, 1).wavefunction)
    helax_spin_down = np.transpose(fn(momenta.T, mass, -1).wavefunction)

    for tu, td, hu, hd in zip(spin_up, spin_down, helax_spin_up, helax_spin_down):
        for i in range(4):
            assert np.real(hu[i]) == pytest.approx(np.real(tu[i]), rel=1e-4, abs=0.0)
            assert np.real(hd[i]) == pytest.approx(np.real(td[i]), rel=1e-4, abs=0.0)
            assert np.imag(hu[i]) == pytest.approx(np.imag(tu[i]), rel=1e-4, abs=0.0)
            assert np.imag(hd[i]) == pytest.approx(np.imag(td[i]), rel=1e-4, abs=0.0)

    # Special case: pm == -pz
    if massive:
        e = 2.0
        mass = np.sqrt(3)
    else:
        e = 1.0
        mass = 0.0

    p = np.expand_dims(np.array([e, 0.0, 0.0, -1.0]), -1)
    em = np.sqrt(e - 1)
    ep = np.sqrt(e + 1)

    if ty == "u":
        spin_up = np.array([0, em, 0, ep])
        spin_down = np.array([-ep, 0, -em, 0])
    elif ty == "v":
        spin_up = np.array([ep, 0, -em, 0])
        spin_down = np.array([0, em, 0, -ep])
    elif ty == "ubar":
        spin_up = np.array([0, ep, 0, em])
        spin_down = np.array([-em, 0, -ep, 0])
    else:
        spin_up = np.array([-em, 0, ep, 0])
        spin_down = np.array([0, -ep, 0, em])

    helax_spin_up = fn(p, mass, 1).wavefunction
    helax_spin_down = fn(p, mass, -1).wavefunction

    assert np.real(helax_spin_up[0, 0]) == pytest.approx(spin_up[0])
    assert np.real(helax_spin_up[1, 0]) == pytest.approx(spin_up[1])
    assert np.real(helax_spin_up[2, 0]) == pytest.approx(spin_up[2])
    assert np.real(helax_spin_up[3, 0]) == pytest.approx(spin_up[3])

    assert np.real(helax_spin_down[0, 0]) == pytest.approx(spin_down[0])
    assert np.real(helax_spin_down[1, 0]) == pytest.approx(spin_down[1])
    assert np.real(helax_spin_down[2, 0]) == pytest.approx(spin_down[2])
    assert np.real(helax_spin_down[3, 0]) == pytest.approx(spin_down[3])


def test_spinor_u_massive():
    run_spinor_tests(spinor_u, "u", massive=True)


def test_spinor_v_massive():
    run_spinor_tests(spinor_v, "v", massive=True)


def test_spinor_ubar_massive():
    run_spinor_tests(spinor_ubar, "ubar", massive=True)


def test_spinor_vbar_massive():
    run_spinor_tests(spinor_vbar, "vbar", massive=True)


def test_spinor_u_massless():
    run_spinor_tests(spinor_u, "u", massive=False)


def test_spinor_v_massless():
    run_spinor_tests(spinor_v, "v", massive=False)


def test_spinor_ubar_massless():
    run_spinor_tests(spinor_ubar, "ubar", massive=False)


def test_spinor_vbar_massless():
    run_spinor_tests(spinor_vbar, "vbar", massive=False)
