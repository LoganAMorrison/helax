import pathlib
import unittest
from typing import Callable

import numpy as np
import pytest

from helax.numpy.dirac import Dirac1, DiracG0, DiracG1, DiracG2, DiracG3
from helax.numpy.wavefunctions import (DiracWf, spinor_u, spinor_ubar,
                                       spinor_v, spinor_vbar)

TEST_DATA = np.load(
    pathlib.Path(__file__).parent.joinpath("testdata").joinpath("spinor_data.npz")
)


def _sigma_momentum(momentum):
    mass = np.sqrt(
        momentum[0] ** 2 - momentum[1] ** 2 - momentum[2] ** 2 - momentum[3] ** 2
    )
    return (
        momentum[0] * DiracG0
        - momentum[1] * DiracG1
        - momentum[2] * DiracG2
        - momentum[3] * DiracG3
        + mass * Dirac1
    )


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
            expect = _sigma_momentum(self.momenta[..., i])
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
