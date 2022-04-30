import pathlib
from typing import Callable

import numpy as np
from pytest import approx

from helax.numpy.wavefunctions import (DiracWf, spinor_u, spinor_ubar,
                                       spinor_v, spinor_vbar, vector_wf)

DWF_DATA = np.load(
    pathlib.Path(__file__).parent.joinpath("test_data").joinpath("spinor_data.npz")
)
VWF_DATA = np.load(
    pathlib.Path(__file__).parent.joinpath("test_data").joinpath("vector_wf_data.npz")
)


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

    momenta = DWF_DATA[prefix + "momenta"]
    spin_up: np.ndarray = DWF_DATA[prefix + "up"]
    spin_down: np.ndarray = DWF_DATA[prefix + "down"]

    helax_spin_up: np.ndarray = np.transpose(fn(momenta.T, mass, 1).wavefunction)
    helax_spin_down = np.transpose(fn(momenta.T, mass, -1).wavefunction)

    for tu, td, hu, hd in zip(spin_up, spin_down, helax_spin_up, helax_spin_down):
        for i in range(4):
            assert np.real(hu[i]) == approx(np.real(tu[i]), rel=1e-4, abs=0.0)
            assert np.real(hd[i]) == approx(np.real(td[i]), rel=1e-4, abs=0.0)
            assert np.imag(hu[i]) == approx(np.imag(tu[i]), rel=1e-4, abs=0.0)
            assert np.imag(hd[i]) == approx(np.imag(td[i]), rel=1e-4, abs=0.0)

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

    assert float(helax_spin_up[0, 0]) == approx(spin_up[0])
    assert float(helax_spin_up[1, 0]) == approx(spin_up[1])
    assert float(helax_spin_up[2, 0]) == approx(spin_up[2])
    assert float(helax_spin_up[3, 0]) == approx(spin_up[3])

    assert float(helax_spin_down[0, 0]) == approx(spin_down[0])
    assert float(helax_spin_down[1, 0]) == approx(spin_down[1])
    assert float(helax_spin_down[2, 0]) == approx(spin_down[2])
    assert float(helax_spin_down[3, 0]) == approx(spin_down[3])


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


def test_vector_wf_massless():
    momenta = np.array(VWF_DATA["massless_momenta"])
    spin_up: np.ndarray = VWF_DATA["massless_up"]
    spin_down: np.ndarray = VWF_DATA["massless_down"]

    helax_spin_up: np.ndarray = np.transpose(
        vector_wf(momenta.T, 0.0, 1, False).wavefunction
    )
    helax_spin_down = np.transpose(vector_wf(momenta.T, 0.0, -1, False).wavefunction)

    for t, h in zip(spin_up, helax_spin_up):
        for i in range(4):
            assert np.real(h[i]) == approx(np.real(t[i]), rel=1e-4, abs=0.0)
            assert np.imag(h[i]) == approx(np.imag(t[i]), rel=1e-4, abs=0.0)

    for t, h in zip(spin_down, helax_spin_down):
        for i in range(4):
            assert np.real(h[i]) == approx(np.real(t[i]), rel=1e-4, abs=0.0)
            assert np.imag(h[i]) == approx(np.imag(t[i]), rel=1e-4, abs=0.0)


def test_vector_wf_massive():
    momenta = np.array(VWF_DATA["massive_momenta"])
    spin_up: np.ndarray = VWF_DATA["massive_up"]
    spin_zero: np.ndarray = VWF_DATA["massive_zero"]
    spin_down: np.ndarray = VWF_DATA["massive_down"]
    mass = 1.0

    helax_spin_up: np.ndarray = np.transpose(
        vector_wf(momenta.T, mass, 1, False).wavefunction
    )
    helax_spin_zero: np.ndarray = np.transpose(
        vector_wf(momenta.T, mass, 0, False).wavefunction
    )
    helax_spin_down = np.transpose(vector_wf(momenta.T, mass, -1, False).wavefunction)

    for t, h in zip(spin_up, helax_spin_up):
        for i in range(4):
            assert np.real(h[i]) == approx(np.real(t[i]), rel=1e-4, abs=0.0)
            assert np.imag(h[i]) == approx(np.imag(t[i]), rel=1e-4, abs=0.0)

    for t, h in zip(spin_zero, helax_spin_zero):
        for i in range(4):
            assert np.real(h[i]) == approx(np.real(t[i]), rel=1e-4, abs=0.0)
            assert np.imag(h[i]) == approx(np.imag(t[i]), rel=1e-4, abs=0.0)

    for t, h in zip(spin_down, helax_spin_down):
        for i in range(4):
            assert np.real(h[i]) == approx(np.real(t[i]), rel=1e-4, abs=0.0)
            assert np.imag(h[i]) == approx(np.imag(t[i]), rel=1e-4, abs=0.0)
