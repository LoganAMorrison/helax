from typing import Callable
from pytest import approx
import pathlib

import numpy as np
import jax.numpy as jnp
from helax.wavefunctions import spinor_u, spinor_v, spinor_ubar, spinor_vbar, DiracWf

DATA = np.load(
    pathlib.Path(__file__).parent.joinpath("test_data").joinpath("spinor_data.npz")
)


def run_spinor_tests(
    fn: Callable[[jnp.ndarray, float, int], DiracWf], ty: str, massive: bool
):
    assert ty in ["u", "v", "ubar", "vbar"], "Invalid string passed to test runner."
    if massive:
        prefix = ty + "_massive_"
        mass = 3.0
    else:
        prefix = ty + "_massless_"
        mass = 0.0

    momenta = jnp.array(DATA[prefix + "momenta"])
    spin_up: np.ndarray = DATA[prefix + "up"]
    spin_down: np.ndarray = DATA[prefix + "down"]

    helax_spin_up: np.ndarray = np.transpose(fn(momenta.T, mass, 1).wavefunction)
    helax_spin_down = np.transpose(fn(momenta.T, mass, -1).wavefunction)

    for tu, td, hu, hd in zip(spin_up, spin_down, helax_spin_up, helax_spin_down):
        for i in range(4):
            assert np.real(hu[i]) == approx(np.real(tu[i]), rel=1e-4, abs=0.0)
            assert np.real(hd[i]) == approx(np.real(td[i]), rel=1e-4, abs=0.0)
            assert np.imag(hu[i]) == approx(np.imag(tu[i]), rel=1e-4, abs=0.0)
            assert np.imag(hd[i]) == approx(np.imag(td[i]), rel=1e-4, abs=0.0)


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
