from pytest import approx
import pathlib

import numpy as np
import jax.numpy as jnp
from helax.wavefunctions import vector_wf

DATA = np.load(
    pathlib.Path(__file__).parent.joinpath("test_data").joinpath("vector_wf_data.npz")
)


def test_vector_wf_massless():
    momenta = jnp.array(DATA["massless_momenta"])
    spin_up: np.ndarray = DATA["massless_up"]
    spin_down: np.ndarray = DATA["massless_down"]

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
    momenta = jnp.array(DATA["massive_momenta"])
    spin_up: np.ndarray = DATA["massive_up"]
    spin_zero: np.ndarray = DATA["massive_zero"]
    spin_down: np.ndarray = DATA["massive_down"]
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
