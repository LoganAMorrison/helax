import pathlib

import numpy as np
import pytest
from pytest import approx

from helax.numpy.lvector import ldot
from helax.numpy.wavefunctions import vector_wf

TEST_DATA = np.load(
    pathlib.Path(__file__).parent.joinpath("testdata").joinpath("vector_wf_data.npz")
)

MetricG = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, -1.0],
    ]
)


@pytest.mark.parametrize(
    "mass,spin", [(0.0, -1), (0.0, 1), (1.0, -1.0), (1.0, 0.0), (1.0, 1.0)]
)
def test_vector_wf_transverse(mass: float, spin: int):
    """Test that polarization vectors are transverse.

    Recall that the polarization vectors satisfy:
        p.eps(p) = 0
    """
    prefix = "massless_" if mass == 0.0 else "massive_"
    momenta = np.array(TEST_DATA[prefix + "momenta"])
    helax_wfs: np.ndarray = np.transpose(
        vector_wf(momenta.T, mass, spin, False).wavefunction
    )

    for p, hwf in zip(momenta, helax_wfs):
        rhs = p[0] * hwf[0]
        lhs = p[1] * hwf[1] + p[2] * hwf[2] + p[3] * hwf[3]
        assert rhs == pytest.approx(lhs)


@pytest.mark.parametrize(
    "mass,spin", [(0.0, -1), (0.0, 1), (1.0, -1.0), (1.0, 0.0), (1.0, 1.0)]
)
def test_vector_wf_normalization(mass: float, spin: int):
    """Test the normalization of polarization vectors.

    Recall that polarization vectors are normalized to:
        sum_mu { eps(p, lam)[mu] conj(eps(p, lam)[mu]) } = -1
    """
    prefix = "massless_" if mass == 0.0 else "massive_"
    momenta = np.array(TEST_DATA[prefix + "momenta"])
    helax_wfs: np.ndarray = np.transpose(
        vector_wf(momenta.T, mass, spin, False).wavefunction
    )

    for hwf in helax_wfs:
        norm_sqr = np.abs(hwf) ** 2
        norm = norm_sqr[0] - norm_sqr[1] - norm_sqr[2] - norm_sqr[3]
        assert norm == pytest.approx(-1.0)


@pytest.mark.parametrize("mass", [0.0, 1.0])
def test_vector_wf_completeness(mass: float):
    """Test the completeness relation for polarization vectors.

    Recall that the completeness relation for polarization vectors is:
        sum(lam)(eps(p, lam)[mu] conj(eps(p, lam))) = -g[mu,nu] + p[mu]p[nu]/m^2
    """
    if mass == 0:
        spins = (-1, 1)
        momenta = np.array(TEST_DATA["massless_momenta"])
    else:
        spins = (-1, 0, 1)
        momenta = np.array(TEST_DATA["massive_momenta"])

    helax_wfs: np.ndarray = np.transpose(
        np.array(
            [vector_wf(momenta.T, mass, spin, False).wavefunction for spin in spins]
        ),
        # (spin, lor, p) => (p, spin, lor)
        (2, 0, 1),
    )

    for p, wfs in zip(momenta, helax_wfs):
        expected = -MetricG
        if mass == 0.0:
            pbar = MetricG @ p
            expected += (np.outer(p, pbar) + np.outer(pbar, p)) / ldot(p, pbar)
        else:
            expected += np.outer(p, p) / mass**2

        actual = np.zeros_like(expected, dtype=np.complex128)
        for wf in wfs:
            actual += np.outer(wf, np.conj(wf))

        residual = np.max(np.abs(expected - actual))

        assert residual < 1e-10


def test_vector_wf_massless():
    momenta = np.array(TEST_DATA["massless_momenta"])
    spin_up: np.ndarray = TEST_DATA["massless_up"]
    spin_down: np.ndarray = TEST_DATA["massless_down"]

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
    momenta = np.array(TEST_DATA["massive_momenta"])
    spin_up: np.ndarray = TEST_DATA["massive_up"]
    spin_zero: np.ndarray = TEST_DATA["massive_zero"]
    spin_down: np.ndarray = TEST_DATA["massive_down"]
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
