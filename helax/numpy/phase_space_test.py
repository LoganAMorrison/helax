"""Tests for the `phase_space` module."""

import unittest

import numpy as np

from helax.numpy.lvector import lnorm
from helax.numpy.phase_space import PhaseSpace


class TestMomenta(unittest.TestCase):
    """Testing properties of the generated momenta."""

    def setUp(self):
        self.masses = [1, 2, 3, 4]
        self.cme = 3 * sum(self.masses)
        self.phase_space = PhaseSpace(cme=self.cme, masses=self.masses)
        self.seed = 0
        self.nevents = 100
        self.momenta, self.weights = self.phase_space.generate(
            self.nevents, seed=self.seed
        )

    def test_momenta_sum(self):
        """Test that the generated momenta sum to correct values."""
        # Momenta have shape (4, nfsp, nevents). Summing over axis=1 gives the
        # sum of all final-state particle momenta.
        momentum_sum = np.sum(self.momenta, axis=1)

        # Energies should be equal to center-of-mass energy
        self.assertTrue(np.allclose(momentum_sum[0], self.cme))
        # 3-momenta should be zero
        self.assertTrue(np.allclose(momentum_sum[1], 0.0))
        self.assertTrue(np.allclose(momentum_sum[2], 0.0))
        self.assertTrue(np.allclose(momentum_sum[3], 0.0))

    def test_masses(self):
        """Test that the generated momenta have correct mass."""
        shape = self.momenta.shape

        for i in range(shape[1]):
            momentum = self.momenta[:, i]
            norm = lnorm(momentum)
            self.assertTrue(np.allclose(norm, self.masses[i]))
