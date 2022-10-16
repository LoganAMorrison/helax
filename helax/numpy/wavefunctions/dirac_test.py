import unittest

import numpy as np
import pytest

from helax.numpy.dirac import Dirac1, DiracG0, DiracG1, DiracG2, DiracG3
from helax.numpy.wavefunctions import spinor_u, spinor_ubar


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
