import unittest

import numpy as np
import pytest

from helax.numpy.dirac import WeylS0, WeylS1, WeylS2, WeylS3

from .weyl import (weyl_lower_index, weyl_raise_index, weyl_x, weyl_xd, weyl_y,
                   weyl_yd)


def _sigma_momentum(momentum, bar=False):
    sgn = -1 if not bar else 1
    return (
        momentum[0] * WeylS0
        + sgn * momentum[1] * WeylS1
        + sgn * momentum[2] * WeylS2
        + sgn * momentum[3] * WeylS3
    )


class TestWeylCompleteness(unittest.TestCase):
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

    def test_completeness_weyl_x(self):
        # shapes: (num_spins, 2, num_momenta)
        wf_x = np.array(
            [weyl_x(self.momenta, self.mass, s).wavefunction for s in (-1, 1)]
        )
        wf_xd = np.array(
            [weyl_xd(self.momenta, self.mass, s).wavefunction for s in (-1, 1)]
        )

        for i in range(self.momenta.shape[-1]):
            x = wf_x[..., i]
            xd = wf_xd[..., i]
            spin_sum = np.einsum("ij,ik", x, xd)
            expect = _sigma_momentum(self.momenta[..., i])
            print()
            print(spin_sum)
            print(expect)
            print()
            # self.assertLess(np.max(np.abs(spin_sum - expect)), 1e-10)

    def test_completeness_weyl_y(self):
        # shapes: (num_spins, 2, num_momenta)
        wf_y = np.array(
            [weyl_y(self.momenta, self.mass, s).wavefunction for s in (-1, 1)]
        )
        wf_yd = np.array(
            [weyl_yd(self.momenta, self.mass, s).wavefunction for s in (-1, 1)]
        )

        for i in range(self.momenta.shape[-1]):
            y = wf_y[..., i]
            yd = wf_yd[..., i]
            spin_sum = np.einsum("ij,ik", yd, y)
            expect = _sigma_momentum(self.momenta[..., i], True)
            print()
            print(spin_sum)
            print(expect)
            print()
            # self.assertLess(np.max(np.abs(spin_sum - expect)), 1e-10)
