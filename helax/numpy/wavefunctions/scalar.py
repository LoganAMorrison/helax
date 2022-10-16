import dataclasses

import numpy as np

from helax.numpy.typing import ComplexArray, RealArray


@dataclasses.dataclass
class ScalarWf:
    wavefunction: ComplexArray
    momentum: RealArray
    direction: int


# =============================================================================
# ---- Scalar Wavefunction ----------------------------------------------------
# =============================================================================


def scalar_wf(momentum: RealArray, out: bool) -> ScalarWf:
    """
    Compute a vector wavefunction.

    Parameters
    ----------
    momentum: ndarray
        Array containing the four-momentum of the particle.
        Must be 1 or 2 dimensional with leading dimension of size 4.
    """
    s = -1 if out else 1
    if len(momentum.shape) > 1:
        shape = (momentum.shape[-1],)
    else:
        shape = (1,)

    wf = np.ones(shape, dtype=momentum.dtype) + 0.0j
    return ScalarWf(wavefunction=wf, momentum=s * momentum, direction=s)
