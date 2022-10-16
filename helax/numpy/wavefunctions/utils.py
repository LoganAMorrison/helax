from helax.numpy.typing import RealArray


def check_momentum_shape(momentum: RealArray, require_batch: bool = True):
    shape = momentum.shape
    assert shape[0] == 4, "Leading dimension must have length 4."
    if require_batch:
        assert len(shape) == 2, "Expected a batch dimension. Found 1-D array."
