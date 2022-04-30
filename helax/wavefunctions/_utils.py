from typing import Union
import chex

NdArray = Union[chex.Array, chex.ArrayNumpy]


def dispatch(unvec_fn, vec_fn, momentum: NdArray, *args):
    assert momentum.shape[0] == 4, "First dimension of `momentum` must have size 4."

    if len(momentum.shape) == 1:
        return unvec_fn(momentum, *args)
    elif len(momentum.shape) > 1:
        return vec_fn(momentum, *args)
    else:
        raise ValueError(
            "Invalid shape for `momentum`. `momentum` must be 1 or 2 dimensional."
        )
