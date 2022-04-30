from typing import Union

import numpy as np
import numpy.typing as npt

Real = Union[np.uint, np.float_, np.complex_, np.int_, np.uint]
Complex = np.complex_
Numeric = Union[Real, Complex]
RealArray = npt.NDArray[Real]
ComplexArray = npt.NDArray[Real]
NumericArray = npt.NDArray[Numeric]
