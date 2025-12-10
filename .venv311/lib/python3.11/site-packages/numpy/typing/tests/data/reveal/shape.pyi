from typing import Any, NamedTuple, assert_type

import numpy as np

# Subtype of tuple[int, int]
class XYGrid(NamedTuple):
    x_axis: int
    y_axis: int

arr: np.ndarray[XYGrid, Any]

# Test shape property matches shape typevar
assert_type(arr.shape, XYGrid)
