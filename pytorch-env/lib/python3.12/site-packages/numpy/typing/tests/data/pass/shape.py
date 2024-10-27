from typing import Any, NamedTuple

import numpy as np
from typing_extensions import assert_type


# Subtype of tuple[int, int]
class XYGrid(NamedTuple):
    x_axis: int
    y_axis: int

arr: np.ndarray[XYGrid, Any] = np.empty(XYGrid(2, 2))

# Test variance of _ShapeType_co
def accepts_2d(a: np.ndarray[tuple[int, int], Any]) -> None:
    return None

accepts_2d(arr)
