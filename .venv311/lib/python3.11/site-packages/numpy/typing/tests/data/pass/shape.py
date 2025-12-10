from typing import Any, NamedTuple, cast

import numpy as np


# Subtype of tuple[int, int]
class XYGrid(NamedTuple):
    x_axis: int
    y_axis: int

# Test variance of _ShapeT_co
def accepts_2d(a: np.ndarray[tuple[int, int], Any]) -> None:
    return None


accepts_2d(np.empty(XYGrid(2, 2)))
accepts_2d(np.zeros(XYGrid(2, 2), dtype=int))
accepts_2d(np.ones(XYGrid(2, 2), dtype=int))
accepts_2d(np.full(XYGrid(2, 2), fill_value=5, dtype=int))
