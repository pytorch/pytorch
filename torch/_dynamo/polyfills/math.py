"""
Python polyfills for math
"""

import math
from typing import SupportsFloat, SupportsIndex, Union

from ..decorators import substitute_in_graph


@substitute_in_graph(math.radians)
def radians(x: Union[SupportsFloat, SupportsIndex], /) -> float:
    return (math.pi / 180.0) * float(x)
