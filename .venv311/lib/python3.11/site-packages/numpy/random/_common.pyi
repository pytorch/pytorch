from collections.abc import Callable
from typing import Any, NamedTuple, TypeAlias

import numpy as np

__all__: list[str] = ["interface"]

_CDataVoidPointer: TypeAlias = Any

class interface(NamedTuple):
    state_address: int
    state: _CDataVoidPointer
    next_uint64: Callable[..., np.uint64]
    next_uint32: Callable[..., np.uint32]
    next_double: Callable[..., np.float64]
    bit_generator: _CDataVoidPointer
