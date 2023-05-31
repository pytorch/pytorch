from typing import Tuple

from .common_types import _params_t
from .optimizer import Optimizer

class Rprop(Optimizer):
    def __init__(
        self,
        params: _params_t,
        lr: float = ...,
        etas: Tuple[float, float] = ...,
        step_sizes: Tuple[float, float] = ...,
    ) -> None: ...
