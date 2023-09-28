from typing import Tuple

from .optimizer import Optimizer, params_t

class Rprop(Optimizer):
    def __init__(
        self,
        params: params_t,
        lr: float = ...,
        etas: Tuple[float, float] = ...,
        step_sizes: Tuple[float, float] = ...,
    ) -> None: ...
