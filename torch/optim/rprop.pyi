from typing import Tuple

from .optimizer import Optimizer, ParamsT

class Rprop(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: float = ...,
        etas: Tuple[float, float] = ...,
        step_sizes: Tuple[float, float] = ...,
    ) -> None: ...
