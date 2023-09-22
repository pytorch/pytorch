from typing import Tuple

from .optimizer import Optimizer, params_t

class Adamax(Optimizer):
    def __init__(
        self,
        params: params_t,
        lr: float = ...,
        betas: Tuple[float, float] = ...,
        eps: float = ...,
        weight_decay: float = ...,
    ) -> None: ...
