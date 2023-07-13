from typing import Tuple

from .optimizer import Optimizer, params_t

class NAdam(Optimizer):
    def __init__(
        self,
        params: params_t,
        lr: float = ...,
        betas: Tuple[float, float] = ...,
        eps: float = ...,
        weight_decay: float = ...,
        momentum_decay: float = ...,
    ) -> None: ...
