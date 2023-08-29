from typing import Tuple

from .optimizer import Optimizer, params_t

class RAdam(Optimizer):
    def __init__(
        self,
        params: params_t,
        lr: float = ...,
        betas: Tuple[float, float] = ...,
        eps: float = ...,
        weight_decay: float = ...,
        decoupled_weight_decay: bool = ...,
    ) -> None: ...
