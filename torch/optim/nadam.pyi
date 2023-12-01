from typing import Tuple

from .optimizer import Optimizer, ParamsT

class NAdam(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: float = ...,
        betas: Tuple[float, float] = ...,
        eps: float = ...,
        weight_decay: float = ...,
        momentum_decay: float = ...,
        decoupled_weight_decay: bool = ...,
    ) -> None: ...
