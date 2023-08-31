from typing import Optional, Tuple

from .optimizer import Optimizer, params_t

class Adam(Optimizer):
    def __init__(
        self,
        params: params_t,
        lr: float = ...,
        betas: Tuple[float, float] = ...,
        eps: float = ...,
        weight_decay: float = ...,
        amsgrad: bool = ...,
        *,
        foreach: Optional[bool] = ...,
        maximize: bool = ...,
        capturable: bool = ...,
        differentiable: bool = ...,
        fused: bool = ...,
    ) -> None: ...
