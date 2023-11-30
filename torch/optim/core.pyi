from typing import Optional, Tuple, Union

from .optimizer import Optimizer, ParamsT

class CoRe(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: float = ...,
        betas: Tuple[float, float, float, float] = ...,
        eps: float = ...,
        etas: Tuple[float, float] = ...,
        step_sizes: Tuple[float, float] = ...,
        weight_decay: Union[float, list] = ...,
        score_history: int = ...,
        frozen: Union[int, list] = ...,
        *,
        foreach: Optional[bool] = ...,
        maximize: bool = ...,
        differentiable: bool = ...,
    ) -> None: ...
