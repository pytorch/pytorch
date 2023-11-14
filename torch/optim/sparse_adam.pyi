from typing import Tuple

from .optimizer import Optimizer, ParamsT

class SparseAdam(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: float = ...,
        betas: Tuple[float, float] = ...,
        eps: float = ...,
    ) -> None: ...
