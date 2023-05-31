from typing import Tuple

from .common_types import _params_t
from .optimizer import Optimizer

class RAdam(Optimizer):
    def __init__(
        self,
        params: _params_t,
        lr: float = ...,
        betas: Tuple[float, float] = ...,
        eps: float = ...,
        weight_decay: float = ...,
    ) -> None: ...
