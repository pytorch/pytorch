from .common_types import _params_t
from .optimizer import Optimizer

class ASGD(Optimizer):
    def __init__(
        self,
        params: _params_t,
        lr: float = ...,
        lambd: float = ...,
        alpha: float = ...,
        t0: float = ...,
        weight_decay: float = ...,
    ) -> None: ...
