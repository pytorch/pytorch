from .common_types import _params_t
from .optimizer import Optimizer

class SGD(Optimizer):
    def __init__(
        self,
        params: _params_t,
        lr: float,
        momentum: float = ...,
        dampening: float = ...,
        weight_decay: float = ...,
        nesterov: bool = ...,
    ) -> None: ...
