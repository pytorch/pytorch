from .common_types import _params_t
from .optimizer import Optimizer

class Adadelta(Optimizer):
    def __init__(
        self,
        params: _params_t,
        lr: float = ...,
        rho: float = ...,
        eps: float = ...,
        weight_decay: float = ...,
    ) -> None: ...
