from .optimizer import Optimizer, params_t

class Adadelta(Optimizer):
    def __init__(
        self,
        params: params_t,
        lr: float = ...,
        rho: float = ...,
        eps: float = ...,
        weight_decay: float = ...,
    ) -> None: ...
