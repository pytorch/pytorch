from .optimizer import Optimizer, params_t

class ASGD(Optimizer):
    def __init__(
        self,
        params: params_t,
        lr: float = ...,
        lambd: float = ...,
        alpha: float = ...,
        t0: float = ...,
        weight_decay: float = ...,
    ) -> None: ...
