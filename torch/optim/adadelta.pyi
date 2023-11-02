from .optimizer import Optimizer, ParamsT

class Adadelta(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: float = ...,
        rho: float = ...,
        eps: float = ...,
        weight_decay: float = ...,
    ) -> None: ...
