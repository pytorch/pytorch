from .optimizer import Optimizer, ParamsT

class ASGD(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: float = ...,
        lambd: float = ...,
        alpha: float = ...,
        t0: float = ...,
        weight_decay: float = ...,
    ) -> None: ...
