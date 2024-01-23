from .optimizer import Optimizer, ParamsT

class SGD(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: float = ...,
        momentum: float = ...,
        dampening: float = ...,
        weight_decay: float = ...,
        nesterov: bool = ...,
    ) -> None: ...
