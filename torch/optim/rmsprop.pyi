from .optimizer import Optimizer, ParamsT

class RMSprop(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: float = ...,
        alpha: float = ...,
        eps: float = ...,
        weight_decay: float = ...,
        momentum: float = ...,
        centered: bool = ...,
    ) -> None: ...
