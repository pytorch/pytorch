from .optimizer import Optimizer, params_t

class RMSprop(Optimizer):
    def __init__(
        self,
        params: params_t,
        lr: float = ...,
        alpha: float = ...,
        eps: float = ...,
        weight_decay: float = ...,
        momentum: float = ...,
        centered: bool = ...,
    ) -> None: ...
