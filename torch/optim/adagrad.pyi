from .optimizer import Optimizer, params_t

class Adagrad(Optimizer):
    def __init__(
        self,
        params: params_t,
        lr: float = ...,
        lr_decay: float = ...,
        weight_decay: float = ...,
        initial_accumulator_value: float = ...,
        eps: float = ...,
    ) -> None: ...
