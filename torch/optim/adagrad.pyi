from .optimizer import Optimizer, ParamsT

class Adagrad(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: float = ...,
        lr_decay: float = ...,
        weight_decay: float = ...,
        initial_accumulator_value: float = ...,
        eps: float = ...,
    ) -> None: ...
