from typing import Tuple
from .optimizer import _params_t, Optimizer

class SGDW(Optimizer):
    def __init__(self, params: _params_t, lr: float, momentum: float=..., dampening: float=..., weight_decay:float=..., nesterov:bool=...) -> None: ...


class AdamW(Optimizer):
    def __init__(self, params: _params_t, lr: float=..., betas: Tuple[float, float]=..., eps: float=..., weight_decay: float=..., amsgrad: bool = ...) -> None: ...
