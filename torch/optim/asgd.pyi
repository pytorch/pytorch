from typing import Tuple

from .optimizer import Optimizer, _params_t

class ASGD(Optimizer):
    def __init__(self, params: _params_t, lr: float=..., lambd: float=..., alpha: float=..., t0: float=..., weight_decay: float=...) -> None: ...
