from typing import Tuple

from .optimizer import Optimizer, _params_t

class Adadelta(Optimizer):
    def __init__(self, params: _params_t, lr: float=..., rho: float=..., eps: float=..., weight_decay: float=...) -> None: ...
