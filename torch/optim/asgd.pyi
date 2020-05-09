from typing import Tuple
from .optimizer import _params_t, Optimizer

class ASGD(Optimizer):
    def __init__(self, params: _params_t, lr: float=..., lambd: float=..., alpha: float=..., t0: float=..., weight_decay: float=...) -> None: ...
