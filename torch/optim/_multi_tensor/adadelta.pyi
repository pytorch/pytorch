from typing import Tuple
from ..optimizer import _params_t, Optimizer

class Adadelta(Optimizer):
    def __init__(self, params: _params_t, lr: float=..., rho: float=..., eps: float=..., weight_decay: float=...) -> None: ...