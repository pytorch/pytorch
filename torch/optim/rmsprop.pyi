from typing import Tuple
from .optimizer import _params_t, Optimizer

class RMSprop(Optimizer):
    def __init__(self, params: _params_t, lr: float=..., alpha: float=..., eps: float=..., weight_decay: float=..., momentum: float=...,  centered: bool=...) -> None: ...
