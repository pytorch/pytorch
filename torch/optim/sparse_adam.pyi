
from typing import Tuple
from .optimizer import _params_t, Optimizer

class SparseAdam(Optimizer):
    def __init__(self, params: _params_t, lr: float=..., betas: Tuple[float, float]=..., eps: float=...) -> None: ...
