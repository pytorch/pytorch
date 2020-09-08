from typing import Tuple
from ..optimizer import _params_t, Optimizer

class Rprop(Optimizer):
    def __init__(self, params: _params_t, lr: float=..., etas: Tuple[float, float]=..., step_sizes: Tuple[float, float]=...) -> None: ...
