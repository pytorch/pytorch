from typing import Tuple, Optional
from .optimizer import _params_t, Optimizer

class LBFGS(Optimizer):
    def __init__(self, params: _params_t, lr: float=..., max_iter: int=..., max_eval: Optional[int]=..., tolerance_grad: float=..., tolerance_change: float=..., history_size: int=..., line_search_fn: Optional[str]=...) -> None: ...
