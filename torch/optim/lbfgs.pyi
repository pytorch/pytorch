from typing import Optional

from .optimizer import Optimizer, params_t

class LBFGS(Optimizer):
    def __init__(
        self,
        params: params_t,
        lr: float = ...,
        max_iter: int = ...,
        max_eval: Optional[int] = ...,
        tolerance_grad: float = ...,
        tolerance_change: float = ...,
        history_size: int = ...,
        line_search_fn: Optional[str] = ...,
    ) -> None: ...
