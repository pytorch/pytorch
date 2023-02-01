from typing import Tuple, Optional
from .optimizer import _params_t, Optimizer

class Adam(Optimizer):
    def __init__(self, params: _params_t, lr: float=..., betas: Tuple[float, float]=..., eps: float=..., weight_decay: float=..., amsgrad: bool = ..., *, foreach: Optional[bool] = ..., maximize: bool = ..., capturable: bool = ..., differentiable: bool = ..., fused: bool = ...) -> None: ...
