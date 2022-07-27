from typing import List, Dict, Optional, Tuple
from torch import Tensor
from ..functional_adamw import _FunctionalAdamW
from .overlapped_optim import OverlappedOptimizer
class OverlappedAdamW(OverlappedOptimizer):
    def __init__(
        self,
        params: List[Tensor],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
        maximize: bool = False,
        foreach: bool = False,
        grad_scaler=None,
        zero_grad=False
    ):
        super().__init__(grad_scaler=grad_scaler, zero_grad=zero_grad)
        self._functional_adamw = _FunctionalAdamW(params,
                                                  lr,
                                                  betas,
                                                  eps,
                                                  weight_decay,
                                                  amsgrad,
                                                  maximize,
                                                  foreach,
                                                  _allow_empty_param_list=True)
    def _step_param(self, param: Tensor, grad: Optional[Tensor]):
        return self._functional_adamw.step_param(param=param, grad=grad)
