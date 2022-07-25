from typing import List, Optional, Union
from torch import Tensor
from ..functional_sgd import _FunctionalSGD
from .overlapped_optim import OverlappedOptimizer
class OverlappedSGD(OverlappedOptimizer):
    def __init__(
        self,
        params: Union(None, List[Tensor]),
        lr: float = 1e-2,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        maximize: bool = False,
        foreach: bool = False,
        grad_scaler=None,
        zero_grad=False
    ):
        super.__init__(grad_scaler=grad_scaler, zero_grad=zero_grad)
        self._functional_sgd = _FunctionalSGD(params,
                                              lr,
                                              momentum,
                                              dampening,
                                              weight_decay,
                                              nesterov,
                                              maximize,
                                              foreach,
                                              _allow_empty_param_list=True)
        self.params = params
    def _step_param(self, param: Tensor, grad: Optional[Tensor]):
        return self._functional_sgd.step_param(param=param, grad=grad)

