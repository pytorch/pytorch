from typing import List, Optional, Union
from torch import Tensor
from ..functional_sgd import _FunctionalSGD
from .overlapped_optim import OverlappedOptimizer


class OverlappedSGD(OverlappedOptimizer):
    def __init__(
        self,
        params: Union[None, List[Tensor]],
        lr: float = 1e-2,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        maximize: bool = False,
        foreach: bool = False,
        grad_scaler=None,
        zero_grad: bool = False,
        zero_grad_to_none: bool =False
    ):
        self._functional_sgd = _FunctionalSGD([],
                                              lr,
                                              momentum,
                                              dampening,
                                              weight_decay,
                                              nesterov,
                                              maximize,
                                              foreach,
                                              _allow_empty_param_list=True)

        super().__init__(functional_optim=self._functional_sgd, 
                         grad_scaler=grad_scaler, 
                         zero_grad=zero_grad,
                         zero_grad_to_none=zero_grad_to_none)
        self.params = params

    def _step_param(self, param: Tensor, grad: Optional[Tensor]):
        return self._functional_sgd.step_param(param=param, grad=grad)
