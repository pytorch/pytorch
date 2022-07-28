from typing import List, Tuple, Optional
from torch import Tensor
from ..functional_adam import _FunctionalAdam
from .overlapped_optim import OverlappedOptimizer
class OverlappedAdam(OverlappedOptimizer):
    def __init__(
        self,
        params: List[Tensor],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
        maximize: bool = False,
        foreach: bool = False,
        grad_scaler=None,
        zero_grad: bool = False,
        zero_grad_to_none: bool = False
    ):
        
        self._functional_adam = _FunctionalAdam([],
                                                lr,
                                                betas,
                                                eps,
                                                weight_decay,
                                                amsgrad,
                                                maximize,
                                                foreach,
                                                _allow_empty_param_list=True)

        super().__init__(functional_optim=self._functional_adam, 
                         grad_scaler=grad_scaler, 
                         zero_grad=zero_grad,
                         zero_grad_to_none=zero_grad_to_none)
        self.params = params

    def _step_param(self, param: Tensor, grad: Optional[Tensor]):
        return self._functional_adam.step_param(param=param, grad=grad)
        