from typing import Optional
from torch import Tensor
class OverlappedOptimizer(object):
    def __init__(self, 
                 functional_optim,
                 grad_scaler=None, 
                 zero_grad=False
                 ) -> None:
        self._functional_optim = functional_optim
        self.grad_scaler = grad_scaler
        self.zero_grad = zero_grad
        
    def step_param(self, param: Tensor, grad: Optional[Tensor]):
        self._pre_step()
        self._step_param(param=param, grad=grad)
        self._post_step()
    def reset_lr(self, lr):
        self._functional_optim.defaults['lr'] = lr
    def _pre_step(self):
        if self.grad_scaler is not None:
            self.grad_scaler.unscale_()
    def _post_step(self, param):
        if self.zero_grad:
            param.zero_grad()
    def _step_param(self, param: Tensor, grad: Optional[Tensor]):
        """The actual optimizing algorithm"""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support overlapped DDP."
        )
