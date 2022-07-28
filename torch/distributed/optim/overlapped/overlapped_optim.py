from typing import Optional
from torch import Tensor
class OverlappedOptimizer(object):
    def __init__(self, 
                 functional_optim,
                 grad_scaler=None, 
                 zero_grad=False,
                 zero_grad_to_none=False
                 ) -> None:

        if not zero_grad and zero_grad_to_none:
            raise ValueError('zero_grad_to_none can only be set to True while zero_grad is True.')

        self._functional_optim = functional_optim
        self.grad_scaler = grad_scaler
        self.zero_grad = zero_grad
        self.zero_grad_to_none = zero_grad_to_none

        # Dummpy param_groups to cooperate with LRScheduler
        self.param_groups = [{'lr': functional_optim.defaults['lr']}]

        self.is_overlapped = True
        
    def step_param(self, param: Tensor, grad: Optional[Tensor]):
        self._pre_step(param=param, grad=grad)
        self._step_param(param=param, grad=grad)
        self._post_step(param=param, grad=grad)

    def set_lr(self, lr):
        self._functional_optim.defaults['lr'] = lr
        self.param_groups[0]['lr'] = lr

    def get_lr(self):
        return self.param_groups[0]['lr']

    def _pre_step(self, param, grad):
        if self.grad_scaler is not None:
            self.grad_scaler.unscale_(param)

    def _post_step(self, param, grad):
        if self.zero_grad:
            if param.grad is not None:
                if self.zero_grad_to_none:
                    param.grad = None
                else:
                    if param.grad.grad_fn is not None:
                        param.grad.detach_()
                    else:
                        param.grad.requires_grad_(False)

                    param.grad.zero_()
            
    def _step_param(self, param: Tensor, grad: Optional[Tensor]):
        """The actual optimizing algorithm"""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support overlapped DDP."
        )
