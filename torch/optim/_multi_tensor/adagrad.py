import torch
from .. import _functional as F
from ..optimizer import Optimizer

class Adagrad(Optimizer):
    """Implements Adagrad algorithm with multi-tensor APIs.

    It has been proposed in `Adaptive Subgradient Methods for Online Learning
    and Stochastic Optimization`_.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        lr_decay (float, optional): learning rate decay (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-10)

    .. _Adaptive Subgradient Methods for Online Learning and Stochastic
        Optimization: http://jmlr.org/papers/v12/duchi11a.html
    """

    def __init__(self, params, lr=1e-2, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= lr_decay:
            raise ValueError("Invalid lr_decay value: {}".format(lr_decay))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= initial_accumulator_value:
            raise ValueError("Invalid initial_accumulator_value value: {}".format(initial_accumulator_value))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))

        defaults = dict(lr=lr, lr_decay=lr_decay, eps=eps, weight_decay=weight_decay,
                        initial_accumulator_value=initial_accumulator_value)
        super(Adagrad, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['sum'] = torch.full_like(p, initial_accumulator_value, memory_format=torch.preserve_format)

    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['sum'].share_memory_()

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            state_sums = []
            state_steps = []

            has_sparse_grad = False
            for p in group['params']:
                if p.grad is not None:
                    if p.grad.is_sparse:
                        has_sparse_grad = True
                    params_with_grad.append(p)
                    grads.append(p.grad)
                    state = self.state[p]
                    state_sums.append(state['sum'])
                    # update the steps for each param group update
                    state['step'] += 1
                    # record the step after step update
                    state_steps.append(state['step'])

            weight_decay = group['weight_decay']
            lr = group['lr']
            lr_decay = group['lr_decay']

            if weight_decay != 0:
                if has_sparse_grad:
                    raise RuntimeError("weight_decay option is not compatible with sparse gradients")
                torch._foreach_add_(grads, params_with_grad, alpha=weight_decay)

            minus_clr = [-lr / (1 + (step - 1) * lr_decay) for step in state_steps]

            if has_sparse_grad:
                # sparse is not supported by multi_tensor. Fall back to optim.adagrad
                # implementation.
                F.adagrad(params_with_grad,
                          grads,
                          state_sums,
                          state_steps,
                          lr=lr,
                          weight_decay=weight_decay,
                          lr_decay=lr_decay,
                          eps=group['eps'])
            else:
                torch._foreach_addcmul_(state_sums, grads, grads, value=1)
                std = torch._foreach_add(torch._foreach_sqrt(state_sums), group['eps'])
                torch._foreach_addcdiv_(
                    params_with_grad, torch._foreach_mul(grads, minus_clr), std
                )

        return loss
