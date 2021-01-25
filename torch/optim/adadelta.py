import torch

from . import functional as F
from .optimizer import Optimizer


class Adadelta(Optimizer):
    """Implements Adadelta algorithm.

    It has been proposed in `ADADELTA: An Adaptive Learning Rate Method`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        rho (float, optional): coefficient used for computing a running average
            of squared gradients (default: 0.9)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-6)
        lr (float, optional): coefficient that scale delta before it is applied
            to the parameters (default: 1.0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    __ https://arxiv.org/abs/1212.5701
    """

    def __init__(self, params, lr=1.0, rho=0.9, eps=1e-6, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= rho <= 1.0:
            raise ValueError("Invalid rho value: {}".format(rho))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, rho=rho, eps=eps, weight_decay=weight_decay)
        super(Adadelta, self).__init__(params, defaults)

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
            grads = []
            params_with_grad = []
            states = []
            square_avgs = []
            acc_deltas = []

            rho, eps = group['rho'], group['eps']

            for p in group['params']:
                if p.grad is not None: 
                    if p.grad.is_sparse:
                        raise RuntimeError('Adadelta does not support sparse gradients')

                    grads.append(p.grad)
                    params_with_grad.append(p)

                    state = self.state[p]

                    # State initialization
                    if len(state) == 0:
                        state['step'] = 0
                        state['square_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['acc_delta'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    square_avgs.append(state['square_avg'])
                    acc_deltas.append(state['acc_delta'])

                    state['step'] += 1
                    states.append(state)

            if group['weight_decay'] != 0:
                torch._foreach_add_(grads, params_with_grad, alpha=group['weight_decay'])

            torch._foreach_mul_(square_avgs, rho)
            torch._foreach_addcmul_(square_avgs, grads, grads, value=1 - rho)

            std = torch._foreach_add(square_avgs, eps)
            torch._foreach_sqrt_(std)

            deltas = torch._foreach_add(acc_deltas, eps)
            torch._foreach_sqrt_(deltas)
            torch._foreach_div_(deltas, std)
            torch._foreach_mul_(deltas, grads)

            torch._foreach_add_(params_with_grad, deltas, alpha=-group['lr'])

            torch._foreach_mul_(acc_deltas, rho)
            torch._foreach_addcmul_(acc_deltas, deltas, deltas, value=1 - rho)

        return loss
