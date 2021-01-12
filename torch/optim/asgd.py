import math
import torch
from .optimizer import Optimizer


class ASGD(Optimizer):
    """Implements Averaged Stochastic Gradient Descent.

    It has been proposed in `Acceleration of stochastic approximation by
    averaging`_.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        lambd (float, optional): decay term (default: 1e-4)
        alpha (float, optional): power for eta update (default: 0.75)
        t0 (float, optional): point at which to start averaging (default: 1e6)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    .. _Acceleration of stochastic approximation by averaging:
        https://dl.acm.org/citation.cfm?id=131098
    """

    def __init__(self, params, lr=1e-2, lambd=1e-4, alpha=0.75, t0=1e6, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, lambd=lambd, alpha=alpha, t0=t0,
                        weight_decay=weight_decay)
        super(ASGD, self).__init__(params, defaults)

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

        grads = []
        params_with_grad = []
        states = []

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    if p.grad.is_sparse:
                        raise RuntimeError('ASGD does not support sparse gradients')

                    grads.append(p.grad)
                    params_with_grad.append(p)
                    state = self.state[p]

                    # State initialization
                    if len(state) == 0:
                        state['step'] = 0
                        state['eta'] = group['lr']
                        state['mu'] = 1
                        state['ax'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    state['step'] += 1
                    states.append(state)

            if group['weight_decay'] != 0:
                torch._foreach_add_(grads, params_with_grad, alpha=group['weight_decay'])

            # decay term
            torch._foreach_mul_(params_with_grad, 1 - group['lambd'] * state['eta'])

            # update parameter
            torch._foreach_add_(params_with_grad, grads, alpha=-state['eta'])

            # averaging
            for i in range(len(states)):
                if states[i]['mu'] != 1:
                    states[i]['ax'].add_(params_with_grad[i].sub(states[i]['ax']).mul(states[i]['mu']))
                else:
                    states[i]['ax'].copy_(params_with_grad[i])

            # update eta and mu
            for state in states:
                state['eta'] = (group['lr'] /
                                math.pow((1 + group['lambd'] * group['lr'] * state['step']), group['alpha']))
                state['mu'] = 1 / max(1, state['step'] - group['t0'])

        return loss
