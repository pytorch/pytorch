import torch
from . import _functional as F
from ..optimizer import Optimizer

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
                        weight_decay=weight_decay, foreach=True)
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

        for group in self.param_groups:
            grads = []
            params_with_grad = []
            mus = []
            axs = []
            etas = []
            state_steps = []

            for p in group['params']:
                if p.grad is not None:
                    if p.grad.is_sparse:
                        raise RuntimeError('ASGD does not support sparse gradients')

                    grads.append(p.grad)
                    params_with_grad.append(p)
                    state = self.state[p]

                    # State initialization
                    if len(state) == 0:
                        state['step'] = torch.tensor(0.)
                        state['eta'] = torch.tensor(group['lr'])
                        state['mu'] = torch.tensor(1.)
                        state['ax'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    mus.append(state['mu'])
                    axs.append(state['ax'])
                    etas.append(state['eta'])
                    state_steps.append(state['step'])

            F.asgd(params_with_grad,
                   grads,
                   axs,
                   mus,
                   etas,
                   state_steps,
                   lambd=group['lambd'],
                   lr=group['lr'],
                   t0=group['t0'],
                   alpha=group['alpha'],
                   weight_decay=group['weight_decay'])

        return loss
