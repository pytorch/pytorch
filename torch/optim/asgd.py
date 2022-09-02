import math
import torch
from torch import Tensor

from .optimizer import Optimizer
from typing import List, Optional

__all__ = ['ASGD', 'asgd']

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
        foreach (bool, optional): whether foreach implementation of optimizer
            is used (default: None)
        maximize (bool, optional): maximize the params based on the objective, instead of
            minimizing (default: False)

    .. _Acceleration of stochastic approximation by averaging:
        https://dl.acm.org/citation.cfm?id=131098
    """

    def __init__(self, params, lr=1e-2, lambd=1e-4, alpha=0.75, t0=1e6, weight_decay=0,
                 foreach: Optional[bool] = None, maximize: bool = False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, lambd=lambd, alpha=alpha, t0=t0,
                        weight_decay=weight_decay, foreach=foreach, maximize=maximize)
        super(ASGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('foreach', None)
            group.setdefault('maximize', False)
        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(state_values[0]['step'])
        if not step_is_tensor:
            for s in state_values:
                s['step'] = torch.tensor(float(s['step']))
        eta_is_tensor = (len(state_values) != 0) and torch.is_tensor(state_values[0]['eta'])
        if not eta_is_tensor:
            for s in state_values:
                s['eta'] = torch.tensor(s['eta'])
        mu_is_tensor = (len(state_values) != 0) and torch.is_tensor(state_values[0]['mu'])
        if not mu_is_tensor:
            for s in state_values:
                s['mu'] = torch.tensor(float(s['mu']))

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            mus = []
            axs = []
            etas = []
            state_steps = []

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('ASGD does not support sparse gradients')
                    grads.append(p.grad)

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

            asgd(params_with_grad,
                 grads,
                 axs,
                 mus,
                 etas,
                 state_steps,
                 lambd=group['lambd'],
                 lr=group['lr'],
                 t0=group['t0'],
                 alpha=group['alpha'],
                 weight_decay=group['weight_decay'],
                 foreach=group['foreach'],
                 maximize=group['maximize'])

        return loss


def asgd(params: List[Tensor],
         grads: List[Tensor],
         axs: List[Tensor],
         mus: List[Tensor],
         etas: List[Tensor],
         state_steps: List[Tensor],
         # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
         # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
         foreach: bool = None,
         maximize: bool = False,
         *,
         lambd: float,
         lr: float,
         t0: float,
         alpha: float,
         weight_decay: float):
    r"""Functional API that performs asgd algorithm computation.

    See :class:`~torch.optim.ASGD` for details.
    """

    if foreach is None:
        # Placeholder for more complex foreach logic to be added when value is not set
        foreach = False

    if foreach and torch.jit.is_scripting():
        raise RuntimeError('torch.jit.script not supported with foreach optimizers')

    if foreach and not torch.jit.is_scripting():
        func = _multi_tensor_asgd
    else:
        func = _single_tensor_asgd

    func(params,
         grads,
         axs,
         mus,
         etas,
         state_steps,
         lambd=lambd,
         lr=lr,
         t0=t0,
         alpha=alpha,
         weight_decay=weight_decay,
         maximize=maximize)


def _single_tensor_asgd(params: List[Tensor],
                        grads: List[Tensor],
                        axs: List[Tensor],
                        mus: List[Tensor],
                        etas: List[Tensor],
                        state_steps: List[Tensor],
                        *,
                        lambd: float,
                        lr: float,
                        t0: float,
                        alpha: float,
                        weight_decay: float,
                        maximize: bool):

    for i, param in enumerate(params):
        grad = grads[i]
        grad = grad if not maximize else -grad
        mu = mus[i]
        ax = axs[i]
        eta = etas[i]
        step_t = state_steps[i]

        if torch.is_complex(param):
            grad = torch.view_as_real(grad)
            param = torch.view_as_real(param)
            ax = torch.view_as_real(ax)

        # update step
        step_t += 1
        step = step_t.item()

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        # decay term
        param.mul_(1 - lambd * eta.item())

        # update parameter
        param.add_(grad, alpha=-eta.item())

        # averaging
        if mu.item() != 1:
            ax.add_(param.sub(ax).mul(mu))
        else:
            ax.copy_(param)

        new_eta = torch.tensor(lr / math.pow((1 + lambd * lr * step), alpha))
        eta.copy_(new_eta)
        new_mu = torch.tensor(1 / max(1, step - t0))
        mu.copy_(new_mu)


def _multi_tensor_asgd(params: List[Tensor],
                       grads: List[Tensor],
                       axs: List[Tensor],
                       mus: List[Tensor],
                       etas: List[Tensor],
                       state_steps: List[Tensor],
                       *,
                       lambd: float,
                       lr: float,
                       t0: float,
                       alpha: float,
                       weight_decay: float,
                       maximize: bool):

    if len(params) == 0:
        return

    if maximize:
        grads = torch._foreach_neg(grads)

    def _view_complex_as_real(tensor_list):
        return [torch.view_as_real(t) if torch.is_complex(t) else t for t in tensor_list]

    grads = _view_complex_as_real(grads)
    params = _view_complex_as_real(params)
    axs = _view_complex_as_real(axs)

    # update step
    torch._foreach_add_(state_steps, 1)

    if weight_decay != 0:
        grads = torch._foreach_add(grads, params, alpha=weight_decay)

    # decay term
    eta = etas[0].item()
    torch._foreach_mul_(params, 1 - lambd * eta)

    # update parameter
    torch._foreach_add_(params, grads, alpha=-eta)

    # averaging
    for i in range(len(axs)):
        if mus[i].item() != 1:
            axs[i].add_(params[i].sub(axs[i]).mul(mus[i]))
        else:
            axs[i].copy_(params[i])

    # update eta and mu
    for i in range(len(mus)):
        new_eta = torch.tensor(lr / math.pow((1 + lambd * lr * state_steps[i].item()), alpha))
        etas[i].copy_(new_eta)
        new_mu = torch.tensor(1 / max(1, state_steps[i].item() - t0))
        mus[i].copy_(new_mu)
