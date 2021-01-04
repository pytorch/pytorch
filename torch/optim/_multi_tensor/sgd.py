import torch
from ..optimizer import Optimizer, required
from collections import defaultdict

class SGD(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + g_{t+1}, \\
                p_{t+1} & = p_{t} - \text{lr} * v_{t+1},
            \end{aligned}

        where :math:`p`, :math:`g`, :math:`v` and :math:`\mu` denote the 
        parameters, gradient, velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + \text{lr} * g_{t+1}, \\
                p_{t+1} & = p_{t} - v_{t+1}.
            \end{aligned}

        The Nesterov version is analogously modified.
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

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
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            grads = []
            params_with_grad = []
            states = []
            has_sparse_grad = False

            for p in group['params']:
                if p.grad is not None:
                    grads.append(p.grad)
                    params_with_grad.append(p)
                    states.append(self.state[p])

                    if p.grad.is_sparse:
                        has_sparse_grad = True

                        if momentum != 0: 
                            raise RuntimeError('SGD does not support momentum for sparse gradients')

            if grads == []:
                return loss

            if weight_decay != 0:
                grads = torch._foreach_add(grads, params_with_grad, alpha=weight_decay)

            if momentum != 0:
                bufs = []

                all_states_with_momentum_buffer = True
                for i in range(len(states)):
                    if 'momentum_buffer' not in states[i]:
                        all_states_with_momentum_buffer = False
                        break
                    else:
                        bufs.append(states[i]['momentum_buffer'])

                if all_states_with_momentum_buffer:
                    torch._foreach_mul_(bufs, momentum)
                    torch._foreach_add_(bufs, grads, alpha=1 - dampening)
                else:
                    bufs = []
                    for i in range(len(states)):
                        if 'momentum_buffer' not in states[i]:
                            buf = states[i]['momentum_buffer'] = torch.clone(grads[i]).detach()
                        else:
                            buf = states[i]['momentum_buffer']
                            buf.mul_(momentum).add_(grads[i], alpha=1 - dampening)

                        bufs.append(buf)

                if nesterov:
                    torch._foreach_add_(grads, bufs, alpha=momentum)
                else:
                    grads = bufs

            if not has_sparse_grad:
                torch._foreach_add_(params_with_grad, grads, alpha=-group['lr'])
            else:
                # foreach APIs dont support sparse
                for i in range(len(params_with_grad)): 
                    params_with_grad[i].add_(grads[i], alpha=-group['lr'])

        return loss

    # TODO: refactor to a base class once foreach ops are in a good shape.
    def zero_grad(self, set_to_none: bool = False):
        per_device_and_dtype_grads = defaultdict(lambda: defaultdict(list))
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        if p.grad.grad_fn is not None:
                            p.grad.detach_()
                        else:
                            p.grad.requires_grad_(False)

                        if p.grad.is_sparse:
                            p.grad.zero_()
                        else:
                            per_device_and_dtype_grads[p.grad.device][p.grad.dtype].append(p.grad)

            for _, per_dtype_grads in per_device_and_dtype_grads.items():
                for grads in per_dtype_grads.values():
                    torch._foreach_zero_(grads)
