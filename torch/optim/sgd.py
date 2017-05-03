import torch
import math
from .optimizer import Optimizer, required


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
                  v = \rho * v + g \\
                  p = p - lr * v

        where p, g, v and :math:`\rho` denote the parameters, gradient, velocity, and
        momentum respectively.

        This is in constrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
             v = \rho * v + lr * g \\
             p = p - v

        The Nesterov version is analogously modified.
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    if d_p.is_sparse:
                        d_p = d_p.coalesce()
                        state = self.sparse_weight_decay_state(p)
                        state['step'] += 1
                        b = 1 - group['lr'] * weight_decay
                        p_s = p.data._sparse_mask(d_p)
                        d_s = (state['step'] - state['last_update']._sparse_mask(d_p.int())).type_as(p.data)
                        p_s.mul_(torch.exp(math.log(b) * d_s))
                        p.data.index_copy_(0, d_p._indices(), p_s)
                        state['last_update'].index_fill_(0, indices, state['step'])
                    else:
                        d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = d_p.clone()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-group['lr'], d_p)

        return loss

    def sparse_weight_decay_state(self, p):
        state = self.state[p]
        if 'sparse_weight_decay' not in state:
            state['sparse_weight_decay'] = {
                    'step': 0,
                    'last_update': torch.IntTensor(p.data.size()).zero_()
                    }
        return state['sparse_weight_decay']

    def flush(self):
        """Flush any pending updates to the parameters.  You should
        call this at the end of optimization over sparse tensors.
        """
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            for p in group['params']:
                state = self.sparse_weight_decay_state(p)
                b = 1 - group['lr'] * weight_decay
                d = (state['step'] - state['last_update']).type_as(p.data)
                # p *= b ** d
                p.data.mul_(torch.exp(math.log(b) * d))
                state['last_update'].fill_(state['step'])

"""
state is used during optimization over sparse vectors.  It records:
    step: number of steps we've carried out so far
    last_update: the last time we carried out a sparse update
"""
