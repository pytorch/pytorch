import torch
from .optimizer import Optimizer, required


class AggMo(Optimizer):
    r"""Implements Aggregated Momentum Gradient Descent,
    as proposed in `Aggregated Momentum: Stability Through Passive Damping`_.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (list, optional): damping vector (default: [0, 0.9, 0.99])
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    Example:
        >>> optimizer = torch.optim.AggMo(model.parameters(), lr=0.1, momentum=[0,0.9,0.99,0.999])
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    .. _Aggregated Momentum: Stability Through Passive Damping:
        https://arxiv.org/abs/1804.00325
    """

    def __init__(self, params, lr=required, momentum=[0.0, 0.9, 0.99], weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        for mom in momentum:
            if not 0.0 <= momentum:
                raise ValueError("Invalid momentum value: {}".format(mom))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super(AggMo, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AggMo, self).__setstate__(state)

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
            total_mom = float(len(momentum))

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                param_state = self.state[p]
                if 'momentum_buffer' not in param_state:
                    param_state['momentum_buffer'] = {}
                    for beta in momentum:
                        param_state['momentum_buffer'][beta] = torch.zeros_like(p.data)
                for beta in momentum:
                    buf = param_state['momentum_buffer'][beta]
                    buf.mul_(beta).add_(d_p)
                    p.data.sub_(group['lr'] / total_mom, buf)
        return loss
