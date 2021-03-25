import torch
from .mlc_optimizer import MLCOptimizer
from ..optim.sgd import SGD
from ..optim.optimizer import required  # type: ignore[attr-defined]


class MLCSGD(MLCOptimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    This optimizer mixes optimization methods where parameters are partitioned
    into two sets; one set which can be accelerated on MLCompute, and another
    set which are handled by PyTorch's optimizer (torch.optim.SGD).

    The limitation here is that parameters should not cross between these two
    sets as momentum information will be lost. This should usually not be a
    problem as this will only occur if the same parameter is used in two places
    with vastly different contexts.

    Args:
        params (iterable): iterable of parameters to optimize
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
        max_gradient_clipping(float, optional):  maximum gradient to be clipped(default: inf)
        min_gradient_clipping(float, optional):  minimum gradient to be clipped(default: -inf)

    Example:
        >>> optimizer = torch.mlc.MLCSGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, max_gradient_clipping=float("inf"),
                 min_gradient_clipping=float("-inf")):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        use_gradient_clipping = True if (max_gradient_clipping is not float("inf") or
                                         min_gradient_clipping is not float("-inf")) else False

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,
                        use_gradient_clipping=use_gradient_clipping,
                        max_gradient_clipping=max_gradient_clipping,
                        min_gradient_clipping=min_gradient_clipping)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        params = list(params)
        super(MLCSGD, self).__init__(params, defaults)
        self.mlcopt = torch._C.MLCOptimizerSGD(lr, momentum, dampening, weight_decay, nesterov, use_gradient_clipping,
                                               max_gradient_clipping, min_gradient_clipping)
        self.torchopt = SGD(params, lr, momentum, dampening, weight_decay, nesterov)

    def __setstate__(self, state):
        super(MLCSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def _same_parameters(self, group):
        keys = ['weight_decay', 'momentum', 'dampening', 'nesterov']
        return all(self.defaults[k] == group[k] for k in keys)
