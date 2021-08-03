import torch
from . import _functional as F
from .optimizer import Optimizer


class Rprop(Optimizer):
    """Implements the resilient backpropagation algorithm.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        etas (Tuple[float, float], optional): pair of (etaminus, etaplis), that
            are multiplicative increase and decrease factors
            (default: (0.5, 1.2))
        step_sizes (Tuple[float, float], optional): a pair of minimal and
            maximal allowed step sizes (default: (1e-6, 50))
    """

    def __init__(self, params, lr=1e-2, etas=(0.5, 1.2), step_sizes=(1e-6, 50)):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 < etas[0] < 1.0 < etas[1]:
            raise ValueError("Invalid eta values: {}, {}".format(etas[0], etas[1]))

        defaults = dict(lr=lr, etas=etas, step_sizes=step_sizes)
        super(Rprop, self).__init__(params, defaults)

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
            params = []
            grads = []
            prevs = []
            step_sizes = []

            for p in group['params']:
                if p.grad is None:
                    continue
                params.append(p)
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Rprop does not support sparse gradients')

                grads.append(grad)
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['prev'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['step_size'] = grad.new().resize_as_(grad).fill_(group['lr'])

                prevs.append(state['prev'])
                step_sizes.append(state['step_size'])

                etaminus, etaplus = group['etas']
                step_size_min, step_size_max = group['step_sizes']

                state['step'] += 1

            F.rprop(params,
                    grads,
                    prevs,
                    step_sizes,
                    step_size_min=step_size_min,
                    step_size_max=step_size_max,
                    etaminus=etaminus,
                    etaplus=etaplus)

        return loss
