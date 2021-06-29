import torch
from . import _functional as F
from ..optimizer import Optimizer
from collections import defaultdict

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
            grads = []
            states = []
            params_with_grad = []
            step_sizes = []
            etaminus, etaplus = group['etas']
            step_size_min, step_size_max = group['step_sizes']

            for p in group['params']:
                if p.grad is not None:
                    if p.grad.is_sparse:
                        raise RuntimeError('RMSprop does not support sparse gradients')

                    grads.append(p.grad)
                    params_with_grad.append(p)

                    state = self.state[p]
                    # State initialization
                    if len(state) == 0:
                        state['step'] = 0
                        state['prev'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['step_size'] = p.grad.new().resize_as_(p.grad).fill_(group['lr'])

                        state['step'] += 1

                    states.append(state)
                    step_sizes.append(state['step_size'])

            F.rprop(params_with_grad,
                    grads,
                    states,
                    step_sizes,
                    step_size_max=step_size_max,
                    step_size_min=step_size_min,
                    etaminus=etaminus,
                    etaplus=etaplus)

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
