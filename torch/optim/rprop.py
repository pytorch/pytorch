import torch
from .optimizer import Optimizer


class Rprop(Optimizer):
    """Implements the resilient backpropagation algorithm.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        etas (Tuple[float, float], optional): pair of (etaminus, etaplis), that
            are multiplicative increase and decrease factors
            (default: (0.5, 1.2))
        step_sizes (Tuple[float, float], optional): a pair of minimal and
            maximal allowed step sizes (default: (1e-6, 50))
    """

    def __init__(self, params, lr=1e-2, weight_decay=0, etas=(0.5, 1.2), step_sizes=(1e-6, 50)):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 < etas[0] < 1.0 < etas[1]:
            raise ValueError("Invalid eta values: {}, {}".format(etas[0], etas[1]))

        min_step, max_step = step_sizes
        defaults = dict(lr=lr, etas=etas, step_sizes=(min_step / lr, max_step / lr))
        super(Rprop, self).__init__(params, defaults)

    def reset_state(self):
        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['prev'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state['step_size'] = torch.ones_like(p, memory_format=torch.preserve_format)

    def get_update(self, par, etas=(0.5, 1.2), step_sizes=(1e-4, 5000), **_):
        grad = par.grad
        state = self.state[par]

        eta_minus, eta_plus = etas
        min_step, max_step = step_sizes
        step_size = state['step_size']
        state['step'] += 1

        sign = grad.mul(state['prev']).sign()
        sign[sign.gt(0)] = eta_plus
        sign[sign.lt(0)] = eta_minus
        sign[sign.eq(0)] = 1

        # update stepsizes with step size updates
        step_size.mul_(sign).clamp_(min_step, max_step)

        # for dir<0, dfdx=0
        # for dir>=0 dfdx=dfdx
        grad = grad.clone(memory_format=torch.preserve_format)
        grad[sign.eq(eta_minus)] = 0
        state['prev'].copy_(grad)
        return step_size * grad.sign()
