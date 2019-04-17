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

    def __init__(self, params, lr=1e-2, etas=(0.5, 1.2), step_sizes=(1e-6, 50)):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 < etas[0] < 1.0 < etas[1]:
            raise ValueError("Invalid eta values: {}, {}".format(etas[0], etas[1]))

        defaults = dict(lr=lr, etas=etas, step_sizes=step_sizes)
        super(Rprop, self).__init__(params, defaults)

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
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Rprop does not support sparse gradients')
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['prev'] = torch.zeros_like(p.data)
                    state['step_size'] = grad.new().resize_as_(grad).fill_(group['lr'])

                etaminus, etaplus = group['etas']
                step_size_min, step_size_max = group['step_sizes']
                step_size = state['step_size']

                state['step'] += 1

                sign = grad.mul(state['prev']).sign()
                sign[sign.gt(0)] = etaplus
                sign[sign.lt(0)] = etaminus
                sign[sign.eq(0)] = 1

                # update stepsizes with step size updates
                step_size.mul_(sign).clamp_(step_size_min, step_size_max)

                # for dir<0, dfdx=0
                # for dir>=0 dfdx=dfdx
                grad = grad.clone()
                grad[sign.eq(etaminus)] = 0

                # update parameters
                p.data.addcmul_(-1, grad.sign(), step_size)

                state['prev'].copy_(grad)

        return loss
