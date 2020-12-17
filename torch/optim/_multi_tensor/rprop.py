import torch
from ..optimizer import Optimizer


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

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        grads = []
        states = []
        params_with_grad = []
        step_sizes = []

        for group in self.param_groups:
            for p in group['params']:
                etaminus, etaplus = group['etas']
                step_size_min, step_size_max = group['step_sizes']

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

            signs = torch._foreach_mul(grads, [s['prev'] for s in states])
            signs = [s.sign() for s in signs]
            for sign in signs:
                sign[sign.gt(0)] = etaplus
                sign[sign.lt(0)] = etaminus
                sign[sign.eq(0)] = 1

            # update stepsizes with step size updates
            torch._foreach_mul_(step_sizes, signs)
            for step_size in step_sizes:
                step_size.clamp_(step_size_min, step_size_max)

            # for dir<0, dfdx=0
            # for dir>=0 dfdx=dfdx
            for i in range(len(grads)): 
                grads[i] = grads[i].clone(memory_format=torch.preserve_format)
                grads[i][signs[i].eq(etaminus)] = 0

            # update parameters
            grad_signs = [grad.sign() for grad in grads]
            torch._foreach_addcmul_(params_with_grad, grad_signs, step_sizes, value=-1)

            for i in range(len(states)):
                states[i]['prev'].copy_(grads[i])

        return loss
