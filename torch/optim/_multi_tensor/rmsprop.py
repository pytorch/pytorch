import torch
from ..optimizer import Optimizer


class RMSprop(Optimizer):
    r"""Implements RMSprop algorithm.

    Proposed by G. Hinton in his
    `course <https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf>`_.

    The centered version first appears in `Generating Sequences
    With Recurrent Neural Networks <https://arxiv.org/pdf/1308.0850v5.pdf>`_.

    The implementation here takes the square root of the gradient average before
    adding epsilon (note that TensorFlow interchanges these two operations). The effective
    learning rate is thus :math:`\alpha/(\sqrt{v} + \epsilon)` where :math:`\alpha`
    is the scheduled learning rate and :math:`v` is the weighted moving average
    of the squared gradient.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        momentum (float, optional): momentum factor (default: 0)
        alpha (float, optional): smoothing constant (default: 0.99)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        centered (bool, optional) : if ``True``, compute the centered RMSProp,
            the gradient is normalized by an estimation of its variance
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    """

    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))

        defaults = dict(lr=lr, momentum=momentum, alpha=alpha, eps=eps, centered=centered, weight_decay=weight_decay)
        super(RMSprop, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RMSprop, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('momentum', 0)
            group.setdefault('centered', False)

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

        for group in self.param_groups:
            grads = []
            params_with_grad = []
            states = []
            alpha = group['alpha']
            square_avg = []

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
                        state['square_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if group['momentum'] > 0:
                            state['momentum_buffer'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if group['centered']:
                            state['grad_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                        state['step'] += 1

                    states.append(state)
                    square_avg.append(state['square_avg'])

            if group['weight_decay'] != 0:
                torch._foreach_add_(grads, params_with_grad, alpha=group['weight_decay'])

            torch._foreach_mul_(square_avg, alpha)
            torch._foreach_addcmul_(square_avg, grads, grads, value=1 - alpha)

            if group['centered']:
                grad_avgs = [s['grad_avg'] for s in states]
                torch._foreach_mul_(grad_avgs, alpha)
                torch._foreach_add_(grad_avgs, grads, alpha=1 - alpha)
                avg = torch._foreach_addcmul(square_avg, grad_avgs, grad_avgs, value=-1)
                torch._foreach_sqrt_(avg)
                torch._foreach_add_(avg, group['eps'])
            else:
                avg = torch._foreach_sqrt(square_avg)
                torch._foreach_add_(avg, group['eps'])

            if group['momentum'] > 0:
                buf = [s['momentum_buffer'] for s in states]
                torch._foreach_mul_(buf, group['momentum'])
                torch._foreach_addcdiv_(buf, grads, avg)
                torch._foreach_add_(params_with_grad, buf, alpha=-group['lr'])
            else:
                torch._foreach_addcdiv_(params_with_grad, grads, avg, value=-group['lr'])

        return loss
