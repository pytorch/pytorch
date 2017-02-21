from .optimizer import Optimizer


class RMSprop(Optimizer):
    """Implements RMSprop algorithm.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        momentum (float, optional): momentum factor (default: 0)
        alpha (float, optional): smoothing constant (default: 0.99)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        centered (bool, optional) : if True, compute the centered RMSProp,
            the gradient is normalized by an estimation of its variance
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    """

    def __init__(self, params, lr=1e-2, momentum=0, alpha=0.99, eps=1e-8, centered=False, weight_decay=0):
        defaults = dict(lr=lr, momentum=0, alpha=alpha, eps=eps, centered=False, weight_decay=weight_decay)
        super(RMSprop, self).__init__(params, defaults)

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
                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['grad_avg'] = grad.new().resize_as_(grad).zero_()
                    state['square_avg'] = grad.new().resize_as_(grad).zero_()
                    state['momentum'] = grad.new().resize_as_(grad).zero_()

                grad_avg = state['grad_avg']
                square_avg = state['square_avg']
                momentum = state['momentum']
                alpha = group['alpha']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                square_avg.mul_(alpha).addcmul_(1 - alpha, grad, grad)

                avg = None
                if group['centered']:
                    grad_avg.mul_(alpha).add_(1 - alpha, grad)
                    avg = square_avg.addcmul(-1, grad_avg, grad_avg).sqrt().add_(group['eps'])
                else:
                    avg = square_avg.sqrt().add_(group['eps'])

                momentum.mul_(group['momentum']).addcdiv_(-group['lr'], grad, avg)
                p.data.add_(-momentum)

        return loss
