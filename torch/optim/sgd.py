from .optimizer import Optimizer, required


class SGD(Optimizer):
    """Implements stochastic gradient descent with optional momentum.

    Args:
        params: parameters to optimize
        lr: learning rate
        momentum: momentum factory (default: 0)
        weight_decay: weight decay (L2 penalty) (default: 0)
    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> def closure():
        ...     output = model(input)
        ...     loss = criterion(output, target)
        ...     loss.backward()
        >>> optimizer.zero_grad()
        >>> optimizer.step(closure)
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay)
        super(SGD, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']

            for p in group['params']:
                d_p = p.grad
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[id(p)]
                    if 'momentum_buffer' not in param_state:
                        param_state['momentum_buffer'] = d_p.clone()
                    else:
                        buf = param_state['momentum_buffer']
                        d_p = buf.mul_(momentum).add_(1 - dampening, d_p)

                p.data.add_(-group['lr'], d_p)

        return loss
