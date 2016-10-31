from .optimizer import Optimizer, required

class SGD(Optimizer):

    def __init__(self, params, lr=required, momentum=0, dampening=0):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening)
        super(SGD, self).__init__(params, defaults)

    def step(self, forward_closure=None):
        loss = None
        if forward_closure is not None:
            loss = self._forward_backward(forward_closure)

        for group in self.param_groups:
            for p in group['params']:
                if group['momentum'] != 0:
                    param_state = self.state[id(p)]
                    if not 'momentum_buffer' in param_state:
                        param_state['momentum_buffer'] = p.grad.clone()
                    else:
                        param_state['momentum_buffer'].mul_(group['momentum']).add_(1 - group['dampening'], p.grad)
                    d_p = param_state['momentum_buffer']
                else:
                    d_p = p.grad
                p.data.add_(-group['lr'], d_p)

        return loss
