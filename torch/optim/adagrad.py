from .optimizer import Optimizer

class Adagrad(Optimizer):

    def __init__(self, params, lr=1e-2, lr_decay=0, weight_decay=0):
        defaults = dict(lr=lr, lr_decay=lr_decay, weight_decay=weight_decay)
        super(Adagrad, self).__init__(params, defaults)

    def step(self, forward_closure=None):
        loss = None
        if forward_closure is not None:
            loss = self._forward_backward(forward_closure)

        for group in self.param_groups:
            for p in group['params']:
                grad = p.grad
                state = self.state[id(p)]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['sum'] = grad.new().resize_as_(grad).zero_()

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                clr = group['lr'] / (1 + (state['step'] - 1) * group['lr_decay'])

                state['sum'].addcmul_(1, grad, grad)
                std = state['sum'].sqrt().add_(1e-10)
                p.data.addcdiv_(-clr, grad, std)

        return loss

