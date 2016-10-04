from .optimizer import Optimizer

class SGD(Optimizer):

    def __init__(self, model, lr, momentum=0, dampening=None):
        super(SGD, self).__init__(model)
        self.lr = lr
        self.momentum = momentum
        self.dampening = dampening or 0

    def __getstate__(self):
        state = super(SGD, self).__getstate__()
        state.update(
            lr=self.lr,
            momentum=self.momentum,
            dampening=self.dampening
        )
        return state

    def step(self, forward_closure):
        loss = self._forward_backward(forward_closure)

        for p in self.parameters:
            if self.momentum != 0:
                param_state = self.state[id(p)]
                if not 'momentum_buffer' in param_state:
                    param_state['momentum_buffer'] = p.grad.clone()
                else:
                    param_state['momentum_buffer'].mul_(self.momentum).add_(1 - self.dampening, p.grad)
                d_p = param_state['momentum_buffer']
            else:
                d_p = p.grad
            p.data.add_(-self.lr, d_p)

        return loss
