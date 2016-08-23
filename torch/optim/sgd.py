from collections import defaultdict
from .optimizer import Optimizer

class SGD(Optimizer):

    def __init__(self, model, lr, momentum=0, dampening=None):
        super(SGD, self).__init__(model)
        self.lr = lr
        self.momentum = momentum
        self.dampening = dampening or momentum
        self.state = defaultdict(dict)

    def step(self, *input):
        loss = self._forward_backward(input)
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
