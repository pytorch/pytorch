from copy import copy
from collections import defaultdict

class Optimizer(object):

    def __init__(self, model):
        self.model = model
        self.state = defaultdict(dict)
        self.parameters = list(self.model.parameters())

    def __getstate__(self):
        return {
            'state': self.state,
            'parameters': self.parameters,
        }

    def state_dict(self):
        return self.__getstate__()

    def _forward_backward(self, forward_closure):
        self.model.zero_grad()
        loss = forward_closure()
        loss.backward()
        return loss.data[0]

    def step(self, forward_closure):
        raise NotImplementedError

