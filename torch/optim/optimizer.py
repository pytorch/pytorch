
class Optimizer(object):

    def __init__(self, model):
        self.model = model
        self.parameters = list(self.model.parameters())

    def _forward_backward(self, forward_closure):
        self.model.zero_grad()
        loss = forward_closure()
        loss.backward()
        return loss.data[0]

    def step(self, forward_closure):
        raise NotImplementedError

