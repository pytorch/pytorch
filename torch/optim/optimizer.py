
class Optimizer(object):

    def __init__(self, model):
        if isinstance(model, tuple):
            self.model = model[0]
            self.criterion = model[1]
        else:
            self.model = model
            self.criterion = None

        self.parameters = list(self.model.parameters())

    def _forward(self, input):
        if self.criterion is not None:
            input, target = input
            return self.criterion(self.model(input), target)
        else:
            return model(input)

    def _forward_backward(self, input):
        loss = self._forward(input)
        loss.backward()
        return loss.data[0]

