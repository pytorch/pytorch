import torch
from .Module import Module
from .utils import clear


class Dropout(Module):

    def __init__(self, p=0.5, inplace=False):
        super(Dropout, self).__init__()
        self.p = p
        self.inplace = inplace
        self.train = True
        self.noise = torch.Tensor()

    def updateOutput(self, input):
        if self.inplace:
            self.output.set_(input)
        else:
            self.output.resize_as_(input).copy_(input)

        if self.p > 0 and self.train:
            self.noise.resize_as_(input)
            self.noise.bernoulli_(1 - self.p)
            self.noise.div_(1 - self.p)
            self.output.mul_(self.noise)

        return self.output

    def updateGradInput(self, input, gradOutput):
        if self.inplace:
            self.gradInput.set_(gradOutput)
        else:
            self.gradInput.resize_as_(gradOutput).copy_(gradOutput)

        if self.p > 0 and self.train:
            self.gradInput.mul_(self.noise)  # simply mask the gradients with the noise vector

        return self.gradInput

    def setp(self, p):
        self.p = p

    def __repr__(self):
        return super(Dropout, self).__repr__() + '({:.4f})'.format(self.p)

    def clearState(self):
        clear(self, 'noise')
        return super(Dropout, self).clearState()
