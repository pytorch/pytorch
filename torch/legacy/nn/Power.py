import torch
from .Module import Module


class Power(Module):

    def __init__(self, p):
        super(Power, self).__init__()
        self.pow = p

    def updateOutput(self, input):
        self.output.resize_as_(input).copy_(input)
        self.output.pow_(self.pow)
        return self.output

    def updateGradInput(self, input, gradOutput):
        self.gradInput.resize_as_(input).copy_(input)
        self.gradInput.pow_(self.pow - 1)
        self.gradInput.mul_(gradOutput).mul_(self.pow)
        return self.gradInput
