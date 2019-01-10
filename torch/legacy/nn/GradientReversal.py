import torch
from .Module import Module


class GradientReversal(Module):

    def __init__(self, lambd=1):
        super(GradientReversal, self).__init__()
        self.lambd = lambd

    def setLambda(self, lambd):
        self.lambd = lambd

    def updateOutput(self, input):
        self.output.set_(input)
        return self.output

    def updateGradInput(self, input, gradOutput):
        self.gradInput.resize_as_(gradOutput)
        self.gradInput.copy_(gradOutput)
        self.gradInput.mul_(-self.lambd)
        return self.gradInput
