import torch
from .Module import Module

class Exp(Module):

    def updateOutput(self, input):
        return torch.exp(self.output, input)

    def updateGradInput(self, input, gradOutput):
        return torch.mul(self.gradInput, self.output, gradOutput)

