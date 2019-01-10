import torch
from .Module import Module


class Exp(Module):

    def updateOutput(self, input):
        return torch.exp(input, out=self.output)

    def updateGradInput(self, input, gradOutput):
        return torch.mul(self.output, gradOutput, out=self.gradInput)
