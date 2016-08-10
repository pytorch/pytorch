import torch
from torch.legacy import nn

class Exp(nn.Module):

    def updateOutput(self, input):
        return torch.exp(self.output, input)

    def updateGradInput(self, input, gradOutput):
        return torch.mul(self.gradInput, self.output, gradOutput)

