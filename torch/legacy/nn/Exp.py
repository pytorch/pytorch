import torch
from torch.legacy import nn

class Exp(nn.Module):

    def updateOutput(self, input):
        return self.output.exp(input)

    def updateGradInput(self, input, gradOutput):
        return self.gradInput.cmul(self.output, gradOutput)

