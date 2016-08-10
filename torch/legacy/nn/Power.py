import torch
from torch.legacy import nn

class Power(nn.Module):

    def __init__(self, p):
        super(Power, self).__init__()
        self.pow = p

    def updateOutput(self, input):
        self.output.resizeAs_(input).copy(input)
        self.output.pow_(self.pow)
        return self.output

    def updateGradInput(self, input, gradOutput):
        self.gradInput.resizeAs_(input).copy(input)
        self.gradInput.pow_(self.pow - 1)
        self.gradInput.mul_(gradOutput).mul_(self.pow)
        return self.gradInput

