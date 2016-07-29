import torch
from torch.legacy import nn

class Power(nn.Module):

    def __init__(self, p):
        super(Power, self).__init__()
        self.pow = p

    def updateOutput(self, input):
        self.output.resizeAs(input).copy(input)
        self.output.pow(self.pow)
        return self.output

    def updateGradInput(self, input, gradOutput):
        self.gradInput.resizeAs(input).copy(input)
        self.gradInput.pow(self.pow - 1)
        self.gradInput.cmul(gradOutput).mul(self.pow)
        return self.gradInput

