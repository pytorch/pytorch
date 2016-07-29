import torch
from torch.legacy import nn

class SoftSign(nn.Module):

    def __init__(self):
        super(SoftSign, self).__init__()
        self.temp = None
        self.tempgrad = None

    def updateOutput(self, input):
        self.temp = self.temp or input.new()
        self.temp.resizeAs(input).copy(input).abs().add(1)
        self.output.resizeAs(input).copy(input).cdiv(self.temp)
        return self.output

    def updateGradInput(self, input, gradOutput):
        self.tempgrad = self.tempgrad or input.new()
        self.tempgrad.resizeAs(self.output).copy(input).abs().add(1).cmul(self.tempgrad)
        self.gradInput.resizeAs(input).copy(gradOutput).cdiv(self.tempgrad)
        return self.gradInput

    def clearState(self):
        nn.utils.clear(self, 'temp', 'tempgrad')
        return super(SoftSign, self).clearState(self)

