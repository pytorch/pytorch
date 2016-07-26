import torch
from torch.legacy import nn

class Log(nn.Module):

    def updateOutput(self, input):
        self.output.resizeAs(input)
        self.output.copy(input)
        self.output.log()
        return self.output

    def updateGradInput(self, input, gradOutput) :
        self.gradInput.resizeAs(input)
        self.gradInput.fill(1)
        self.gradInput.cdiv(input)
        self.gradInput.cmul(gradOutput)
        return self.gradInput

