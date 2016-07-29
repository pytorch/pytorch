import torch
from torch.legacy import nn

class TanhShrink(nn.Module):

    def __init__(self):
        super(TanhShrink, self).__init__()
        self.tanh = nn.Tanh()

    def updateOutput(self, input):
        th = self.tanh.updateOutput(input)
        self.output.resizeAs(input).copy(input)
        self.output.add(-1, th)
        return self.output

    def updateGradInput(self, input, gradOutput):
        dth = self.tanh.updateGradInput(input, gradOutput)
        self.gradInput.resizeAs(input).copy(gradOutput)
        self.gradInput.add(-1, dth)
        return self.gradInput

