import torch
from .Module import Module
from .Tanh import Tanh

class TanhShrink(Module):

    def __init__(self):
        super(TanhShrink, self).__init__()
        self.tanh = Tanh()

    def updateOutput(self, input):
        th = self.tanh.updateOutput(input)
        self.output.resizeAs_(input).copy_(input)
        self.output.add_(-1, th)
        return self.output

    def updateGradInput(self, input, gradOutput):
        dth = self.tanh.updateGradInput(input, gradOutput)
        self.gradInput.resizeAs_(input).copy_(gradOutput)
        self.gradInput.add_(-1, dth)
        return self.gradInput

