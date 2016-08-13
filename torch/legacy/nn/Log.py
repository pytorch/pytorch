import torch
from .Module import Module

class Log(Module):

    def updateOutput(self, input):
        self.output.resizeAs_(input)
        self.output.copy_(input)
        self.output.log_()
        return self.output

    def updateGradInput(self, input, gradOutput) :
        self.gradInput.resizeAs_(input)
        self.gradInput.fill_(1)
        self.gradInput.div_(input)
        self.gradInput.mul_(gradOutput)
        return self.gradInput

