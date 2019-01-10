import torch
from .Module import Module


class Copy(Module):

    def __init__(self, intype, outtype, dontCast=False):
        self.dontCast = dontCast
        super(Copy, self).__init__()
        self.gradInput = intype()
        self.output = outtype()

    def updateOutput(self, input):
        self.output.resize_(input.size()).copy_(input)
        return self.output

    def updateGradInput(self, input, gradOutput):
        self.gradInput.resize_(gradOutput.size()).copy_(gradOutput)
        return self.gradInput

    def type(self, type=None, tensorCache=None):
        if type and self.dontCast:
            return self

        return super(Copy, self).type(self, type, tensorCache)
