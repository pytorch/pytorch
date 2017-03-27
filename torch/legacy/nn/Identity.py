import torch
from .Module import Module
from .utils import clear


class Identity(Module):

    def updateOutput(self, input):
        self.output = input
        return self.output

    def updateGradInput(self, input, gradOutput):
        self.gradInput = gradOutput
        return self.gradInput

    def clearState(self):
        clear(self, [
            'output',
            'gradInput',
        ])
        return super(Identity, self).clearState()
