import torch
from .Module import Module
from .utils import clear


class SoftMin(Module):

    def __init__(self):
        super(SoftMin, self).__init__()
        self.mininput = None

    def updateOutput(self, input):
        if self.mininput is None:
            self.mininput = input.new()
        self.mininput.resize_as_(input).copy_(input).mul_(-1)
        self._backend.SoftMax_updateOutput(
            self._backend.library_state,
            self.mininput,
            self.output
        )
        return self.output

    def updateGradInput(self, input, gradOutput):
        if self.mininput is None:
            self.mininput = input.new()
        self.mininput.resize_as_(input).copy_(input).mul_(-1)
        self._backend.SoftMax_updateGradInput(
            self._backend.library_state,
            self.mininput,
            gradOutput,
            self.gradInput,
            self.output
        )

        self.gradInput.mul_(-1)
        return self.gradInput

    def clearState(self):
        clear(self, 'mininput')
        return super(SoftMin, self).clearState()
