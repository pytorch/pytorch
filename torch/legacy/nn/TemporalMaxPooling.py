import torch
from .Module import Module
from .utils import clear


class TemporalMaxPooling(Module):

    def __init__(self, kW, dW=None):
        super(TemporalMaxPooling, self).__init__()
        self.kW = kW
        self.dW = dW or kW
        self.indices = None

    def updateOutput(self, input):
        if self.indices is None:
            self.indices = input.new()
        self._backend.TemporalMaxPooling_updateOutput(
            self._backend.library_state,
            input,
            self.output,
            self.indices,
            self.kW,
            self.dW
        )
        return self.output

    def updateGradInput(self, input, gradOutput):
        if self.gradInput is None:
            return
        self._backend.TemporalMaxPooling_updateGradInput(
            self._backend.library_state,
            input,
            gradOutput,
            self.gradInput,
            self.indices,
            self.kW,
            self.dW
        )
        return self.gradInput

    def clearState(self):
        clear(self, 'indices')
        return super(TemporalMaxPooling, self).clearState()
