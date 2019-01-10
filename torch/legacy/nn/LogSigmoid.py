import torch
from .Module import Module
from .utils import clear


class LogSigmoid(Module):

    def __init__(self):
        super(LogSigmoid, self).__init__()
        self.buffer = None

    def updateOutput(self, input):
        if self.buffer is None:
            self.buffer = input.new()
        self._backend.LogSigmoid_updateOutput(
            self._backend.library_state,
            input,
            self.output,
            self.buffer
        )
        return self.output

    def updateGradInput(self, input, gradOutput):
        self._backend.LogSigmoid_updateGradInput(
            self._backend.library_state,
            input,
            gradOutput,
            self.gradInput,
            self.buffer
        )
        return self.gradInput

    def clearState(self):
        clear(self, 'buffer')
        return super(LogSigmoid, self).clearState()
