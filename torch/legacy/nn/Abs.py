import torch
from .Module import Module


class Abs(Module):

    def __init__(self):
        super(Abs, self).__init__()

    def updateOutput(self, input):
        self._backend.Abs_updateOutput(
            self._backend.library_state,
            input,
            self.output
        )
        return self.output

    def updateGradInput(self, input, gradOutput):
        self._backend.Abs_updateGradInput(
            self._backend.library_state,
            input,
            gradOutput,
            self.gradInput
        )
        return self.gradInput
