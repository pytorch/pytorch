import torch
from .Module import Module


class Tanh(Module):

    def updateOutput(self, input):
        self._backend.Tanh_updateOutput(
            self._backend.library_state,
            input,
            self.output
        )
        return self.output

    def updateGradInput(self, input, gradOutput):
        self._backend.Tanh_updateGradInput(
            self._backend.library_state,
            gradOutput,
            self.gradInput,
            self.output
        )
        return self.gradInput
