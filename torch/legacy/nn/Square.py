import torch
from .Module import Module


class Square(Module):

    def updateOutput(self, input):
        self._backend.Square_updateOutput(
            self._backend.library_state,
            input,
            self.output
        )
        return self.output

    def updateGradInput(self, input, gradOutput):
        self._backend.Square_updateGradInput(
            self._backend.library_state,
            input,
            gradOutput,
            self.gradInput
        )
        return self.gradInput
