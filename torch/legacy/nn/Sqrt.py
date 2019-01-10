import torch
from .Module import Module


class Sqrt(Module):

    def __init__(self, b=0, eps=0):
        super(Sqrt, self).__init__()
        self.eps = b
        self.eps = eps

    def updateOutput(self, input):
        self._backend.Sqrt_updateOutput(
            self._backend.library_state,
            input,
            self.output,
            self.eps
        )
        return self.output

    def updateGradInput(self, input, gradOutput):
        self._backend.Sqrt_updateGradInput(
            self._backend.library_state,
            input,
            gradOutput,
            self.gradInput,
            self.output
        )
        return self.gradInput
