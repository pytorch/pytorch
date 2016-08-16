import torch
from .Module import Module


class SoftShrink(Module):

    def __init__(self, lambd=0.5):
        super(SoftShrink, self).__init__()
        self.lambd = lambd

    def updateOutput(self, input):
        self._backend.SoftShrink_updateOutput(
            self._backend.library_state,
            input,
            self.output,
            self.lambd
        )
        return self.output

    def updateGradInput(self, input, gradOutput):
        self._backend.SoftShrink_updateGradInput(
            self._backend.library_state,
            input,
            gradOutput,
            self.gradInput,
            self.lambd
        )
        return self.gradInput
