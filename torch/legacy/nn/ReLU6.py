import torch
from .Module import Module


class ReLU6(Module):

    def __init__(self, inplace=False):
        super(ReLU6, self).__init__()
        self.inplace = inplace

    def updateOutput(self, input):
        self._backend.HardTanh_updateOutput(
            self._backend.library_state,
            input,
            self.output,
            0, 6, self.inplace
        )
        return self.output

    def updateGradInput(self, input, gradOutput):
        self._backend.HardTanh_updateGradInput(
            self._backend.library_state,
            input,
            gradOutput,
            self.gradInput,
            0, 6, self.inplace
        )
        return self.gradInput
