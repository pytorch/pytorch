import torch
from .Module import Module


class SpatialSoftMax(Module):

    def updateOutput(self, input):
        self._backend.SoftMax_updateOutput(
            self._backend.library_state,
            input,
            self.output
        )
        return self.output

    def updateGradInput(self, input, gradOutput):
        self._backend.SoftMax_updateGradInput(
            self._backend.library_state,
            input,
            gradOutput,
            self.gradInput,
            self.output
        )
        return self.gradInput
