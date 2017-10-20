import torch
from .Module import Module


class LogSoftMax(Module):

    def __init__(self, dim=None):
        super(LogSoftMax, self).__init__()
        if dim is not None:
            self.dim = dim

    def _get_dim(self, input):
        return getattr(self, 'dim', 0 if input.dim() == 1 or input.dim() == 3 else 1)

    def updateOutput(self, input):
        self._backend.LogSoftMax_updateOutput(
            self._backend.library_state,
            input,
            self.output,
            self._get_dim(input)
        )
        return self.output

    def updateGradInput(self, input, gradOutput):
        self._backend.LogSoftMax_updateGradInput(
            self._backend.library_state,
            input,
            gradOutput,
            self.gradInput,
            self.output,
            self._get_dim(input)
        )
        return self.gradInput
