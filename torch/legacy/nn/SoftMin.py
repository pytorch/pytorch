import torch
from .Module import Module
from .utils import clear


class SoftMin(Module):

    def __init__(self, dim=None):
        super(SoftMin, self).__init__()
        self.mininput = None
        if dim is not None:
            self.dim = dim

    def _get_dim(self, input):
        return getattr(self, 'dim', 0 if input.dim() == 1 or input.dim() == 3 else 1)

    def updateOutput(self, input):
        if self.mininput is None:
            self.mininput = input.new()
        self.mininput.resize_as_(input).copy_(input).mul_(-1)
        self._backend.SoftMax_updateOutput(
            self._backend.library_state,
            self.mininput,
            self.output,
            self._get_dim(input)
        )
        return self.output

    def updateGradInput(self, input, gradOutput):
        if self.mininput is None:
            self.mininput = input.new()
        self.mininput.resize_as_(input).copy_(input).mul_(-1)
        self._backend.SoftMax_updateGradInput(
            self._backend.library_state,
            self.mininput,
            gradOutput,
            self.gradInput,
            self.output,
            self._get_dim(input)
        )

        self.gradInput.mul_(-1)
        return self.gradInput

    def clearState(self):
        clear(self, 'mininput')
        return super(SoftMin, self).clearState()
