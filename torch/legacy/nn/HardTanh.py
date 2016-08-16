import torch
from .Module import Module


class HardTanh(Module):

    def __init__(self, min_value=-1, max_value=1, inplace=False):
        super(HardTanh, self).__init__()
        self.min_val = min_value
        self.max_val = max_value
        self.inplace = inplace
        assert self.max_val > self.min_val

    def updateOutput(self, input):
        self._backend.HardTanh_updateOutput(
            self._backend.library_state,
            input,
            self.output,
            self.min_val,
            self.max_val,
            self.inplace
        )
        return self.output

    def updateGradInput(self, input, gradOutput):
        self._backend.HardTanh_updateGradInput(
            self._backend.library_state,
            input,
            gradOutput,
            self.gradInput,
            self.min_val,
            self.max_val,
            self.inplace
        )
        return self.gradInput
