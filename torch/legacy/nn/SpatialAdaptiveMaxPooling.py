import torch
from .Module import Module
from .utils import clear


class SpatialAdaptiveMaxPooling(Module):

    def __init__(self, w, h):
        super(SpatialAdaptiveMaxPooling, self).__init__()
        self.w = w
        self.h = h
        self.indices = None

    def updateOutput(self, input):
        if self.indices is None:
            self.indices = input.new()
        self.indices = self.indices.long()
        self._backend.SpatialAdaptiveMaxPooling_updateOutput(
            self._backend.library_state,
            input,
            self.output,
            self.indices,
            self.w,
            self.h
        )
        return self.output

    def updateGradInput(self, input, gradOutput):
        self._backend.SpatialAdaptiveMaxPooling_updateGradInput(
            self._backend.library_state,
            input,
            gradOutput,
            self.gradInput,
            self.indices
        )
        return self.gradInput

    def clearState(self):
        clear(self, 'indices')
        return super(SpatialAdaptiveMaxPooling, self).clearState()
