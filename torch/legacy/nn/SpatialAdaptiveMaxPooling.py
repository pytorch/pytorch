import torch
from torch.legacy import nn

class SpatialAdaptiveMaxPooling(nn.Module):

    def __init__(self, w, h):
        super(SpatialAdaptiveMaxPooling, self).__init__()
        self.w = w
        self.h = h
        self.indices = None

    def updateOutput(self, input):
        self.indices = self.indices or input.new()
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
        nn.utils.clear(self, 'indices')
        return super(SpatialAdaptiveMaxPooling, self).clearState()

