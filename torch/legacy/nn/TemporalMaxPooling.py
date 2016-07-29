import torch
from torch.legacy import nn

class TemporalMaxPooling(nn.Module):

    def __init__(self, kW, dW=None):
        super(TemporalMaxPooling, self).__init__()
        self.kW = kW
        self.dW = dW or kW
        self.indices = None

    def updateOutput(self, input):
        self.indices = self.indices or input.new()
        self._backend.TemporalMaxPooling_updateOutput(
            self._backend.library_state,
            input,
            self.output,
            self.indices,
            self.kW,
            self.dW
        )
        return self.output


    def updateGradInput(self, input, gradOutput):
        if not self.gradInput:
             return
        self._backend.TemporalMaxPooling_updateGradInput(
            self._backend.library_state,
            input,
            gradOutput,
            self.gradInput,
            self.indices,
            self.kW,
            self.dW
        )
        return self.gradInput

    def clearState(self):
        nn.utils.clear(self, 'indices')
        return super(TemporalMaxPooling, self).clearState()

