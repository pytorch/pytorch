import torch
from torch.legacy import nn

class LogSigmoid(nn.Module):

    def __init__(self):
        super(LogSigmoid, self).__init__()
        self.buffer = None

    def updateOutput(self, input):
        self.buffer = self.buffer or input.new()
        self._backend.LogSigmoid_updateOutput(
            self._backend.library_state,
            input,
            self.output,
            self.buffer
        )
        return self.output

    def updateGradInput(self, input, gradOutput):
        self._backend.LogSigmoid_updateGradInput(
            self._backend.library_state,
            input,
            gradOutput,
            self.gradInput,
            self.buffer
        )
        return self.gradInput

    def clearState(self):
        nn.utils.clear(self, 'buffer')
        return super(LogSigmoid, self).clearState()


