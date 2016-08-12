import torch
from torch.legacy import nn

class SoftMin(nn.Module):

    def __init__(self):
        super(SoftMin, self).__init__()
        self.mininput = None

    def updateOutput(self, input):
        self.mininput = self.mininput or input.new()
        self.mininput.resizeAs_(input).copy_(input).mul_(-1)
        self._backend.SoftMax_updateOutput(
            self._backend.library_state,
            self.mininput,
            self.output
        )
        return self.output

    def updateGradInput(self, input, gradOutput):
        self.mininput = self.mininput or input.new()
        self.mininput.resizeAs_(input).copy_(input).mul_(-1)
        self._backend.SoftMax_updateGradInput(
            self._backend.library_state,
            self.mininput,
            gradOutput,
            self.gradInput,
            self.output
        )

        self.gradInput.mul_(-1)
        return self.gradInput

    def clearState(self):
        nn.utils.clear(self, 'mininput')
        return super(SoftMin, self).clearState()

