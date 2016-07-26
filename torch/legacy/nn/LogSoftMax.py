import torch
from torch.legacy import nn

class LogSoftMax(nn.Module):

    def updateOutput(self, input):
        self._backend.LogSoftMax_updateOutput(
            self._backend.library_state,
            input,
            self.output
        )
        return self.output


    def updateGradInput(self, input, gradOutput):
        self._backend.LogSoftMax_updateGradInput(
            self._backend.library_state,
            input,
            gradOutput,
            self.gradInput,
            self.output
        )
        return self.gradInput

