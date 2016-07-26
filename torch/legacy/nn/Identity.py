import torch
from torch.legacy import nn

class Identity(nn.Module):

    def updateOutput(self, input):
        self.output = input
        return self.output

    def updateGradInput(self, input, gradOutput):
        self.gradInput = gradOutput
        return self.gradInput

    def clearState(self):
        nn.utils.clear(self, [
           'output',
           'gradInput',
        ])
        return super(Identity, self).clearState()

