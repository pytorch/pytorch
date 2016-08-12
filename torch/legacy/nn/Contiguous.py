import torch
from torch.legacy import nn

class Contiguous(nn.Module):

    def updateOutput(self, input):
        if not input.isContiguous():
            self.output.resizeAs_(input).copy_(input)
        else:
            self.output.set_(input)

        return self.output


    def updateGradInput(self, input, gradOutput):
        if not gradOutput.isContiguous():
            self.gradInput.resizeAs_(gradOutput).copy_(gradOutput)
        else:
            self.gradInput.set_(gradOutput)

        return self.gradInput

