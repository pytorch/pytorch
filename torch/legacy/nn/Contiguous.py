import torch
from torch.legacy import nn

class Contiguous(nn.Module):

    def updateOutput(self, input):
        if not input.isContiguous():
            self.output.resizeAs(input).copy(input)
        else:
            self.output.set(input)

        return self.output


    def updateGradInput(self, input, gradOutput):
        if not gradOutput.isContiguous():
            self.gradInput.resizeAs(gradOutput).copy(gradOutput)
        else:
            self.gradInput.set(gradOutput)

        return self.gradInput

