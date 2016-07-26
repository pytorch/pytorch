import torch
from torch.legacy import nn

class MulConstant(nn.Module):

    def __init__(self, constant_scalar, inplace=False):
        super(MulConstant, self).__init__()
        self.constant_scalar = constant_scalar

        # default for inplace is False
        self.inplace = inplace

    def updateOutput(self, input):
        if self.inplace:
            input.mul(self.constant_scalar)
            self.output.set(input)
        else:
            self.output.resizeAs(input)
            self.output.copy(input)
            self.output.mul(self.constant_scalar)

        return self.output


    def updateGradInput(self, input, gradOutput):
        if not self.gradInput:
            return

        if self.inplace:
            gradOutput.mul(self.constant_scalar)
            self.gradInput.set(gradOutput)
            # restore previous input value
            input.div(self.constant_scalar)
        else:
            self.gradInput.resizeAs(gradOutput)
            self.gradInput.copy(gradOutput)
            self.gradInput.mul(self.constant_scalar)

        return self.gradInput

