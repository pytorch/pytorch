import torch
from torch.legacy import nn

class MulConstant(nn.Module):

    def __init__(self, constant_scalar, inplace=False):
        super(MulConstant, self).__init__()
        self.constant_scalar = constant_scalar
        self.inplace = inplace

    def updateOutput(self, input):
        if self.inplace:
            input.mul_(self.constant_scalar)
            self.output.set_(input)
        else:
            self.output.resizeAs_(input)
            self.output.copy_(input)
            self.output.mul_(self.constant_scalar)

        return self.output


    def updateGradInput(self, input, gradOutput):
        if not self.gradInput:
            return

        if self.inplace:
            gradOutput.mul_(self.constant_scalar)
            self.gradInput.set_(gradOutput)
            # restore previous input value
            input.div_(self.constant_scalar)
        else:
            self.gradInput.resizeAs_(gradOutput)
            self.gradInput.copy_(gradOutput)
            self.gradInput.mul_(self.constant_scalar)

        return self.gradInput

