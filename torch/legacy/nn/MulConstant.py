import torch
from .Module import Module


class MulConstant(Module):

    def __init__(self, constant_scalar, inplace=False):
        super(MulConstant, self).__init__()
        self.constant_scalar = constant_scalar
        self.inplace = inplace

    def updateOutput(self, input):
        if self.inplace:
            input.mul_(self.constant_scalar)
            self.output.set_(input)
        else:
            self.output.resize_as_(input)
            self.output.copy_(input)
            self.output.mul_(self.constant_scalar)

        return self.output

    def updateGradInput(self, input, gradOutput):
        if self.gradInput is None:
            return

        if self.inplace:
            gradOutput.mul_(self.constant_scalar)
            self.gradInput.set_(gradOutput)
            # restore previous input value
            input.div_(self.constant_scalar)
        else:
            self.gradInput.resize_as_(gradOutput)
            self.gradInput.copy_(gradOutput)
            self.gradInput.mul_(self.constant_scalar)

        return self.gradInput
