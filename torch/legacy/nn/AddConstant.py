import torch
from .Module import Module


class AddConstant(Module):

    def __init__(self, constant_scalar, inplace=False):
        super(AddConstant, self).__init__()
        self.constant_scalar = constant_scalar
        self.inplace = inplace

    def updateOutput(self, input):
        if self.inplace:
            input.add_(self.constant_scalar)
            self.output.set_(input)
        else:
            self.output.resize_as_(input)
            self.output.copy_(input)
            self.output.add_(self.constant_scalar)

        return self.output

    def updateGradInput(self, input, gradOutput):
        if self.inplace:
            self.gradInput.set_(gradOutput)
            # restore previous input value
            input.add_(-self.constant_scalar)
        else:
            self.gradInput.resize_as_(gradOutput)
            self.gradInput.copy_(gradOutput)

        return self.gradInput
