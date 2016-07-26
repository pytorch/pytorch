import torch
from torch.legacy import nn

class AddConstant(nn.Module):

    def __init__(self, constant_scalar, inplace=False):
        super(AddConstant, self).__init__()
        self.constant_scalar = constant_scalar
        self.inplace = inplace

    def updateOutput(self, input):
        if self.inplace:
            input.add(self.constant_scalar)
            self.output.set(input)
        else:
            self.output.resizeAs(input)
            self.output.copy(input)
            self.output.add(self.constant_scalar)

        return self.output

    def updateGradInput(self, input, gradOutput):
        if self.inplace:
            self.gradInput.set(gradOutput)
            # restore previous input value
            input.add(-self.constant_scalar)
        else:
            self.gradInput.resizeAs(gradOutput)
            self.gradInput.copy(gradOutput)

        return self.gradInput

