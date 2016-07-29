import torch
from torch.legacy import nn

class Narrow(nn.Module):

    def __init__(self, dimension, offset, length=1):
        super(Narrow, self).__init__()
        self.dimension = dimension
        self.index = offset
        self.length = length

    def updateOutput(self, input):
        length = self.length
        if length < 0:
           length = input.size(self.dimension) - self.index + self.length + 1

        output = input.narrow(self.dimension, self.index, length)
        self.output = self.output.typeAs(output)
        self.output.resizeAs(output).copy(output)
        return self.output


    def updateGradInput(self, input, gradOutput):
        length = self.length
        if length < 0:
           length = input.size(self.dimension) - self.index + self.length + 1

        self.gradInput = self.gradInput.typeAs(input)
        self.gradInput.resizeAs(input).zero()
        self.gradInput.narrow(self.dimension, self.index, length).copy(gradOutput)
        return self.gradInput

