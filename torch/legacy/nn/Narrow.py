import torch
from .Module import Module


class Narrow(Module):

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
        self.output = self.output.type_as(output)
        self.output.resize_as_(output).copy_(output)
        return self.output

    def updateGradInput(self, input, gradOutput):
        length = self.length
        if length < 0:
            length = input.size(self.dimension) - self.index + self.length + 1

        self.gradInput = self.gradInput.type_as(input)
        self.gradInput.resize_as_(input).zero_()
        self.gradInput.narrow(self.dimension, self.index, length).copy_(gradOutput)
        return self.gradInput
