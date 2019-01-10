import torch
from .Module import Module


class SplitTable(Module):

    def __init__(self, dimension):
        super(SplitTable, self).__init__()
        self.dimension = dimension

    def _getPositiveDimension(self, input):
        dimension = self.dimension
        if dimension < 0:
            dimension = input.dim() + dimension

        return dimension

    def updateOutput(self, input):
        dimension = self._getPositiveDimension(input)
        slices = input.size(dimension)

        currentOutput = []
        for i in range(slices):
            currentOutput.append(input.select(dimension, i))

        self.output = currentOutput
        return self.output

    def updateGradInput(self, input, gradOutput):
        if self.gradInput is None:
            return
        dimension = self._getPositiveDimension(input)
        slices = input.size(dimension)
        self.gradInput.resize_as_(input)

        for i in range(slices):
            self.gradInput.select(dimension, i).copy_(gradOutput[i])

        return self.gradInput
