import torch
from torch.legacy import nn

class SplitTable(nn.Module):

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
        if not self.gradInput:
            return
        dimension = self._getPositiveDimension(input)
        slices = input.size(dimension)
        self.gradInput.resizeAs_(input)

        for i in range(slices):
            self.gradInput.select(dimension, i).copy(gradOutput[i])

        return self.gradInput

