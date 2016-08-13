import torch
from .Module import Module

class JoinTable(Module):

    def __init__(self, dimension):
        super(JoinTable, self).__init__()
        self.size = torch.LongStorage()
        self.dimension = dimension
        self.gradInput = []

    def _getPositiveDimension(self, input):
        dimension = self.dimension
        if dimension < 0:
            dimension = input[0].dim() + dimension

        return dimension

    def updateOutput(self, input):
        dimension = self._getPositiveDimension(input)

        for i in range(len(input)):
           currentOutput = input[i]
           if i == 0:
              self.size.resize_(currentOutput.dim()).copy_(currentOutput.size())
           else:
              self.size[dimension] = self.size[dimension] + currentOutput.size(dimension)

        self.output.resize_(self.size)

        # TODO: use cat?
        offset = 0
        for i in range(len(input)):
           currentOutput = input[i]
           self.output.narrow(dimension, offset,
              currentOutput.size(dimension)).copy_(currentOutput)
           offset = offset + currentOutput.size(dimension)

        return self.output

    def updateGradInput(self, input, gradOutput):
        dimension = self._getPositiveDimension(input)

        for i in range(len(input)):
           if i not in self.gradInput:
              self.gradInput.append(input[i].new())
           self.gradInput[i].resizeAs_(input[i])
        self.gradInput = self.gradInput[:len(input)]

        offset = 0
        for i in range(len(input)):
           currentOutput = input[i]
           currentGradInput = gradOutput.narrow(dimension, offset,
                           currentOutput.size(dimension))
           self.gradInput[i].copy_(currentGradInput)
           offset = offset + currentOutput.size(dimension)

        return self.gradInput

    def type(self, type=None, tensorCache=None):
        self.gradInput = []
        return super(JoinTable, self).type(type, tensorCache)

