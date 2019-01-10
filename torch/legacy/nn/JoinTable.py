import torch
from .Module import Module


class JoinTable(Module):

    def __init__(self, dimension):
        super(JoinTable, self).__init__()
        self.size = torch.Size()
        self.dimension = dimension
        self.gradInput = []

    def _getPositiveDimension(self, input):
        dimension = self.dimension
        if dimension < 0:
            dimension = input[0].dim() + dimension

        return dimension

    def updateOutput(self, input):
        dim = self._getPositiveDimension(input)

        for i in range(len(input)):
            currentOutput = input[i]
            if i == 0:
                size = list(currentOutput.size())
            else:
                size[dim] += currentOutput.size(dim)

        self.size = torch.Size(size)
        self.output.resize_(self.size)

        # TODO: use cat?
        offset = 0
        for i in range(len(input)):
            currentOutput = input[i]
            self.output.narrow(dim, offset, currentOutput.size(dim)).copy_(currentOutput)
            offset += currentOutput.size(dim)

        return self.output

    def updateGradInput(self, input, gradOutput):
        dim = self._getPositiveDimension(input)

        for i in range(len(input)):
            if len(self.gradInput) < i + 1:
                self.gradInput.append(input[i].new())
            self.gradInput[i].resize_as_(input[i])
        self.gradInput = self.gradInput[:len(input)]

        offset = 0
        for i in range(len(input)):
            currentOutput = input[i]
            currentGradInput = gradOutput.narrow(dim, offset, currentOutput.size(dim))
            self.gradInput[i].copy_(currentGradInput)
            offset = offset + currentOutput.size(dim)

        return self.gradInput

    def type(self, type=None, tensorCache=None):
        self.gradInput = []
        return super(JoinTable, self).type(type, tensorCache)
