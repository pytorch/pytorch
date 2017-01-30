import torch
from .Module import Module
from .utils import clear, recursiveResizeAs, recursiveFill


class NarrowTable(Module):

    def __init__(self, offset, length=1):
        super(NarrowTable, self).__init__()
        self.offset = offset
        self.length = length
        self.output = []
        self.gradInput = []

    def updateOutput(self, input):
        self.output[:] = [input[self.offset + i] for i in range(self.length)]
        return self.output

    def updateGradInput(self, input, gradOutput):
        if len(self.gradInput) != len(input):
            self.gradInput[:] = [None for i in range(len(input))]

        assert len(gradOutput) == self.length
        for i in range(self.length):
            self.gradInput[self.offset + i] = gradOutput[i]

        for i in range(len(input)):
            if i < self.offset or i >= self.offset + self.length:
                gi = self.gradInput[i]
                if gi is None:
                    gi = input[i].new()
                self.gradInput[i] = recursiveResizeAs(gi, input[i])[0]
                recursiveFill(self.gradInput[i], 0)

        return self.gradInput

    def type(self, type=None, tensorCache=None):
        if not type:
            return self._type
        clear(self, 'output', 'gradInput')
        return super(NarrowTable, self).type(self, type, tensorCache)
