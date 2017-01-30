import torch
from .Module import Module
from .utils import recursiveCopy


class SelectTable(Module):

    def __init__(self, index):
        super(SelectTable, self).__init__()
        self.index = index
        self.gradInput = []

    def updateOutput(self, input):
        # handle negative indices
        index = self.index if self.index >= 0 else input.size(self.dimension) + self.index
        assert len(input) > index
        self.output = input[index]
        return self.output

    def _zeroTableCopy(self, l1, l2):
        for i, v in enumerate(l2):
            if isinstance(v, list):
                if len(l1) > i:
                    l1[i] = self._zeroTableCopy(l1[i], l2[i])
                else:
                    l1.append(self._zeroTableCopy([], l2[i]))
            else:
                if i >= len(l1):
                    l1.append(v.new().resize_as_(v).zero_())
                else:
                    l1[i].resize_as_(v)
                    l1[i].zero_()
        del l1[len(l2):]
        return l1

    def updateGradInput(self, input, gradOutput):
        # make gradInput a zeroed copy of input
        self._zeroTableCopy(self.gradInput, input)
        # handle negative indices
        index = self.index if self.index >= 0 else input.size(self.dimension) + self.index
        # copy into gradInput[index] (necessary for variable sized inputs)
        assert self.gradInput[index] is not None
        recursiveCopy(self.gradInput[index], gradOutput)
        return self.gradInput

    def type(self, type, tensorCache=None):
        del self.gradInput[:]
        if isinstance(self.output, list):
            del self.output[:]
        return super(SelectTable, self).type(type, tensorCache)

    def __repr__(self):
        return super(SelectTable, self).__repr__() + '({})'.format(self.index)
