import torch
from .Module import Module


class View(Module):

    def resetSize(self, *args):
        if len(args) == 1 and isinstance(args[0], torch.Size):
            self.size = args[0]
        else:
            self.size = torch.Size(args)

        self.numElements = 1
        inferdim = False
        for i in range(len(self.size)):
            szi = self.size[i]
            if szi >= 0:
                self.numElements = self.numElements * self.size[i]
            else:
                assert szi == -1
                assert not inferdim
                inferdim = True

        return self

    def __init__(self, *args):
        super(View, self).__init__()
        self.resetSize(*args)

    def updateOutput(self, input):
        if self.output is None:
            self.output = input.new()
        self.output = input.view(self.size)
        return self.output

    def updateGradInput(self, input, gradOutput):
        if self.gradInput is None:
            self.gradInput = gradOutput.new()
        self.gradInput = gradOutput.contiguous().view(input.size())
        return self.gradInput

    def __repr__(self):
        return super(View, self).__repr__() + '({})'.format(', '.join(map(str, self.size)))
