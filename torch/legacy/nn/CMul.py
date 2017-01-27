import math

import torch
from .Module import Module
from .utils import clear, contiguousView


class CMul(Module):

    def __init__(self, *args):
        super(CMul, self).__init__()

        if len(args) == 1 and isinstance(args[0], torch.Size):
            self.size = args[0]
        else:
            self.size = torch.Size(args)

        self.weight = torch.Tensor(self.size)
        self.gradWeight = torch.Tensor(self.size)
        self.output.resize_(self.size)
        self.reset()

        self._output = None
        self._weight = None
        self._expand = None
        self._repeat = None
        self._gradOutput = None
        self._gradInput = None
        self._input = None
        self._gradWeight = None
        self._sum = None

    def reset(self, stdv=None):
        if stdv is not None:
            stdv = stdv * math.sqrt(3)
        else:
            stdv = 1. / math.sqrt(self.weight.nelement())

        self.weight.uniform_(-stdv, stdv)

    def updateOutput(self, input):
        # lazy-initialize
        if self._output is None:
            self._output = input.new()
            self._weight = input.new()
            self._expand = input.new()
            self._repeat = input.new()

        self.output.resize_as_(input).copy_(input)
        batchSize = input.size(0)
        # TODO: expand_as_, view_
        self._output = self.output.view(batchSize, -1)
        self._weight = self.weight.view(1, -1)
        self._expand = self._weight.expand_as(self._output)

        if torch.typename(input) == 'torch.cuda.FloatTensor':
            self._repeat.resize_as_(self._expand).copy_(self._expand)
            self._output.mul_(self._repeat)
        else:
            self._output.mul_(self._expand)

        return self.output

    def updateGradInput(self, input, gradOutput):
        if self.gradInput is None:
            return

        if self._gradOutput is None:
            self._gradOutput = input.new()
            self._gradInput = input.new()

        self.gradInput.resize_as_(input).zero_()
        batchSize = input.size(0)
        contiguousView(self._gradOutput, gradOutput, batchSize, -1)
        contiguousView(self._gradInput, self.gradInput, batchSize, -1)
        self._weight = self.weight.view(1, -1)
        self._expand = self._weight.expand_as(self._gradOutput)

        if torch.typename(input) == 'torch.cuda.FloatTensor':
            self._repeat.resize_as_(self._expand).copy_(self._expand)
            self._gradInput.addcmul_(1, self._repeat, self._gradOutput)
        else:
            self._gradInput.addcmul_(1, self._expand, self._gradOutput)

        return self.gradInput

    def accGradParameters(self, input, gradOutput, scale=1):
        if self._input is None:
            self._input = input.new()
            self._gradWeight = input.new()
            self._sum = input.new()

        batchSize = input.size(0)
        contiguousView(self._input, input, batchSize, -1)
        contiguousView(self._gradOutput, gradOutput, batchSize, -1)
        self._gradWeight = self.gradWeight.view(1, -1)

        torch.mul(self._input, self._gradOutput, out=self._repeat)
        torch.sum(self._repeat, 0, out=self._sum)
        self._gradWeight.add_(scale, self._sum)

    def type(self, type=None, tensorCache=None):
        if type:
            self.clearState()
        return super(CMul, self).type(type, tensorCache)

    def clearState(self):
        clear(self, [
            '_input',
            '_output',
            '_weight',
            '_gradWeight',
            '_expand',
            '_repeat',
            '_sum',
        ])
        return super(CMul, self).clearState()
