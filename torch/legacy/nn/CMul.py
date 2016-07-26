import math

import torch
from torch.legacy import nn

class CMul(nn.Module):

    def __init__(self, *args):
        super(CMul, self).__init__()

        self.size = torch.LongStorage()
        if len(args) == 1 and torch.type(args[0]) == 'torch.LongStorage':
            self.size.resize(arg[0].size()).copy(arg[0])
        else:
            self.size.resize(len(args))
            for i, arg in enumerate(args):
                    self.size[i] = arg

        self.weight = torch.Tensor(self.size)
        self.gradWeight = torch.Tensor(self.size)
        self.output.resize(self.size)
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
            stdv = 1./math.sqrt(self.weight.nElement())

        self.weight.uniform(-stdv, stdv)


    def updateOutput(self, input):
        # lazy-initialize
        if self._output is None:
            self._output = input.new()
            self._weight = input.new()
            self._expand = input.new()
            self._repeat = input.new()

        self.output.resizeAs(input).copy(input)
        batchSize = input.size(0)
        self._output.view(self.output, batchSize, -1)
        self._weight.view(self.weight, 1, -1)
        self._expand.expandAs(self._weight, self._output)

        # TODO: verify
        if torch.typename(input) == 'torch.CudaTensor':
            self._repeat.resizeAs(self._expand).copy(self._expand)
            self._output.cmul(self._repeat)
        else:
            self._output.cmul(self._expand)

        return self.output


    def updateGradInput(self, input, gradOutput):
        if not self.gradInput:
           return

        if self._gradOutput is None:
            self._gradOutput = input.new()
            self._gradInput = input.new()

        self.gradInput.resizeAs(input).zero()
        batchSize = input.size(0)
        nn.utils.contiguousView(self._gradOutput, gradOutput, batchSize, -1)
        nn.utils.contiguousView(self._gradInput, self.gradInput, batchSize, -1)
        self._weight.view(self.weight, 1, -1)
        self._expand.expandAs(self._weight, self._gradOutput)

        if torch.typename(input) == 'torch.CudaTensor':
            self._repeat.resizeAs(self._expand).copy(self._expand)
            self._gradInput.addcmul(1, self._repeat, self._gradOutput)
        else:
            self._gradInput.addcmul(1, self._expand, self._gradOutput)

        return self.gradInput


    def accGradParameters(self, input, gradOutput, scale=1):
        if self._input is None:
            self._input = input.new()
            self._gradWeight = input.new()
            self._sum = input.new()

        batchSize = input.size(0)
        nn.utils.contiguousView(self._input, input, batchSize, -1)
        nn.utils.contiguousView(self._gradOutput, gradOutput, batchSize, -1)
        self._gradWeight.view(self.gradWeight, 1, -1)

        self._repeat.cmul(self._input, self._gradOutput)
        self._sum.sum(self._repeat, 0)
        self._gradWeight.add(scale, self._sum)

    def type(self, type=None, tensorCache=None):
        if type:
           self.clearState()
        return super(CMul, self).type(self, type, tensorCache)

    def clearState(self):
        nn.utils.clear(self, [
           '_input',
           '_output',
           '_weight',
           '_gradWeight',
           '_expand',
           '_repeat',
           '_sum',
        ])
        return super(CMult, self).clearState(self)

