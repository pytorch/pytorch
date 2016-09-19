import math
import torch
from .Module import Module
from .utils import clear

class Euclidean(Module):

    def __init__(self, inputSize, outputSize):
        super(Euclidean, self).__init__()

        self.weight = torch.Tensor(inputSize, outputSize)
        self.gradWeight = torch.Tensor(inputSize, outputSize)

        # state
        self.gradInput.resize_(inputSize)
        self.output.resize_(outputSize)

        self.fastBackward = True
        self.reset()

        self._input   = None
        self._weight  = None
        self._expand  = None
        self._expand2 = None
        self._repeat  = None
        self._repeat2 = None
        self._div = None
        self._output = None
        self._gradOutput = None
        self._expand3 = None
        self._sum = None

    def reset(self, stdv=None):
        if stdv is not None:
           stdv = stdv * math.sqrt(3)
        else:
           stdv = 1./math.sqrt(self.weight.size(0))

        self.weight.uniform_(-stdv, stdv)

    def _view(self, res, src, *args):
        if src.is_contiguous():
           res.set_(src.view(*args))
        else:
           res.set_(src.contiguous().view(*args))

    def updateOutput(self, input):
        # lazy initialize buffers
        self._input = self._input or input.new()
        self._weight = self._weight or self.weight.new()
        self._expand = self._expand or self.output.new()
        self._expand2 = self._expand2 or self.output.new()
        self._repeat = self._repeat or self.output.new()
        self._repeat2 = self._repeat2 or self.output.new()

        inputSize, outputSize = self.weight.size(0), self.weight.size(1)

        # y_j = || w_j - x || = || x - w_j ||
        assert input.dim() == 2

        batchSize = input.size(0)
        self._view(self._input, input, batchSize, inputSize, 1)
        self._expand = self._input.expand(batchSize, inputSize, outputSize)
        # make the expanded tensor contiguous (requires lots of memory)
        self._repeat.resize_as_(self._expand).copy_(self._expand)

        self._weight = self.weight.view(1, inputSize, outputSize)
        self._expand2 = self._weight.expand_as(self._repeat)

        if torch.typename(input) == 'torch.cuda.FloatTensor':
            # TODO: after adding new allocators this can be changed
            # requires lots of memory, but minimizes cudaMallocs and loops
            self._repeat2.resize_as_(self._expand2).copy_(self._expand2)
            self._repeat.add_(-1, self._repeat2)
        else:
            self._repeat.add_(-1, self._expand2)

        torch.norm(self.output, self._repeat, 2, 1)
        self.output.resize_(batchSize, outputSize)

        return self.output

    def updateGradInput(self, input, gradOutput):
        if not self.gradInput:
           return

        self._div = self._div or input.new()
        self._output = self._output or self.output.new()
        self._gradOutput = self._gradOutput or input.new()
        self._expand3 = self._expand3 or input.new()

        if not self.fastBackward:
           self.updateOutput(input)

        inputSize, outputSize = self.weight.size(0), self.weight.size(1)

        """
        dy_j   -2 * (w_j - x)     x - w_j
        ---- = ---------------- = -------
         dx    2 || w_j - x ||      y_j
        """

        # to prevent div by zero (NaN) bugs
        self._output.resize_as_(self.output).copy_(self.output).add_(0.0000001)
        self._view(self._gradOutput, gradOutput, gradOutput.size())
        torch.div(self._div, gradOutput, self._output)
        assert input.dim() == 2
        batchSize = input.size(0)

        self._div.resize_(batchSize, 1, outputSize)
        self._expand3 = self._div.expand(batchSize, inputSize, outputSize)

        if torch.typename(input) == 'torch.cuda.FloatTensor':
            self._repeat2.resize_as_(self._expand3).copy_(self._expand3)
            self._repeat2.mul_(self._repeat)
        else:
            torch.mul(self._repeat2, self._repeat, self._expand3)


        torch.sum(self.gradInput, self._repeat2, 2)
        self.gradInput.resize_as_(input)

        return self.gradInput


    def accGradParameters(self, input, gradOutput, scale=1):
        inputSize, outputSize = self.weight.size(0), self.weight.size(1)

        """
        dy_j    2 * (w_j - x)    w_j - x
        ---- = --------------- = -------
        dw_j   2 || w_j - x ||     y_j
        """
        # assumes a preceding call to updateGradInput
        assert input.dim() == 2
        self._sum = self._sum or input.new()
        torch.sum(self._sum, self._repeat2, 0)
        self._sum.resize_(inputSize, outputSize)
        self.gradWeight.add_(-scale, self._sum)

    def type(self, type=None, tensorCache=None):
        if type:
           # prevent premature memory allocations
           self.clearState()

        return super(Euclidean, self).type(type, tensorCache)


    def clearState(self):
        clear(self, [
           '_input',
           '_output',
           '_gradOutput',
           '_weight',
           '_div',
           '_sum',
           '_expand',
           '_expand2',
           '_expand3',
           '_repeat',
           '_repeat2',
        ])
        return super(Euclidean, self).clearState()

