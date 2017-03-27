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

        self._input = None
        self._weight = None
        self._expand = None
        self._expand2 = None
        self._repeat = None
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
            stdv = 1. / math.sqrt(self.weight.size(0))

        self.weight.uniform_(-stdv, stdv)

    def _view(self, res, src, *args):
        if src.is_contiguous():
            res.set_(src.view(*args))
        else:
            res.set_(src.contiguous().view(*args))

    def updateOutput(self, input):
        # lazy initialize buffers
        if self._input is None:
            self._input = input.new()
        if self._weight is None:
            self._weight = self.weight.new()
        if self._expand is None:
            self._expand = self.output.new()
        if self._expand2 is None:
            self._expand2 = self.output.new()
        if self._repeat is None:
            self._repeat = self.output.new()
        if self._repeat2 is None:
            self._repeat2 = self.output.new()

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

        torch.norm(self._repeat, 2, 1, out=self.output)
        self.output.resize_(batchSize, outputSize)

        return self.output

    def updateGradInput(self, input, gradOutput):
        if self.gradInput is None:
            return

        if self._div is None:
            self._div = input.new()
        if self._output is None:
            self._output = self.output.new()
        if self._gradOutput is None:
            self._gradOutput = input.new()
        if self._expand3 is None:
            self._expand3 = input.new()

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
        torch.div(gradOutput, self._output, out=self._div)
        assert input.dim() == 2
        batchSize = input.size(0)

        self._div.resize_(batchSize, 1, outputSize)
        self._expand3 = self._div.expand(batchSize, inputSize, outputSize)

        if torch.typename(input) == 'torch.cuda.FloatTensor':
            self._repeat2.resize_as_(self._expand3).copy_(self._expand3)
            self._repeat2.mul_(self._repeat)
        else:
            torch.mul(self._repeat, self._expand3, out=self._repeat2)

        torch.sum(self._repeat2, 2, out=self.gradInput)
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
        if self._sum is None:
            self._sum = input.new()
        torch.sum(self._repeat2, 0, out=self._sum)
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
