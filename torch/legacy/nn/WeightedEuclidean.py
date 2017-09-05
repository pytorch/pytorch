import math
import torch
from .Module import Module


class WeightedEuclidean(Module):

    def __init__(self, inputSize, outputSize):
        super(WeightedEuclidean, self).__init__()

        self.weight = torch.Tensor(inputSize, outputSize)
        self.gradWeight = torch.Tensor(inputSize, outputSize)

        # each template (output dim) has its own diagonal covariance matrix
        self.diagCov = torch.Tensor(inputSize, outputSize)
        self.gradDiagCov = torch.Tensor(inputSize, outputSize)

        self.reset()
        self._diagCov = self.output.new()

        # TODO: confirm
        self.fastBackward = False

        self._input = None
        self._weight = None
        self._expand = None
        self._expand2 = None
        self._expand3 = None
        self._repeat = None
        self._repeat2 = None
        self._repeat3 = None
        self._div = None
        self._output = None
        self._expand4 = None
        self._gradOutput = None
        self._sum = None

    def reset(self, stdv=None):
        if stdv is not None:
            stdv = stdv * math.sqrt(3)
        else:
            stdv = 1. / math.sqrt(self.weight.size(1))

        self.weight.uniform_(-stdv, stdv)
        self.diagCov.fill_(1)

    def _view(self, res, src, *args):
        if src.is_contiguous():
            res.set_(src.view(*args))
        else:
            res.set_(src.contiguous().view(*args))

    def updateOutput(self, input):
        # lazy-initialize
        if self._diagCov is None:
            self._diagCov = self.output.new()

        if self._input is None:
            self._input = input.new()
        if self._weight is None:
            self._weight = self.weight.new()
        if self._expand is None:
            self._expand = self.output.new()
        if self._expand2 is None:
            self._expand2 = self.output.new()
        if self._expand3 is None:
            self._expand3 = self.output.new()
        if self._repeat is None:
            self._repeat = self.output.new()
        if self._repeat2 is None:
            self._repeat2 = self.output.new()
        if self._repeat3 is None:
            self._repeat3 = self.output.new()

        inputSize, outputSize = self.weight.size(0), self.weight.size(1)

        # y_j = || c_j * (w_j - x) ||
        if input.dim() == 1:
            self._view(self._input, input, inputSize, 1)
            self._expand.expand_as(self._input, self.weight)
            self._repeat.resize_as_(self._expand).copy_(self._expand)
            self._repeat.add_(-1, self.weight)
            self._repeat.mul_(self.diagCov)
            torch.norm(self._repeat, 2, 0, True, out=self.output)
            self.output.resize_(outputSize)
        elif input.dim() == 2:
            batchSize = input.size(0)

            self._view(self._input, input, batchSize, inputSize, 1)
            self._expand = self._input.expand(batchSize, inputSize, outputSize)
            # make the expanded tensor contiguous (requires lots of memory)
            self._repeat.resize_as_(self._expand).copy_(self._expand)

            self._weight = self.weight.view(1, inputSize, outputSize)
            self._expand2 = self._weight.expand_as(self._repeat)

            self._diagCov = self.diagCov.view(1, inputSize, outputSize)
            self._expand3 = self._diagCov.expand_as(self._repeat)
            if input.type() == 'torch.cuda.FloatTensor':
                # TODO: this can be fixed with a custom allocator
                # requires lots of memory, but minimizes cudaMallocs and loops
                self._repeat2.resize_as_(self._expand2).copy_(self._expand2)
                self._repeat.add_(-1, self._repeat2)
                self._repeat3.resize_as_(self._expand3).copy_(self._expand3)
                self._repeat.mul_(self._repeat3)
            else:
                self._repeat.add_(-1, self._expand2)
                self._repeat.mul_(self._expand3)

            torch.norm(self._repeat, 2, 1, True, out=self.output)
            self.output.resize_(batchSize, outputSize)
        else:
            raise RuntimeError("1D or 2D input expected")

        return self.output

    def updateGradInput(self, input, gradOutput):
        if self.gradInput is None:
            return

        if self._div is None:
            self._div = input.new()
        if self._output is None:
            self._output = self.output.new()
        if self._expand4 is None:
            self._expand4 = input.new()
        if self._gradOutput is None:
            self._gradOutput = input.new()

        if not self.fastBackward:
            self.updateOutput(input)

        inputSize, outputSize = self.weight.size(0), self.weight.size(1)

        """
        dy_j   -2 * c_j * c_j * (w_j - x)   c_j * c_j * (x - w_j)
        ---- = -------------------------- = ---------------------
         dx     2 || c_j * (w_j - x) ||              y_j
        """

        # to prevent div by zero (NaN) bugs
        self._output.resize_as_(self.output).copy_(self.output).add_(1e-7)
        self._view(self._gradOutput, gradOutput, gradOutput.size())
        torch.div(gradOutput, self._output, out=self._div)
        if input.dim() == 1:
            self._div.resize_(1, outputSize)
            self._expand4 = self._div.expand_as(self.weight)

            if torch.type(input) == 'torch.cuda.FloatTensor':
                self._repeat2.resize_as_(self._expand4).copy_(self._expand4)
                self._repeat2.mul_(self._repeat)
            else:
                self._repeat2.mul_(self._repeat, self._expand4)

            self._repeat2.mul_(self.diagCov)
            torch.sum(self._repeat2, 1, True, out=self.gradInput)
            self.gradInput.resize_as_(input)
        elif input.dim() == 2:
            batchSize = input.size(0)

            self._div.resize_(batchSize, 1, outputSize)
            self._expand4 = self._div.expand(batchSize, inputSize, outputSize)

            if input.type() == 'torch.cuda.FloatTensor':
                self._repeat2.resize_as_(self._expand4).copy_(self._expand4)
                self._repeat2.mul_(self._repeat)
                self._repeat2.mul_(self._repeat3)
            else:
                torch.mul(self._repeat, self._expand4, out=self._repeat2)
                self._repeat2.mul_(self._expand3)

            torch.sum(self._repeat2, 2, True, out=self.gradInput)
            self.gradInput.resize_as_(input)
        else:
            raise RuntimeError("1D or 2D input expected")

        return self.gradInput

    def accGradParameters(self, input, gradOutput, scale=1):
        inputSize, outputSize = self.weight.size(0), self.weight.size(1)

        """
        dy_j   2 * c_j * c_j * (w_j - x)    c_j * c_j * (w_j - x)
        ---- = -------------------------- = ---------------------
        dw_j    2 || c_j * (w_j - x) ||             y_j

        dy_j    2 * c_j * (w_j - x)^2    c_j * (w_j - x)^2
        ---- = ----------------------- = -----------------
        dc_j   2 || c_j * (w_j - x) ||         y_j
        #"""
        # assumes a preceding call to updateGradInput
        if input.dim() == 1:
            self.gradWeight.add_(-scale, self._repeat2)

            self._repeat.div_(self.diagCov)
            self._repeat.mul_(self._repeat)
            self._repeat.mul_(self.diagCov)

            if torch.type(input) == 'torch.cuda.FloatTensor':
                self._repeat2.resize_as_(self._expand4).copy_(self._expand4)
                self._repeat2.mul_(self._repeat)
            else:
                torch.mul(self._repeat, self._expand4, out=self._repeat2)

            self.gradDiagCov.add_(self._repeat2)
        elif input.dim() == 2:
            if self._sum is None:
                self._sum = input.new()
            torch.sum(self._repeat2, 0, True, out=self._sum)
            self._sum.resize_(inputSize, outputSize)
            self.gradWeight.add_(-scale, self._sum)

            if input.type() == 'torch.cuda.FloatTensor':
                # requires lots of memory, but minimizes cudaMallocs and loops
                self._repeat.div_(self._repeat3)
                self._repeat.mul_(self._repeat)
                self._repeat.mul_(self._repeat3)
                self._repeat2.resize_as_(self._expand4).copy_(self._expand4)
                self._repeat.mul_(self._repeat2)
            else:
                self._repeat.div_(self._expand3)
                self._repeat.mul_(self._repeat)
                self._repeat.mul_(self._expand3)
                self._repeat.mul_(self._expand4)

            torch.sum(self._repeat, 0, True, out=self._sum)
            self._sum.resize_(inputSize, outputSize)
            self.gradDiagCov.add_(scale, self._sum)
        else:
            raise RuntimeError("1D or 2D input expected")

    def type(self, type=None, tensorCache=None):
        if type:
            # prevent premature memory allocations
            self._input = None
            self._output = None
            self._gradOutput = None
            self._weight = None
            self._div = None
            self._sum = None
            self._expand = None
            self._expand2 = None
            self._expand3 = None
            self._expand4 = None
            self._repeat = None
            self._repeat2 = None
            self._repeat3 = None
        return super(WeightedEuclidean, self).type(type, tensorCache)

    def parameters(self):
        return [self.weight, self.diagCov], [self.gradWeight, self.gradDiagCov]

    def accUpdateGradParameters(self, input, gradOutput, lr):
        gradWeight = self.gradWeight
        gradDiagCov = self.gradDiagCov
        self.gradWeight = self.weight
        self.gradDiagCov = self.diagCov
        self.accGradParameters(input, gradOutput, -lr)
        self.gradWeight = gradWeight
        self.gradDiagCov = gradDiagCov
