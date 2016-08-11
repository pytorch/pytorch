import math
import torch
from torch.legacy import nn

class WeightedEuclidean(nn.Module):

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
        if src.isContiguous():
           res.set_(src.view(*args))
        else:
           res.set_(src.contiguous().view(*args))

    def updateOutput(self, input):
        # lazy-initialize
        self._diagCov = self._diagCov or self.output.new()

        self._input = self._input or input.new()
        self._weight = self._weight or self.weight.new()
        self._expand = self._expand or self.output.new()
        self._expand2 = self._expand or self.output.new()
        self._expand3 = self._expand3 or self.output.new()
        self._repeat = self._repeat or self.output.new()
        self._repeat2 = self._repeat2 or self.output.new()
        self._repeat3 = self._repeat3 or self.output.new()

        inputSize, outputSize = self.weight.size(0), self.weight.size(1)

        # y_j = || c_j * (w_j - x) ||
        if input.dim() == 1:
            self._view(self._input, input, inputSize, 1)
            self._expand.expandAs(self._input, self.weight)
            self._repeat.resizeAs_(self._expand).copy(self._expand)
            self._repeat.add_(-1, self.weight)
            self._repeat.mul_(self.diagCov)
            torch.norm(self.output, self._repeat, 2, 0)
            self.output.resize_(outputSize)
        elif input.dim() == 2:
            batchSize = input.size(0)

            self._view(self._input, input, batchSize, inputSize, 1)
            self._expand = self._input.expand(batchSize, inputSize, outputSize)
            # make the expanded tensor contiguous (requires lots of memory)
            self._repeat.resizeAs_(self._expand).copy(self._expand)

            self._weight = self.weight.view(1, inputSize, outputSize)
            self._expand2 = self._weight.expandAs(self._repeat)

            self._diagCov = self.diagCov.view(1, inputSize, outputSize)
            self._expand3 = self._diagCov.expandAs(self._repeat)
            if input.type() == 'torch.cuda.FloatTensor':
                # TODO: this can be fixed with a custom allocator
                # requires lots of memory, but minimizes cudaMallocs and loops
                self._repeat2.resizeAs_(self._expand2).copy(self._expand2)
                self._repeat.add_(-1, self._repeat2)
                self._repeat3.resizeAs_(self._expand3).copy(self._expand3)
                self._repeat.mul_(self._repeat3)
            else:
                self._repeat.add_(-1, self._expand2)
                self._repeat.mul_(self._expand3)


            torch.norm(self.output, self._repeat, 2, 1)
            self.output.resize_(batchSize, outputSize)
        else:
           raise RuntimeError("1D or 2D input expected")

        return self.output

    def updateGradInput(self, input, gradOutput):
        if not self.gradInput:
           return

        self._div = self._div or input.new()
        self._output = self._output or self.output.new()
        self._expand4 = self._expand4 or input.new()
        self._gradOutput = self._gradOutput or input.new()

        if not self.fastBackward:
           self.updateOutput(input)

        inputSize, outputSize = self.weight.size(0), self.weight.size(1)

        """
        dy_j   -2 * c_j * c_j * (w_j - x)   c_j * c_j * (x - w_j)
        ---- = -------------------------- = ---------------------
         dx     2 || c_j * (w_j - x) ||              y_j
        """

        # to prevent div by zero (NaN) bugs
        self._output.resizeAs_(self.output).copy(self.output).add_(1e-7)
        self._view(self._gradOutput, gradOutput, gradOutput.size())
        torch.div(self._div, gradOutput, self._output)
        if input.dim() == 1:
            self._div.resize_(1, outputSize)
            self._expand4 = self._div.expandAs(self.weight)

            if torch.type(input) == 'torch.cuda.FloatTensor':
                self._repeat2.resizeAs_(self._expand4).copy(self._expand4)
                self._repeat2.mul_(self._repeat)
            else:
                self._repeat2.mul_(self._repeat, self._expand4)

            self._repeat2.mul_(self.diagCov)
            torch.sum(self.gradInput, self._repeat2, 1)
            self.gradInput.resizeAs_(input)
        elif input.dim() == 2:
            batchSize = input.size(0)

            self._div.resize_(batchSize, 1, outputSize)
            self._expand4 = self._div.expand(batchSize, inputSize, outputSize)

            if input.type() == 'torch.cuda.FloatTensor':
                self._repeat2.resizeAs_(self._expand4).copy(self._expand4)
                self._repeat2.mul_(self._repeat)
                self._repeat2.mul_(self._repeat3)
            else:
                torch.mul(self._repeat2, self._repeat, self._expand4)
                self._repeat2.mul_(self._expand3)


            torch.sum(self.gradInput, self._repeat2, 2)
            self.gradInput.resizeAs_(input)
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
                self._repeat2.resizeAs_(self._expand4).copy(self._expand4)
                self._repeat2.mul_(self._repeat)
            else:
                torch.mul(self._repeat2, self._repeat, self._expand4)


            self.gradDiagCov.add_(self._repeat2)
        elif input.dim() == 2:
            self._sum = self._sum or input.new()
            torch.sum(self._sum, self._repeat2, 0)
            self._sum.resize_(inputSize, outputSize)
            self.gradWeight.add_(-scale, self._sum)

            if input.type() == 'torch.cuda.FloatTensor':
                # requires lots of memory, but minimizes cudaMallocs and loops
                self._repeat.div_(self._repeat3)
                self._repeat.mul_(self._repeat)
                self._repeat.mul_(self._repeat3)
                self._repeat2.resizeAs_(self._expand4).copy(self._expand4)
                self._repeat.mul_(self._repeat2)
            else:
                self._repeat.div_(self._expand3)
                self._repeat.mul_(self._repeat)
                self._repeat.mul_(self._expand3)
                self._repeat.mul_(self._expand4)


            torch.sum(self._sum, self._repeat, 0)
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

