import math
import torch
from .Module import Module
from .utils import clear


class Cosine(Module):

    def __init__(self, inputSize, outputSize):
        super(Cosine, self).__init__()
        self.weight = torch.Tensor(outputSize, inputSize)
        self.gradWeight = torch.Tensor(outputSize, inputSize)
        self.reset()

        self._weight = None
        self._sum = None
        self._gradOutput = None
        self._sum = None
        self._weightNorm = None
        self._inputNorm = None

    def reset(self, stdv=None):
        if stdv is not None:
            stdv = stdv * math.sqrt(3)
        else:
            stdv = 1. / math.sqrt(self.weight.size(0))
        self.weight.uniform_(-stdv, stdv)

    def updateOutput(self, input):
        assert input.dim() == 2

        inputSize = self.weight.size(1)
        outputSize = self.weight.size(0)

        if self._weightNorm is None:
            self._weightNorm = self.weight.new()
        if self._inputNorm is None:
            self._inputNorm = self.weight.new()

        # y_j = (w_j * x) / ( || w_j || * || x || )

        torch.norm(self.weight, 2, 1, out=self._weightNorm, keepdim=True).add_(1e-12)

        batchSize = input.size(0)
        nelement = self.output.nelement()
        self.output.resize_(batchSize, outputSize)
        if self.output.nelement() != nelement:
            self.output.zero_()

        self.output.addmm_(0., 1., input, self.weight.t())

        torch.norm(input, 2, 1, out=self._inputNorm, keepdim=True).add_(1e-12)
        self.output.div_(self._weightNorm.view(1, outputSize).expand_as(self.output))
        self.output.div_(self._inputNorm.expand_as(self.output))
        return self.output

    def updateGradInput(self, input, gradOutput):
        assert input.dim() == 2

        if self.gradInput is None:
            return

        inputSize = self.weight.size(1)
        outputSize = self.weight.size(0)

        """
        dy_j           w_ji                   x_i
        ---- = -------------------  -  y_j ---------
        dx_i   || w_j || * || x ||         || x ||^2
        """

        nelement = self.gradInput.nelement()
        self.gradInput.resize_as_(input)
        if self.gradInput.nelement() != nelement:
            self.gradInput.zero_()

        inputNorm = self._inputNorm.expand_as(input)
        weightNorm = self._weightNorm.view(1, outputSize).expand_as(gradOutput)

        if self._gradOutput is None:
            self._gradOutput = gradOutput.new()
        if self._sum is None:
            self._sum = input.new()

        self.gradInput.copy_(input).div_(inputNorm)
        self._gradOutput.resize_as_(gradOutput).copy_(gradOutput)
        self._gradOutput.mul_(self.output)
        torch.sum(self._gradOutput, 1, out=self._sum, keepdim=True)
        self.gradInput.mul_(self._sum.expand_as(input))

        self._gradOutput.resize_as_(gradOutput).copy_(gradOutput)
        self._gradOutput.div_(weightNorm)
        self.gradInput.addmm_(-1, 1, self._gradOutput, self.weight)
        self.gradInput.div_(inputNorm)

        return self.gradInput

    def accGradParameters(self, input, gradOutput, scale=1):
        assert input.dim() == 2
        inputSize = self.weight.size(1)
        outputSize = self.weight.size(0)

        """
        dy_j            x_i                     w_ji
        ----- = -------------------  -  y_j -----------
        dw_ji   || w_j || * || x ||         || w_j ||^2
        """

        if self._weight is None:
            self._weight = self.weight.new()
        if self._sum is None:
            self._sum = input.new()

        self._weight.resize_as_(self.weight).copy_(self.weight)
        if self._gradOutput is None:
            self._gradOutput = gradOutput.new()
        self._gradOutput.resize_as_(gradOutput).copy_(gradOutput)
        self._gradOutput.mul_(self.output)
        torch.sum(self._gradOutput, 0, out=self._sum, keepdim=True)
        grad = self._sum[0]
        grad.div_(self._weightNorm.select(1, 0))
        self._weight.mul_(grad.view(outputSize, 1).expand_as(self._weight))

        input_ = self._gradOutput
        input_.resize_as_(input).copy_(input)
        input_.div_(self._inputNorm.expand_as(input))
        self._weight.addmm_(-1, 1, gradOutput.t(), input_)

        self._weight.div_(self._weightNorm.expand_as(self._weight))
        self.gradWeight.add_(self._weight)

    def type(self, type=None, tensorCache=None):
        if type is not None:
            # prevent premature memory allocations
            self._input = None
            self._weight = None
            self._inputNorm = None
            self._weightNorm = None
            self._gradOutput = None
            self._sum = None

        return super(Cosine, self).type(type, tensorCache)

    def clearState(self):
        clear(self, [
            '_input',
            '_weight',
            '_gradOutput',
            '_sum',
            '_inputNorm',
            '_weightNorm',
        ])
        return super(Cosine, self).clearState()
