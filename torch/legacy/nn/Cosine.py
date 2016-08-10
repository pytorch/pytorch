import math
import torch
from torch.legacy import nn

class Cosine(nn.Module):

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
            stdv = 1./math.sqrt(self.weight.size(0))
        self.weight.uniform_(-stdv, stdv)

    def updateOutput(self, input):
        assert input.dim() == 2

        inputSize = self.weight.size(1)
        outputSize = self.weight.size(0)

        self._weightNorm = self._weightNorm or self.weight.new()
        self._inputNorm = self._inputNorm or self.weight.new()

        # y_j = (w_j * x) / ( || w_j || * || x || )

        torch.norm(self._weightNorm, self.weight, 2, 1).add_(1e-12)

        batchSize = input.size(0)
        nElement = self.output.nElement()
        self.output.resize_(batchSize, outputSize)
        if self.output.nElement() != nElement:
            self.output.zero_()

        self.output.addmm_(0., 1., input, self.weight.t())

        torch.norm(self._inputNorm, input, 2, 1).add_(1e-12)
        self.output.div_(self._weightNorm.view(1, outputSize).expandAs(self.output))
        self.output.div_(self._inputNorm.expandAs(self.output))
        return self.output


    def updateGradInput(self, input, gradOutput):
        assert input.dim() == 2

        if not self.gradInput:
           return

        inputSize = self.weight.size(1)
        outputSize = self.weight.size(0)

        """
        dy_j           w_ji                   x_i
        ---- = -------------------  -  y_j ---------
        dx_i   || w_j || * || x ||         || x ||^2
        """

        nElement = self.gradInput.nElement()
        self.gradInput.resizeAs_(input)
        if self.gradInput.nElement() != nElement:
           self.gradInput.zero_()

        inputNorm = self._inputNorm.expandAs(input)
        weightNorm = self._weightNorm.view(1, outputSize).expandAs(gradOutput)

        self._gradOutput = self._gradOutput or gradOutput.new()
        self._sum = self._sum or input.new()

        self.gradInput.copy(input).div_(inputNorm)
        self._gradOutput.resizeAs_(gradOutput).copy(gradOutput)
        self._gradOutput.mul_(self.output)
        torch.sum(self._sum, self._gradOutput, 1)
        self.gradInput.mul_(self._sum.expandAs(input))

        self._gradOutput.resizeAs_(gradOutput).copy(gradOutput)
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

        self._weight = self._weight or self.weight.new()
        self._sum = self._sum or input.new()

        self._weight.resizeAs_(self.weight).copy(self.weight)
        self._gradOutput = self._gradOutput or gradOutput.new()
        self._gradOutput.resizeAs_(gradOutput).copy(gradOutput)
        self._gradOutput.mul_(self.output)
        torch.sum(self._sum, self._gradOutput, 0)
        grad = self._sum[0]
        grad.div_(self._weightNorm.select(1, 0))
        self._weight.mul_(grad.view(outputSize, 1).expandAs(self._weight))

        input_ = self._gradOutput
        input_.resizeAs_(input).copy(input)
        input_.div_(self._inputNorm.expandAs(input))
        self._weight.addmm_(-1, 1, gradOutput.t(), input_)

        self._weight.div_(self._weightNorm.expandAs(self._weight))
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
        nn.utils.clear(self, [
           '_input',
           '_weight',
           '_gradOutput',
           '_sum',
           '_inputNorm',
           '_weightNorm',
        ])
        return super(Cosine, self).clearState()

