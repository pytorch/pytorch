import math
import torch
from .Module import Module
from .utils import clear

class Bilinear(Module):

    def _assertInput(self, input):
        if len(input) != 2 or not torch.isTensor(input[0]) or not torch.isTensor(input[1]):
            raise RuntimeError('input should be a table containing two data Tensors')
        if input[0].nDimension() != 2 or input[1].nDimension() != 2:
            raise RuntimeError('input Tensors should be two-dimensional')
        if input[0].size(0) != input[1].size(0):
            raise RuntimeError('input Tensors should have the same number of rows')
        if input[0].size(1) != self.weight.size(1):
            raise RuntimeError('dimensionality of first input is erroneous')
        if input[1].size(1) != self.weight.size(2):
            raise RuntimeError('dimensionality of second input is erroneous')

    def _assertInputGradOutput(self, input, gradOutput):
        if input[0].size(0) != gradOutput.size(0):
            raise RuntimeError('number of rows in gradOutput.es not match input')
        if gradOutput.size(1) != self.weight.size(0):
            raise RuntimeError('number of columns in gradOutput does not match layer\'s output size')


    def __init__(self, inputSize1, inputSize2, outputSize, bias=True):
        # set up model:
        super(Bilinear, self).__init__()
        self.weight     = torch.Tensor(outputSize, inputSize1, inputSize2)
        self.gradWeight = torch.Tensor(outputSize, inputSize1, inputSize2)
        if bias:
            self.bias     = torch.Tensor(outputSize)
            self.gradBias = torch.Tensor(outputSize)
        else:
            self.bias = None
            self.gradBias = None

        self.buff1 = None
        self.buff2 = None

        self.gradInput = [torch.Tensor(), torch.Tensor()]
        self.reset()

    def reset(self, stdv=None):
        if stdv is not None:
            stdv = stdv * math.sqrt(3)
        else:
            stdv = 1. / math.sqrt(self.weight.size(1))

        self.weight.uniform_(-stdv, stdv)
        if self.bias:
            self.bias.uniform_(-stdv, stdv)
        return self


    def updateOutput(self, input):
        self._assertInput(input)

        # set up buffer:
        self.buff2 = self.buff2 or input[0].new()
        self.buff2.resizeAs_(input[1])

        # compute output scores:
        self.output.resize_(input[0].size(0), self.weight.size(0))
        for k in range(self.weight.size(0)):
            torch.mm(self.buff2, input[0], self.weight[k])
            self.buff2.mul_(input[1])
            torch.sum(self.output.narrow(1, k, 1), self.buff2, 1)

        if self.bias:
            self.output.add_(self.bias.view(1, self.bias.nElement()).expandAs(self.output))

        return self.output


    def updateGradInput(self, input, gradOutput):
        if not self.gradInput:
            return

        self._assertInputGradOutput(input, gradOutput)
        # compute d output / d input:
        self.gradInput[0].resizeAs_(input[0]).fill_(0)
        self.gradInput[1].resizeAs_(input[1]).fill_(0)

        #: first slice of weight tensor (k = 1)
        self.gradInput[0].addmm_(input[1], self.weight[0].t())
        self.gradInput[0].mul_(gradOutput.narrow(1, 0, 1).expand(self.gradInput[0].size(0),
            self.gradInput[0].size(1)))
        self.gradInput[1].addmm_(input[0], self.weight[0])
        self.gradInput[1].mul_(gradOutput.narrow(1, 0, 1).expand(self.gradInput[1].size(0),
            self.gradInput[1].size(1)))

        #: remaining slices of weight tensor
        if self.weight.size(0) > 1:
            self.buff1 = self.buff1 or input[0].new()
            self.buff1.resizeAs_(input[0])

            for k in range(1, self.weight.size(0)):
                torch.mm(self.buff1, input[1], self.weight[k].t())
                self.buff1.mul_(gradOutput.narrow(1, k, 1).expand(self.gradInput[0].size(0),
                    self.gradInput[0].size(1)))
                self.gradInput[0].add_(self.buff1)

                torch.mm(self.buff2, input[0], self.weight[k])
                self.buff2.mul_(gradOutput.narrow(1, k, 1).expand(self.gradInput[1].size(0),
                    self.gradInput[1].size(1)))
                self.gradInput[1].add_(self.buff2)

        return self.gradInput



    def accGradParameters(self, input, gradOutput, scale=1):
        self._assertInputGradOutput(input, gradOutput)

        # make sure we have buffer:
        self.buff1 = self.buff1 or input[0].new()
        self.buff1.resizeAs_(input[0])

        # accumulate parameter gradients:
        for k in range(self.weight.size(0)):
            torch.mul(self.buff1, input[0], gradOutput.narrow(1, k, 1).expandAs(input[0]))
            self.gradWeight[k].addmm_(self.buff1.t(), input[1])

        if self.bias:
            self.gradBias.add_(scale, gradOutput.sum(0))


    def __repr__(self):
        return str(type(self)) + \
                '({}x{} -> {}) {}'.format(
                    self.weight.size(1), self.weight.size(2), self.weight.size(0),
                    (' without bias' if self.bias is None else '')
                )

    def clearState(self):
        clear(self, 'buff1', 'buff2')
        return super(Bilinear, self).clearState()

