import torch
from .Module import Module
from .Identity import Identity
from .LookupTable import LookupTable
from .Sequential import Sequential
from .ParallelTable import ParallelTable
from .MM import MM


class PartialLinear(Module):
    """
    PartialLinear is a Linear layer that allows the user to a set a collection of
    column indices. When the column indices are set, the layer will behave like a
    Linear layer that only has those columns. Meanwhile, all parameters are
    preserved, so resetting the PartialLinear layer will result in a module that
    behaves just like a regular Linear layer.

    This module is useful, for instance, when you want to: forward-backward on
    only a subset of a Linear layer during training but use the full Linear layer
    at test time.
    """

    def __init__(self, inputsize, outputsize, bias=True):
        super(PartialLinear, self).__init__()

        # define the layer as a small network:
        pt = ParallelTable()
        pt.add(Identity()).add(LookupTable(outputsize, inputsize))
        self.network = Sequential().add(pt).add(MM(False, True))
        if bias:
            self.bias = torch.zeros(1, outputsize)
            self.gradBias = torch.zeros(1, outputsize)
        else:
            self.bias = self.gradBias = None

        # set partition:
        self.inputsize = inputsize
        self.outputsize = outputsize
        self.allcolumns = torch.arange(0, self.outputsize).long()
        self.resetPartition()
        self.addBuffer = None
        self.buffer = None

    def setPartition(self, indices):
        self.partition = indices.type(self.allcolumns.type())
        return self

    def resetPartition(self):
        self.partition = self.allcolumns
        return self

    def parameters(self):
        return [self.network.get(0).get(1).weight, self.bias], \
               [self.network.get(0).get(1).gradWeight, self.gradBias]
        # should return only the relevant partition?

    def updateOutput(self, input):
        self.output.set_(self.network.forward([input, self.partition]))
        if self.bias is not None:
            self.output.add_(torch.index_select(self.bias, 1, self.partition).expand_as(self.output))
            if self.addBuffer is None:
                self.addBuffer = input.new()
            if self.addBuffer.nelement() != input.size(0):
                self.addBuffer.resize_(input.size(0)).fill_(1)

        return self.output

    def updateGradInput(self, input, gradOutput):
        if self.gradInput is not None:
            self.network.updateGradInput([input, self.partition], gradOutput)
            self.gradInput.set_(self.network.gradInput[0])

        return self.gradInput

    def accGradParameters(self, input, gradOutput, scale=1):
        self.network.accGradParameters([input, self.partition], gradOutput, scale)
        if self.bias is not None:
            if self.buffer is None:
                self.buffer = input.new()
            self.buffer.resize_(gradOutput.size(1))
            torch.mv(gradOutput.t(), self.addBuffer, out=self.buffer).mul_(scale)
            self.gradBias.index_add_(
                1, self.partition, self.buffer.view(1, self.buffer.nelement())
            )

    def accUpdateGradParameters(self, input, gradOutput, lr):
        gradWeight = self.network.get(0).get(1).gradWeight
        gradBias = self.gradBias
        self.network.get(0).get(1).gradWeight = self.network.get(0).get(1).weight
        self.gradBias = self.bias
        self.accGradParameters(input, gradOutput, -lr)
        self.network.get(0).get(1).gradWeight = gradWeight
        self.gradBias = gradBias

    def zeroGradParameters(self):
        self.network.zeroGradParameters()
        self.gradBias.zero_()

    def updateParameters(self, learningRate):
        self.network.updateParameters(learningRate)
        self.bias._add(-learningRate, self.gradBias)

    def type(self, type=None, tensorCache=None):
        result = super(PartialLinear, self).type(type, tensorCache)
        self.partition = self.partition.long()
        self.allcolumns = self.allcolumns.long()
        if type == 'torch.cuda.FloatTensor':
            self.allcolumns = self.allcolumns.cuda()
            self.partition = self.partition.cuda()
        return result

    def __repr__(self):
        return super(ParallelTable, self).__repr__() + \
            '({} -> {})'.format(self.inputsize, self.outputsize) + \
            ' without bias' if self.bias is None else ''
