import torch
from torch.legacy import nn

class PartialLinear(nn.Module):
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
        raise NotImplementedError

        # define the layer as a small network:
        pt = nn.ParallelTable()
        pt.add(nn.Identity()).add(nn.LookupTable(outputsize, inputsize))
        self.network = nn.Sequential().add(pt).add(nn.MM(False, True))
        if bias:
            self.bias     = torch.zeros(1, outputsize)
            self.gradBias = torch.zeros(1, outputsize)
        else:
            self.bias = self.gradBias = None

        # set partition:
        self.inputsize  = inputsize
        self.outputsize = outputsize
        self.allcolumns = torch.range(0, self.outputsize-1)
        self.resetPartition()
        self.addBuffer = None
        self.buffer = None

    def setPartition(self, indices):
        self.partition = indices.type(self.allcolumns.type())

    def resetPartition(self):
        self.partition = self.allcolumns

    def parameters(self):
        return [self.network.get(0).get(1).weight,     self.bias], \
               [self.network.get(0).get(1).gradWeight, self.gradBias]
        # should return only the relevant partition?

    def updateOutput(self, input):
        self.output.set(self.network.forward([input, self.partition]))
        if self.bias:
            self.output.add(self.bias.index(1, self.partition.long()).expandAs(self.output))
            self.addBuffer = self.addBuffer or input.new()
            if self.addBuffer.nElement() != input.size(0):
                self.addBuffer.resize(input.size(0)).fill(1)

        return self.output

    def updateGradInput(self, input, gradOutput):
        if self.gradInput:
           self.network.updateGradInput([input, self.partition], gradOutput)
           self.gradInput.set(self.network.gradInput[0])

        return self.gradInput

    def accGradParameters(self, input, gradOutput, scale=1):
        self.network.accGradParameters([input, self.partition], gradOutput, scale)
        if self.bias:
            self.buffer = self.buffer or input.new()
            self.buffer.resize(gradOutput.size(1))
            self.buffer.mv(gradOutput.t(), self.addBuffer).mul(scale)
            self.gradBias.indexAdd(
                1, self.partition.long(), self.buffer.view(1, self.buffer.nElement())
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
        self.gradBias.zero()

    def updateParameters(self, learningRate):
        self.network.updateParameters(learningRate)
        self.bias.add(-learningRate, self.gradBias)

    def __repr__(self):
        return super(ParallelTable, self).__repr__() + \
           '({} -> {})'.format(self.inputsize, self.outputsize) + \
           ' without bias' if self.bias is None else ''

