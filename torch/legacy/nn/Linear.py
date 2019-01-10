import math
import torch
from .Module import Module
from .utils import clear


class Linear(Module):

    def __init__(self, inputSize, outputSize, bias=True):
        super(Linear, self).__init__()
        self.weight = torch.Tensor(outputSize, inputSize)
        self.gradWeight = torch.Tensor(outputSize, inputSize)
        self.bias = torch.Tensor(outputSize) if bias else None
        self.gradBias = torch.Tensor(outputSize) if bias else None
        self.reset()

        self.addBuffer = None

    def noBias(self):
        self.bias = None
        self.gradBias = None
        return self

    def reset(self, stdv=None):
        if stdv is not None:
            stdv = stdv * math.sqrt(3)
        else:
            stdv = 1. / math.sqrt(self.weight.size(1))

        self.weight.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.uniform_(-stdv, stdv)

        return self

    def _updateAddBuffer(self, input):
        nframe = input.size(0)
        if self.addBuffer is None:
            self.addBuffer = input.new()
        if self.addBuffer.nelement() != nframe:
            self.addBuffer.resize_(nframe).fill_(1)

    def updateOutput(self, input):
        assert input.dim() == 2
        nframe = input.size(0)
        nelement = self.output.nelement()
        self.output.resize_(nframe, self.weight.size(0))
        if self.output.nelement() != nelement:
            self.output.zero_()

        self._updateAddBuffer(input)
        self.output.addmm_(0, 1, input, self.weight.t())
        if self.bias is not None:
            self.output.addr_(self.addBuffer, self.bias)

        return self.output

    def updateGradInput(self, input, gradOutput):
        if self.gradInput is None:
            return

        nelement = self.gradInput.nelement()
        self.gradInput.resize_as_(input)
        if self.gradInput.nelement() != nelement:
            self.gradInput.zero_()

        assert input.dim() == 2
        self.gradInput.addmm_(0, 1, gradOutput, self.weight)

        return self.gradInput

    def accGradParameters(self, input, gradOutput, scale=1):
        assert input.dim() == 2
        self.gradWeight.addmm_(scale, gradOutput.t(), input)
        if self.bias is not None:
            # update the size of addBuffer if the input is not the same size as the one we had in last updateGradInput
            self._updateAddBuffer(input)
            self.gradBias.addmv_(scale, gradOutput.t(), self.addBuffer)

    def clearState(self):
        clear(self, 'addBuffer')
        return super(Linear, self).clearState()

    def __repr__(self):
        return super(Linear, self).__repr__() + \
            '({} -> {})'.format(self.weight.size(1), self.weight.size(0)) + \
            (' without bias' if self.bias is None else '')
