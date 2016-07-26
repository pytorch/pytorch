import math
import torch
from torch.legacy import nn

class Linear(nn.Module):

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
            stdv = 1./math.sqrt(self.weight.size(1))

        # TODO: is removing oldSeed ok?
        self.weight.uniform(-stdv, stdv)
        if self.bias is not None:
            self.bias.uniform(-stdv, stdv)

        return self

    def _updateAddBuffer(self, input):
        nframe = input.size(0)
        self.addBuffer = self.addBuffer or input.new()
        if self.addBuffer.nElement() != nframe:
            self.addBuffer.resize(nframe).fill(1)

    def updateOutput(self, input):
        assert input.dim() == 2
        nframe = input.size(0)
        nElement = self.output.nElement()
        self.output.resize(nframe, self.weight.size(0))
        if self.output.nElement() != nElement:
            self.output.zero()

        self._updateAddBuffer(input)
        self.output.addmm(0, self.output, 1, input, self.weight.t())
        if self.bias is not None:
            self.output.addr(1, self.addBuffer, self.bias)

        return self.output

    def updateGradInput(self, input, gradOutput):
        if not self.gradInput:
            return

        nElement = self.gradInput.nElement()
        self.gradInput.resizeAs(input)
        if self.gradInput.nElement() != nElement:
            self.gradInput.zero()

        assert input.dim() == 2
        self.gradInput.addmm(0, 1, gradOutput, self.weight)

        return self.gradInput



    def accGradParameters(self, input, gradOutput, scale=1):
        assert input.dim() == 2
        self.gradWeight.addmm(scale, gradOutput.t(), input)
        if self.bias is not None:
            # update the size of addBuffer if the input is not the same size as the one we had in last updateGradInput
            self._updateAddBuffer(input)
            self.gradBias.addmv(scale, gradOutput.t(), self.addBuffer)

    # we: not need to accumulate parameters when sharing
    # TODO
    #Linear.sharedAccUpdateGradParameters = Linear.accUpdateGradParameters

    def clearState(self):
        # TODO: this shouldn't call set
        if self.addBuffer:
            self.addBuffer.set()
        return super(Linear, self).clearState()


    def __repr__(self):
        return torch.typename(self) + \
                '({} -> {})'.format(self.weight.size(1), self.weight.size(0)) + \
                (' without bias' if not self.bias else '')

