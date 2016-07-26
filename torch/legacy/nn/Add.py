import math
import torch
from torch.legacy import nn

class Add(nn.Module):

    def __init__(self, inputSize, scalar=False):
        super(Add, self).__init__()
        size = inputSize
        if scalar:
            assert size == 1
        self.scalar = scalar
        self.bias = torch.Tensor(size)
        self.gradBias = torch.Tensor(size)

        self._ones = torch.Tensor((1,))

        self.reset()

    def reset(self, stdv=None):
        if stdv is not None:
           stdv = stdv * math.sqrt(3)
        else:
           stdv = 1./math.sqrt(self.bias.size(0))

        self.bias.uniform(-stdv, stdv)

    def updateOutput(self, input):
        self.output.resizeAs(input).copy(input)
        if self.scalar:
            self.output.add(self.bias[0]);
        else:
            batchSize = input.size(0)
            if self._ones.size(0) != batchSize:
                self._ones.resize(batchSize).fill(1)

            bias = self.bias.view(-1)
            output = self.output.view(batchSize, -1)
            output.addr(1, self._ones, bias)

        return self.output

    def updateGradInput(self, input, gradOutput):
        if self.gradInput:
           self.gradInput.resizeAs(gradOutput).copy(gradOutput)
           return self.gradInput

    def accGradParameters(self, input, gradOutput, scale=1):
        if self.gradBias.size(0) == 1:
           self.gradBias[0] = self.gradBias[0] + scale*gradOutput.sum();
        else:
           if input.isSameSizeAs(self.bias):
              self.gradBias.add(scale, gradOutput)
           else:
              gradOutput = gradOutput.view(input.size(0), -1)
              self.gradBias.view(-1).addmv(scale, gradOutput.t(), self._ones)

