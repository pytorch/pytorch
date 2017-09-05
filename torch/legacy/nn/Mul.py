import math
import torch
from .Module import Module


class Mul(Module):

    def __init__(self):
        super(Mul, self).__init__()
        self.weight = torch.Tensor(1)
        self.gradWeight = torch.Tensor(1)
        self.reset()

    def reset(self, stdv=None):
        if stdv is not None:
            stdv = stdv * math.sqrt(3)
        else:
            stdv = 1. / math.sqrt(self.weight.size(0))
        self.weight.uniform_(-stdv, stdv)

    def updateOutput(self, input):
        self.output.resize_as_(input).copy_(input)
        self.output.mul_(self.weight[0])
        return self.output

    def updateGradInput(self, input, gradOutput):
        self.gradInput.resize_as_(input).zero_()
        self.gradInput.add_(self.weight[0], gradOutput)
        return self.gradInput

    def accGradParameters(self, input, gradOutput, scale=1):
        self.gradWeight[0] = (self.gradWeight[0] +
                              scale * input.contiguous().view(-1).dot(gradOutput.contiguous().view(-1)))
