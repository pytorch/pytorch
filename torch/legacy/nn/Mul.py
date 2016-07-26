import math
import torch
from torch.legacy import nn

class Mul(nn.Module):

    def __init__(self):
        super(Mul, self).__init__()
        self.weight = torch.Tensor(1)
        self.gradWeight = torch.Tensor(1)
        self.reset()

    def reset(self, stdv=None):
        if stdv is not None:
           stdv = stdv * math.sqrt(3)
        else:
           stdv = 1./math.sqrt(self.weight.size(0))
        self.weight.uniform(-stdv, stdv)

    def updateOutput(self, input):
        self.output.resizeAs(input).copy(input);
        self.output.mul(self.weight[0]);
        return self.output

    def updateGradInput(self, input, gradOutput):
        self.gradInput.resizeAs(input).zero()
        self.gradInput.add(self.weight[0], gradOutput)
        return self.gradInput

    def accGradParameters(self, input, gradOutput, scale=1):
        self.gradWeight[0] = self.gradWeight[0] + scale*input.dot(gradOutput);

