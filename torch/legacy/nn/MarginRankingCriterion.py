import torch
from .Criterion import Criterion


class MarginRankingCriterion(Criterion):

    def __init__(self, margin=1, sizeAverage=True):
        super(MarginRankingCriterion, self).__init__()
        self.margin = margin
        self.sizeAverage = sizeAverage
        self.gradInput = [torch.Tensor(), torch.Tensor()]

        self._output = None
        self.dist = None
        self.mask = None

    def updateOutput(self, input, y):
        if input[0].size(0) == 1:
            self.output = max(0, -y * (input[0][0] - input[1][0]) + self.margin)
        else:
            if self._output is None:
                self._output = input[0].clone()
            self._output.resize_as_(input[0])
            self._output.copy_(input[0])

            self._output.add_(-1, input[1])
            self._output.mul_(-1).mul_(y)
            self._output.add_(self.margin)

            self._output.clamp_(min=0)

            self.output = self._output.sum()

            if self.sizeAverage:
                self.output = self.output / y.size(0)

        return self.output

    def updateGradInput(self, input, y):
        if input[0].size(0) == 1:
            dist = -y * (input[0][0] - input[1][0]) + self.margin
            if dist < 0:
                self.gradInput[0][0] = 0
                self.gradInput[1][0] = 0
            else:
                self.gradInput[0][0] = -y
                self.gradInput[1][0] = y
        else:
            if self.dist is None:
                self.dist = input[0].new()
            self.dist = self.dist.resize_as_(input[0]).copy_(input[0])
            dist = self.dist

            dist.add_(-1, input[1])
            dist.mul_(-1).mul_(y)
            dist.add_(self.margin)

            if self.mask is None:
                self.mask = input[0].new()
            self.mask = self.mask.resize_as_(input[0]).copy_(dist)
            mask = self.mask

            torch.ge(dist, 0, out=mask)

            self.gradInput[0].resize_(dist.size())
            self.gradInput[1].resize_(dist.size())

            self.gradInput[0].copy_(mask)
            self.gradInput[0].mul_(-1).mul_(y)
            self.gradInput[1].copy_(mask)
            self.gradInput[1].mul_(y)

            if self.sizeAverage:
                self.gradInput[0].div_(y.size(0))
                self.gradInput[1].div_(y.size(0))

        return self.gradInput
