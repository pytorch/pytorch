import torch
from .Criterion import Criterion


class HingeEmbeddingCriterion(Criterion):

    def __init__(self, margin=1, sizeAverage=True):
        super(HingeEmbeddingCriterion, self).__init__()
        self.margin = margin
        self.sizeAverage = sizeAverage
        self.buffer = None

    def updateOutput(self, input, y):
        if self.buffer is None:
            self.buffer = input.new()
        self.buffer.resize_as_(input).copy_(input)
        self.buffer[torch.eq(y, -1.)] = 0
        self.output = self.buffer.sum()

        self.buffer.fill_(self.margin).add_(-1, input)
        self.buffer.clamp_(min=0)
        self.buffer[torch.eq(y, 1.)] = 0
        self.output = self.output + self.buffer.sum()

        if self.sizeAverage:
            self.output = self.output / input.nelement()

        return self.output

    def updateGradInput(self, input, y):
        self.gradInput.resize_as_(input).copy_(y)
        self.gradInput[torch.mul(torch.eq(y, -1), torch.gt(input, self.margin))] = 0

        if self.sizeAverage:
            self.gradInput.mul_(1. / input.nelement())

        return self.gradInput
