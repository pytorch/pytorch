import torch
from torch.legacy import nn

class HingeEmbeddingCriterion(nn.Criterion):

    def __init__(self, margin=1, sizeAverage=True):
        super(HingeEmbeddingCriterion, self).__init__()
        self.margin = margin
        self.sizeAverage = sizeAverage
        self.buffer = None

    def updateOutput(self, input, y):
        self.buffer = self.buffer or input.new()
        self.buffer.resizeAs(input).copy(input)
        self.buffer[torch.eq(y, float(-1))] = 0
        self.output = self.buffer.sum()

        self.buffer.fill(self.margin).add(-1, input)
        self.buffer.cmax(0)
        self.buffer[torch.eq(y, float(1))] = 0
        self.output = self.output + self.buffer.sum()

        if self.sizeAverage:
           self.output = self.output / input.nElement()

        return self.output

    def updateGradInput(self, input, y):
        self.gradInput.resizeAs(input).copy(y)
        self.gradInput[torch.cmul(torch.eq(y, -1), torch.gt(input, self.margin))] = 0

        if self.sizeAverage:
           self.gradInput.mul(1 / input.nElement())

        return self.gradInput

