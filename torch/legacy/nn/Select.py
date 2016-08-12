import torch
from torch.legacy import nn

class Select(nn.Module):

    def __init__(self, dimension, index):
        super(Select, self).__init__()
        self.dimension = dimension
        self.index = index

    def updateOutput(self, input):
        index = self.index if self.index >= 0 else input.size(self.dimension) + self.index
        output = input.select(self.dimension, index)
        self.output.resizeAs_(output)
        return self.output.copy_(output)

    def updateGradInput(self, input, gradOutput):
        index = self.index if self.index >= 0 else input.size(self.dimension) + self.index
        self.gradInput.resizeAs_(input)
        self.gradInput.zero_()
        self.gradInput.select(self.dimension, index).copy_(gradOutput)
        return self.gradInput

