import torch
from torch.legacy import nn

class Index(nn.Module):

    def __init__(self, dimension):
         super(Index, self).__init__()
         self.dimension = dimension
         self.gradInput = [self.gradInput]

    def updateOutput(self, input):
         t = input[0]
         index = input[1]
         self.output.index(t, self.dimension, index)
         return self.output

    def updateGradInput(self, input, gradOutput):
         t = input[0]
         index = input[1]

         gradInput = self.gradInput[0]  # no gradient for the index variable
         gradInput.resizeAs(t).zero()
         gradInput.indexAdd(self.dimension, index, gradOutput)
         return self.gradInput

