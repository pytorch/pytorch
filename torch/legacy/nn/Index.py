import torch
from .Module import Module

class Index(Module):

    def __init__(self, dimension):
         super(Index, self).__init__()
         self.dimension = dimension
         self.gradInput = [self.gradInput]

    def updateOutput(self, input):
         t = input[0]
         index = input[1]
         torch.index_select(self.output, t, self.dimension, index)
         return self.output

    def updateGradInput(self, input, gradOutput):
         t = input[0]
         index = input[1]

         gradInput = self.gradInput[0]  # no gradient for the index variable
         gradInput.resize_as_(t).zero_()
         gradInput.index_add_(self.dimension, index, gradOutput)
         return self.gradInput

