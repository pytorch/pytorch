import torch
from torch.legacy import nn

class Sum(nn.Module):

    def __init__(self, dimension=0, sizeAverage=False):
        super(Sum, self).__init__()
        self.dimension   = dimension
        self.sizeAverage = sizeAverage
        self._gradOutput = None

    def _getPositiveDimension(self, input):
         dimension = self.dimension
         if dimension < 0:
             dimension = input.dim() + dimension
         return dimension

    def updateOutput(self, input):
         dimension = self._getPositiveDimension(input)

         self.output.sum(input, dimension)
         if self.sizeAverage:
             self.output.div(input.size(dimension))

         return self.output

    def updateGradInput(self, input, gradOutput):
        dimension = self._getPositiveDimension(input)
        # zero-strides dont work with MKL/BLAS, so
        # dont set self.gradInput to zero-stride tensor.
        # Instead, do a deepcopy.
        size = input.size()
        size[dimension] = 1
        if not gradOutput.isContiguous():
            self._gradOutput = self._gradOutput or gradOutput.new()
            self._gradOutput.resizeAs(gradOutput).copy(gradOutput)
            gradOutput = self._gradOutput

        gradOutput = gradOutput.view(size)
        self.gradInput.resizeAs(input)
        self.gradInput.copy(gradOutput.expandAs(input))
        if self.sizeAverage:
            self.gradInput.div(input.size(dimension))

        return self.gradInput


    def clearState(self):
         nn.utils.clear(self, '_gradOutput')
         return super(Sum, self).clearState(self)

