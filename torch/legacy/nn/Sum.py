import torch
from .Module import Module
from .utils import clear


class Sum(Module):

    def __init__(self, dimension=0, sizeAverage=False):
        super(Sum, self).__init__()
        self.dimension = dimension
        self.sizeAverage = sizeAverage
        self._gradOutput = None

    def _getPositiveDimension(self, input):
        dimension = self.dimension
        if dimension < 0:
            dimension = input.dim() + dimension
        return dimension

    def updateOutput(self, input):
        dimension = self._getPositiveDimension(input)

        torch.sum(input, dimension, out=self.output)
        if self.sizeAverage:
            self.output.div_(input.size(dimension))
        if self.output.dim() > 1:
            self.output.set_(self.output.select(dimension, 0))

        return self.output

    def updateGradInput(self, input, gradOutput):
        dimension = self._getPositiveDimension(input)
        # zero-strides dont work with MKL/BLAS, so
        # dont set self.gradInput to zero-stride tensor.
        # Instead, do a deepcopy.
        size = list(input.size())
        size[dimension] = 1
        if not gradOutput.is_contiguous():
            if self._gradOutput is None:
                self._gradOutput = gradOutput.new()
            self._gradOutput.resize_as_(gradOutput).copy_(gradOutput)
            gradOutput = self._gradOutput

        gradOutput = gradOutput.view(*size)
        self.gradInput.resize_as_(input)
        self.gradInput.copy_(gradOutput.expand_as(input))
        if self.sizeAverage:
            self.gradInput.div_(input.size(dimension))

        return self.gradInput

    def clearState(self):
        clear(self, '_gradOutput')
        return super(Sum, self).clearState()
