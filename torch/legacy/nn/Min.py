import torch
from .Module import Module
from .utils import clear, addSingletondimension


class Min(Module):

    def __init__(self, dimension=0):
        super(Min, self).__init__()
        self.dimension = dimension
        self._output = None
        self._indices = None

    def _getPositiveDimension(self, input):
        dimension = self.dimension
        if dimension < 0:
            dimension = input.dim() + dimension

        return dimension

    def _lazyInit(self):
        if self._output is None:
            self._output = self.output.new()
        if self._indices is None:
            self._indices = \
                (torch.cuda.LongTensor() if torch.typename(self.output) == 'torch.cuda.FloatTensor'
                 else torch.LongTensor())

    def updateOutput(self, input):
        self._lazyInit()
        dimension = self._getPositiveDimension(input)
        torch.min(input, dimension, out=(self._output, self._indices))
        if input.dim() > 1:
            self.output.set_(self._output.select(dimension, 0))
        else:
            self.output.set_(self._output)

        return self.output

    def updateGradInput(self, input, gradOutput):
        self._lazyInit()
        dimension = self._getPositiveDimension(input)
        if input.dim() > 1:
            gradOutputView = addSingletondimension(gradOutput, dimension)
        else:
            gradOutputView = gradOutput

        self.gradInput.resize_as_(input).zero_().scatter_(dimension, self._indices, gradOutputView)
        return self.gradInput

    def type(self, type, tensorCache=None):
        # torch.min expects a LongTensor as indices, whereas cutorch.max expects a CudaTensor.
        if type == 'torch.cuda.FloatTensor':
            indices, self._indices = self._indices, None
            super(Min, self).type(type, tensorCache)
            self._indices = indices.type('torch.cuda.LongTensor') if indices is not None else None
        else:
            # self._indices must be a LongTensor. Setting it to nil temporarily avoids
            # unnecessary memory allocations.
            indices, self._indices = self._indices, None
            super(Min, self).type(type, tensorCache)
            self._indices = indices.long() if indices is not None else None

        return self

    def clearState(self):
        clear(self, '_indices', '_output')
        return super(Min, self).clearState()
