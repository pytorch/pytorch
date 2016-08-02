import torch
from torch.legacy import nn

class Max(nn.Module):

    def __init__(self, dimension=0):
        super(Max, self).__init__()
        self.dimension = dimension
        self._output = None
        self._indices = None

    def _getPositiveDimension(self, input):
        dimension = self.dimension
        if dimension < 0:
           dimension = input.dim() + dimension + 1

        return dimension

    def _lazyInit(self):
        self._output = self._output or self.output.new()
        self._indices = self._indices or \
           (torch.cuda.FloatTensor() if str(type(self.output)) == 'torch.cuda.FloatTensor' else torch.LongTensor())

    def updateOutput(self, input):
        self._lazyInit()
        dimension = self._getPositiveDimension(input)
        torch.max(self._output, self._indices, input, dimension)
        if input.dim() > 1:
          self.output.set(self._output.select(dimension, 0))
        else:
          self.output.set(self._output)

        return self.output

    def updateGradInput(self, input, gradOutput):
        self._lazyInit()
        dimension = self._getPositiveDimension(input)
        if input.dim() > 1:
          gradOutputView = nn.utils.addSingletonDimension(gradOutput, dimension)
        else:
          gradOutputView = gradOutput

        self.gradInput.resizeAs(input).zero().scatter(dimension, self._indices, gradOutputView)
        return self.gradInput

    def type(self, type, tensorCache):
        # torch.max expects a LongTensor as indices, whereas cutorch.max expects a CudaTensor.
        if type == 'torch.cuda.FloatTensor':
            super(Max, self).type(type, tensorCache)
        else:
            # self._indices must be a LongTensor. Setting it to nil temporarily avoids
            # unnecessary memory allocations.
            indices, self._indices = self._indices, None
            super(Max, self).type(self, type, tensorCache)
            self._indices = indices.long() if indices else None

        return self

    def clearState(self):
        nn.utils.clear(self, '_indices', '_output')
        return super(Max, self).clearState()

