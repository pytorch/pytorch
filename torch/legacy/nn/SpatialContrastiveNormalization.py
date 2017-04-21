import torch
from .Module import Module
from .Sequential import Sequential
from .SpatialSubtractiveNormalization import SpatialSubtractiveNormalization
from .SpatialDivisiveNormalization import SpatialDivisiveNormalization


class SpatialContrastiveNormalization(Module):

    def __init__(self, nInputPlane=1, kernel=None, threshold=1e-4, thresval=1e-4):
        super(SpatialContrastiveNormalization, self).__init__()

        # get args
        self.nInputPlane = nInputPlane
        if kernel is None:
            self.kernel = torch.Tensor(9, 9).fill_(1)
        else:
            self.kernel = kernel
        self.threshold = threshold
        self.thresval = thresval or threshold
        kdim = self.kernel.ndimension()

        # check args
        if kdim != 2 and kdim != 1:
            raise ValueError('SpatialContrastiveNormalization averaging kernel must be 2D or 1D')

        if self.kernel.size(0) % 2 == 0 or (kdim == 2 and (self.kernel.size(1) % 2) == 0):
            raise ValueError('SpatialContrastiveNormalization averaging kernel must have ODD dimensions')

        # instantiate sub+div normalization
        self.normalizer = Sequential()
        self.normalizer.add(SpatialSubtractiveNormalization(self.nInputPlane, self.kernel))
        self.normalizer.add(SpatialDivisiveNormalization(self.nInputPlane, self.kernel,
                                                         self.threshold, self.thresval))

    def updateOutput(self, input):
        self.output = self.normalizer.forward(input)
        return self.output

    def updateGradInput(self, input, gradOutput):
        self.gradInput = self.normalizer.backward(input, gradOutput)
        return self.gradInput
