import math
import torch
from .Module import Module
from .Sequential import Sequential
from .SpatialZeroPadding import SpatialZeroPadding
from .SpatialConvolution import SpatialConvolution
from .SpatialConvolutionMap import SpatialConvolutionMap
from .Replicate import Replicate
from .CSubTable import CSubTable
from .CDivTable import CDivTable
from .utils import clear


class SpatialSubtractiveNormalization(Module):

    def __init__(self, nInputPlane=1, kernel=None):
        super(SpatialSubtractiveNormalization, self).__init__()

        # get args
        self.nInputPlane = nInputPlane
        if kernel is None:
            kernel = torch.Tensor(9, 9).fill_(1)
        self.kernel = kernel
        kdim = self.kernel.ndimension()

        # check args
        if kdim != 2 and kdim != 1:
            raise ValueError('SpatialSubtractiveNormalization averaging kernel must be 2D or 1D')

        if (self.kernel.size(0) % 2) == 0 or (kdim == 2 and (self.kernel.size(1) % 2) == 0):
            raise ValueError('SpatialSubtractiveNormalization averaging kernel must have ODD dimensions')

        # normalize kernel
        self.kernel.div_(self.kernel.sum() * self.nInputPlane)

        # padding values
        padH = int(math.floor(self.kernel.size(0) / 2))
        padW = padH
        if kdim == 2:
            padW = int(math.floor(self.kernel.size(1) / 2))

        # create convolutional mean extractor
        self.meanestimator = Sequential()
        self.meanestimator.add(SpatialZeroPadding(padW, padW, padH, padH))
        if kdim == 2:
            self.meanestimator.add(SpatialConvolution(self.nInputPlane, 1, self.kernel.size(1), self.kernel.size(0)))
        else:
            # TODO: map
            self.meanestimator.add(SpatialConvolutionMap(
                SpatialConvolutionMap.maps.oneToOne(self.nInputPlane), self.kernel.size(0), 1))
            self.meanestimator.add(SpatialConvolution(self.nInputPlane, 1, 1, self.kernel.size(0)))

        self.meanestimator.add(Replicate(self.nInputPlane, 0))

        # set kernel and bias
        if kdim == 2:
            for i in range(self.nInputPlane):
                self.meanestimator.modules[1].weight[0][i] = self.kernel
            self.meanestimator.modules[1].bias.zero_()
        else:
            for i in range(self.nInputPlane):
                self.meanestimator.modules[1].weight[i].copy_(self.kernel)
                self.meanestimator.modules[2].weight[0][i].copy_(self.kernel)

            self.meanestimator.modules[1].bias.zero_()
            self.meanestimator.modules[2].bias.zero_()

        # other operation
        self.subtractor = CSubTable()
        self.divider = CDivTable()

        # coefficient array, to adjust side effects
        self.coef = torch.Tensor(1, 1, 1)

        self.ones = None
        self._coef = None

    def updateOutput(self, input):
        # compute side coefficients
        dim = input.dim()
        if (input.dim() + 1 != self.coef.dim() or
                (input.size(dim - 1) != self.coef.size(dim - 1)) or
                (input.size(dim - 2) != self.coef.size(dim - 2))):
            if self.ones is None:
                self.ones = input.new()
            if self._coef is None:
                self._coef = self.coef.new()

            self.ones.resize_as_(input[0:1]).fill_(1)
            coef = self.meanestimator.updateOutput(self.ones).squeeze(0)
            self._coef.resize_as_(coef).copy_(coef)  # make contiguous for view
            size = list(coef.size())
            size = [input.size(0)] + size
            self.coef = self._coef.view(1, *self._coef.size()).expand(*size)

        # compute mean
        self.localsums = self.meanestimator.updateOutput(input)
        self.adjustedsums = self.divider.updateOutput([self.localsums, self.coef])
        self.output = self.subtractor.updateOutput([input, self.adjustedsums])

        return self.output

    def updateGradInput(self, input, gradOutput):
        # resize grad
        self.gradInput.resize_as_(input).zero_()

        # backprop through all modules
        gradsub = self.subtractor.updateGradInput([input, self.adjustedsums], gradOutput)
        graddiv = self.divider.updateGradInput([self.localsums, self.coef], gradsub[1])
        size = self.meanestimator.updateGradInput(input, graddiv[0]).size()
        self.gradInput.add_(self.meanestimator.updateGradInput(input, graddiv[0]))
        self.gradInput.add_(gradsub[0])

        return self.gradInput

    def clearState(self):
        clear(self, 'ones', '_coef')
        self.meanestimator.clearState()
        return super(SpatialSubtractiveNormalization, self).clearState()
