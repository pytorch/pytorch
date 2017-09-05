import math
import torch
from .Module import Module
from .Sequential import Sequential
from .SpatialZeroPadding import SpatialZeroPadding
from .SpatialConvolution import SpatialConvolution
from .SpatialConvolutionMap import SpatialConvolutionMap
from .Replicate import Replicate
from .Square import Square
from .Sqrt import Sqrt
from .CDivTable import CDivTable
from .Threshold import Threshold
from .utils import clear


class SpatialDivisiveNormalization(Module):

    def __init__(self, nInputPlane=1, kernel=None, threshold=1e-4, thresval=None):
        super(SpatialDivisiveNormalization, self).__init__()

        # get args
        self.nInputPlane = nInputPlane
        if kernel is None:
            kernel = torch.Tensor(9, 9).fill_(1)
        self.kernel = kernel
        self.threshold = threshold
        self.thresval = thresval if thresval is not None else threshold
        kdim = self.kernel.ndimension()

        # check args
        if kdim != 2 and kdim != 1:
            raise ValueError('SpatialDivisiveNormalization averaging kernel must be 2D or 1D')

        if (self.kernel.size(0) % 2) == 0 or (kdim == 2 and (self.kernel.size(1) % 2) == 0):
            raise ValueError('SpatialDivisiveNormalization averaging kernel must have ODD dimensions')

        # padding values
        padH = int(math.floor(self.kernel.size(0) / 2))
        padW = padH
        if kdim == 2:
            padW = int(math.floor(self.kernel.size(1) / 2))

        # create convolutional mean estimator
        self.meanestimator = Sequential()
        self.meanestimator.add(SpatialZeroPadding(padW, padW, padH, padH))
        if kdim == 2:
            self.meanestimator.add(SpatialConvolution(self.nInputPlane, 1, self.kernel.size(1), self.kernel.size(0)))
        else:
            self.meanestimator.add(SpatialConvolutionMap(
                SpatialConvolutionMap.maps.oneToOne(self.nInputPlane), self.kernel.size(0), 1))
            self.meanestimator.add(SpatialConvolution(self.nInputPlane, 1, 1, self.kernel.size(0)))

        self.meanestimator.add(Replicate(self.nInputPlane, 1))

        # create convolutional std estimator
        self.stdestimator = Sequential()
        self.stdestimator.add(Square())
        self.stdestimator.add(SpatialZeroPadding(padW, padW, padH, padH))
        if kdim == 2:
            self.stdestimator.add(SpatialConvolution(self.nInputPlane, 1, self.kernel.size(1), self.kernel.size(0)))
        else:
            self.stdestimator.add(SpatialConvolutionMap(
                SpatialConvolutionMap.maps.oneToOne(self.nInputPlane), self.kernel.size(0), 1))
            self.stdestimator.add(SpatialConvolution(self.nInputPlane, 1, 1, self.kernel.size(0)))

        self.stdestimator.add(Replicate(self.nInputPlane, 1))
        self.stdestimator.add(Sqrt())

        # set kernel and bias
        if kdim == 2:
            self.kernel.div_(self.kernel.sum() * self.nInputPlane)
            for i in range(self.nInputPlane):
                self.meanestimator.modules[1].weight[0][i] = self.kernel
                self.stdestimator.modules[2].weight[0][i] = self.kernel

            self.meanestimator.modules[1].bias.zero_()
            self.stdestimator.modules[2].bias.zero_()
        else:
            self.kernel.div_(self.kernel.sum() * math.sqrt(self.nInputPlane))
            for i in range(self.nInputPlane):
                self.meanestimator.modules[1].weight[i].copy_(self.kernel)
                self.meanestimator.modules[2].weight[0][i].copy_(self.kernel)
                self.stdestimator.modules[2].weight[i].copy_(self.kernel)
                self.stdestimator.modules[3].weight[0][i].copy_(self.kernel)

            self.meanestimator.modules[1].bias.zero_()
            self.meanestimator.modules[2].bias.zero_()
            self.stdestimator.modules[2].bias.zero_()
            self.stdestimator.modules[3].bias.zero_()

        # other operation
        self.normalizer = CDivTable()
        self.divider = CDivTable()
        self.thresholder = Threshold(self.threshold, self.thresval)

        # coefficient array, to adjust side effects
        self.coef = torch.Tensor(1, 1, 1)

        self.ones = None
        self._coef = None

    def updateOutput(self, input):
        self.localstds = self.stdestimator.updateOutput(input)

        # compute side coefficients
        dim = input.dim()
        if (self.localstds.dim() != self.coef.dim() or
                (input.size(dim - 1) != self.coef.size(dim - 1)) or
                (input.size(dim - 2) != self.coef.size(dim - 2))):
            if self.ones is None:
                self.ones = input.new()
            self.ones.resize_as_(input[0:1]).fill_(1)
            coef = self.meanestimator.updateOutput(self.ones).squeeze(0)
            if self._coef is None:
                self._coef = input.new()
            self._coef.resize_as_(coef).copy_(coef)  # make contiguous for view
            self.coef = self._coef.view(1, *self._coef.size()).expand_as(self.localstds)

        # normalize std dev
        self.adjustedstds = self.divider.updateOutput([self.localstds, self.coef.contiguous().view_as(self.localstds)])
        self.thresholdedstds = self.thresholder.updateOutput(self.adjustedstds)
        self.output = self.normalizer.updateOutput([input, self.thresholdedstds.contiguous().view_as(input)])

        return self.output

    def updateGradInput(self, input, gradOutput):
        # resize grad
        self.gradInput.resize_as_(input).zero_()

        # backprop through all modules
        gradnorm = (self.normalizer.updateGradInput(
            [input, self.thresholdedstds.contiguous().view_as(input)], gradOutput))
        gradadj = self.thresholder.updateGradInput(self.adjustedstds, gradnorm[1])
        graddiv = (self.divider.updateGradInput(
            [self.localstds, self.coef.contiguous().view_as(self.localstds)], gradadj))
        self.gradInput.add_(self.stdestimator.updateGradInput(input, graddiv[0]))
        self.gradInput.add_(gradnorm[0])

        return self.gradInput

    def clearState(self):
        clear(self, 'ones', '_coef')
        self.meanestimator.clearState()
        self.stdestimator.clearState()
        return super(SpatialDivisiveNormalization, self).clearState()
