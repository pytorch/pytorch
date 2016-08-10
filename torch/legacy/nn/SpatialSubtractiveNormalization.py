import math
import torch
from torch.legacy import nn

class SpatialSubtractiveNormalization(nn.Module):

    def __init__(self, nInputPlane=1, kernel=None):
        super(SpatialSubtractiveNormalization, self).__init__()

        # get args
        self.nInputPlane = nInputPlane
        self.kernel = kernel or torch.Tensor(9, 9).fill_(1)
        kdim = self.kernel.nDimension()

        # check args
        if kdim != 2 and kdim != 1:
           error('SpatialSubtractiveNormalization averaging kernel must be 2D or 1D')

        if (self.kernel.size(0) % 2) == 0 or (kdim == 2 and (self.kernel.size(1) % 2) == 0):
           error('<SpatialSubtractiveNormalization> averaging kernel must have ODD dimensions')

        # normalize kernel
        self.kernel.div_(self.kernel.sum() * self.nInputPlane)

        # padding values
        padH = math.floor(self.kernel.size(0)/2)
        padW = padH
        if kdim == 2:
           padW = math.floor(self.kernel.size(1)/2)

        # create convolutional mean extractor
        self.meanestimator = nn.Sequential()
        self.meanestimator.add(nn.SpatialZeroPadding(padW, padW, padH, padH))
        if kdim == 2:
            self.meanestimator.add(nn.SpatialConvolution(self.nInputPlane, 1, self.kernel.size(1), self.kernel.size(0)))
        else:
            # TODO: map
            self.meanestimator.add(nn.SpatialConvolutionMap(nn.SpatialConvolutionMap.maps.oneToOne(self.nInputPlane), self.kernel.size(0), 1))
            self.meanestimator.add(nn.SpatialConvolution(self.nInputPlane, 1, 1, self.kernel.size(0)))

        self.meanestimator.add(nn.Replicate(self.nInputPlane, 0))

        # set kernel and bias
        if kdim == 2:
            for i in range(self.nInputPlane):
                self.meanestimator.modules[1].weight[0][i] = self.kernel
            self.meanestimator.modules[1].bias.zero_()
        else:
            for i in range(self.nInputPlane):
                self.meanestimator.modules[1].weight[i].copy(self.kernel)
                self.meanestimator.modules[2].weight[0][i].copy(self.kernel)

            self.meanestimator.modules[1].bias.zero_()
            self.meanestimator.modules[2].bias.zero_()

        # other operation
        self.subtractor = nn.CSubTable()
        self.divider = nn.CDivTable()

        # coefficient array, to adjust side effects
        self.coef = torch.Tensor(1, 1, 1)

        self.ones = None
        self._coef = None

    def updateOutput(self, input):
        # compute side coefficients
        dim = input.dim()
        if input.dim() + 1 != self.coef.dim() or (input.size(dim-1) != self.coef.size(dim-1)) or (input.size(dim-2) != self.coef.size(dim-2)):
            self.ones = self.ones or input.new()
            self._coef = self._coef or self.coef.new()

            self.ones.resizeAs_(input[0:1]).fill_(1)
            coef = self.meanestimator.updateOutput(self.ones).squeeze(0)
            self._coef.resizeAs_(coef).copy(coef) # make contiguous for view
            size = coef.size().tolist()
            size = [input.size(0)] + size
            self.coef = self._coef.view(1, *(self._coef.size().tolist())).expand(*size)

        # compute mean
        self.localsums = self.meanestimator.updateOutput(input)
        self.adjustedsums = self.divider.updateOutput([self.localsums, self.coef])
        self.output = self.subtractor.updateOutput([input, self.adjustedsums])

        return self.output

    def updateGradInput(self, input, gradOutput):
        # resize grad
        self.gradInput.resizeAs_(input).zero_()

        # backprop through all modules
        gradsub = self.subtractor.updateGradInput([input, self.adjustedsums], gradOutput)
        graddiv = self.divider.updateGradInput([self.localsums, self.coef], gradsub[1])
        size = self.meanestimator.updateGradInput(input, graddiv[0]).size()
        self.gradInput.add_(self.meanestimator.updateGradInput(input, graddiv[0]))
        self.gradInput.add_(gradsub[0])

        return self.gradInput

    def clearState(self):
        nn.utils.clear(self, 'ones', '_coef')
        self.meanestimator.clearState()
        return super(SpatialSubtractiveNormalization, self).clearState()

