import torch
from .Module import Module
from .utils import clear

class SpatialCrossMapLRN(Module):

    def __init__(self, size, alpha=1e-4, beta=0.75, k=1):
        super(SpatialCrossMapLRN, self).__init__()

        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.scale = None
        self.paddedRatio = None
        self.accumRatio = None

    def updateOutput(self, input):
        assert input.dim() == 4

        self.scale = self.scale or input.new()
        if input.type() == 'torch.cuda.FloatTensor':
            self._backend.SpatialCrossMapLRN_updateOutput(
                self._backend.library_state,
                input,
                self.output,
                self.scale,
                self.size,
                self.alpha,
                self.beta,
                self.k
            )
        else:
            batchSize   = input.size(0)
            channels    = input.size(1)
            inputHeight = input.size(2)
            inputWidth  = input.size(3)

            self.output.resizeAs_(input)
            self.scale.resizeAs_(input)

            # use output storage as temporary buffer
            inputSquare = self.output
            torch.pow(inputSquare, input, 2)

            prePad = int((self.size - 1)/2 + 1)
            prePadCrop = channels if prePad > channels else prePad

            scaleFirst = self.scale.select(1, 0)
            scaleFirst.zero_()
            # compute first feature map normalization
            for c in range(prePadCrop):
                scaleFirst.add_(inputSquare.select(1, c))

            # reuse computations for next feature maps normalization
            # by adding the next feature map and removing the previous
            for c in range(1, channels):
                scalePrevious = self.scale.select(1, c - 1)
                scaleCurrent  = self.scale.select(1, c)
                scaleCurrent.copy_(scalePrevious)
                if c < channels - prePad + 1:
                    squareNext   = inputSquare.select(1, c + prePad - 1)
                    scaleCurrent.add_(1, squareNext)

                if c > prePad:
                    squarePrevious = inputSquare.select(1, c - prePad)
                    scaleCurrent.add_(-1, squarePrevious)

            self.scale.mul_(self.alpha / self.size).add_(self.k)

            torch.pow(self.output, self.scale, -self.beta)
            self.output.mul_(input)

        return self.output

    def updateGradInput(self, input, gradOutput):
        assert input.dim() == 4

        if input.type() == 'torch.cuda.FloatTensor':
            self._backend.SpatialCrossMapLRN_updateGradInput(
                self._backend.library_state,
                input,
                gradOutput,
                self.gradInput,
                self.scale,
                self.output,
                self.size,
                self.alpha,
                self.beta,
                self.k
            )
        else:
            batchSize   = input.size(0)
            channels    = input.size(1)
            inputHeight = input.size(2)
            inputWidth  = input.size(3)

            self.paddedRatio = self.paddedRatio or input.new()
            self.accumRatio = self.accumRatio or input.new()
            self.paddedRatio.resize_(channels + self.size - 1, inputHeight, inputWidth)
            self.accumRatio.resize_(inputHeight, inputWidth)

            cacheRatioValue = 2 * self.alpha * self.beta / self.size
            inversePrePad = int(self.size - (self.size - 1) / 2)

            self.gradInput.resizeAs_(input)
            torch.pow(self.gradInput, self.scale, -self.beta).mul_(gradOutput)

            self.paddedRatio.zero_()
            paddedRatioCenter = self.paddedRatio.narrow(0, inversePrePad, channels)
            for n in range(batchSize):
                torch.mul(paddedRatioCenter, gradOutput[n], self.output[n])
                paddedRatioCenter.div_(self.scale[n])
                torch.sum(self.accumRatio, self.paddedRatio.narrow(0, 0,self.size-1), 0)
                for c in range(channels):
                    self.accumRatio.add_(self.paddedRatio[c+self.size-1])
                    self.gradInput[n][c].addcmul_(-cacheRatioValue, input[n][c], self.accumRatio)
                    self.accumRatio.add_(-1, self.paddedRatio[c])

        return self.gradInput

    def clearState(self):
        clear(self, 'scale', 'paddedRatio', 'accumRatio')
        return super(SpatialCrossMapLRN, self).clearState()

