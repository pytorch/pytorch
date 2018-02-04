import torch
from .Module import Module
from torch.legacy.nn.View import View
import numpy

class PixelShuffle(Module):
    def __init__(self, upscaleFactor):
        super(PixelShuffle, self).__init__()
        self.upscaleFactor = upscaleFactor
        self.upscaleFactorSquared = self.upscaleFactor * self.upscaleFactor
        self._intermediateShape = None
        self._outShape = None
        self._shuffleOut = None

    def updateOutput(self, input):

        if hasattr(self, '_intermediateShape') is False:
            self._intermediateShape = None
        if hasattr(self, '_outShape') is False:
            self._outShape = None
        if hasattr(self, '_shuffleOut') is False:
            self._shuffleOut = None

        if self._intermediateShape is None:
            self._intermediateShape = torch.Tensor(6, 0)
        if self._outShape is None:
            self._outShape = torch.Tensor()
        if self._shuffleOut is None:
            self._shuffleOut = input.new()

        batched = False
        batchSize = 1
        inputStartIdx = 0
        outShapeIdx = 0

        if len(input.size()) == 4:
            batched = True
            batchSize = input.size(0)
            inputStartIdx = 1
            outShapeIdx = 1
            self._outShape = torch.Tensor([0, 0, 0, 0])
            self._outShape[0] = batchSize

        else:
            self._outShape = torch.Tensor([0, 0, 0])

        channels = input.size(inputStartIdx) / self.upscaleFactorSquared
        inHeight = input.size(inputStartIdx + 1)
        inWidth = input.size(inputStartIdx + 2)

        (self._intermediateShape)[0] = batchSize
        (self._intermediateShape)[1] = channels
        (self._intermediateShape)[2] = self.upscaleFactor
        (self._intermediateShape)[3] = self.upscaleFactor
        (self._intermediateShape)[4] = inHeight
        (self._intermediateShape)[5] = inWidth

        self._outShape[outShapeIdx] = channels
        self._outShape[outShapeIdx + 1] = inHeight * self.upscaleFactor
        self._outShape[outShapeIdx + 2] = inWidth * self.upscaleFactor

        tmp = None
        tmp = self._intermediateShape.type(torch.IntTensor)
        tmp = torch.Size((tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5]))
        inputView = input.view(tmp)

        self._shuffleOut.resize_(inputView.size(0), inputView.size(1),
                                 inputView.size(4), inputView.size(2),
                                 inputView.size(5), inputView.size(3))
        self._shuffleOut.copy_(inputView.permute(0, 1, 4, 2, 5, 3))

        tmp = None
        if len(input.size()) == 4:
            tmp = self._outShape.type(torch.IntTensor)
            tmp = torch.Size((tmp[0], tmp[1], tmp[2], tmp[3]))
        else:
            tmp = self._outShape.type(torch.IntTensor)
            tmp = torch.Size((tmp[0], tmp[1], tmp[2]))
        self.output = self._shuffleOut.view(tmp)

        return self.output

    def updateGradInput(self, input, gradOutput):

        if hasattr(self, '_intermediateShape') is False:
            self._intermediateShape = None
        if hasattr(self, '_shuffleIn') is False:
            self._shuffleIn = None

        if self._intermediateShape is None:
            self._intermediateShape = torch.Tensor(6, 0)
        if self._shuffleIn is None:
            self._shuffleIn = input.new()

        batchSize = 1
        inputStartIdx = 0
        if len(input.size()) == 4:
            batchSize = input.size(0)
            inputStartIdx = 1
        end

        channels = input.size(inputStartIdx) / self.upscaleFactorSquared
        height = input.size(inputStartIdx + 1)
        width = input.size(inputStartIdx + 2)

        self._intermediateShape[0] = batchSize
        self._intermediateShape[1] = channels
        self._intermediateShape[2] = height
        self._intermediateShape[3] = self.upscaleFactor
        self._intermediateShape[4] = width
        self._intermediateShape[5] = self.upscaleFactor

        tmp = None
        tmp = self._intermediateShape.type(torch.IntTensor)
        tmp = torch.Size((tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5]))
        gradOutputView = gradOutput.view(tmp)

        self._shuffleIn.resize_(gradOutputView.size(0), gradOutputView.size(1),
                                gradOutputView.size(3), gradOutputView.size(5),
                                gradOutputView.size(2), gradOutputView.size(4))
        self._shuffleIn.copy_(gradOutputView.permute(0, 1, 3, 5, 2, 4))

        self.gradInput = self._shuffleIn.view(input.size())

        return self.gradInput

    def clearState(self):
        clear(self, '_intermediateShape', '_outShape', '_shuffleIn', '_shuffleOut')
        return super(PixelShuffle, self).clearState()
