import math
import torch
from .Module import Module


class SpatialFractionalMaxPooling(Module):
    # Usage:
    # nn.SpatialFractionalMaxPooling(poolSizeW, poolSizeH, outW, outH)
    #   the output should be the exact size (outH x outW)
    # nn.SpatialFractionalMaxPooling(poolSizeW, poolSizeH, ratioW, ratioH)
    #   the output should be the size (floor(inH x ratioH) x floor(inW x ratioW))
    #   ratios are numbers between (0, 1) exclusive

    def __init__(self, poolSizeW, poolSizeH, arg1, arg2):
        super(SpatialFractionalMaxPooling, self).__init__()
        assert poolSizeW >= 2
        assert poolSizeH >= 2

        # Pool size (how wide the pooling for each output unit is)
        self.poolSizeW = poolSizeW
        self.poolSizeH = poolSizeH

        # Random samples are drawn for all
        # batch * plane * (height, width; i.e., 2) points. This determines
        # the 2d "pseudorandom" overlapping pooling regions for each
        # (batch element x input plane). A new set of random samples is
        # drawn every updateOutput call, unless we disable it via
        # .fixPoolingRegions().
        self.randomSamples = None

        # Flag to disable re-generation of random samples for producing
        # a new pooling. For testing purposes
        self.newRandomPool = False

        self.indices = None

        if arg1 >= 1 and arg2 >= 1:
            # Desired output size: the input tensor will determine the reduction
            # ratio
            self.outW = arg1
            self.outH = arg2
            self.ratioW = self.ratioH = None
        else:
            # Reduction ratio specified per each input
            # This is the reduction ratio that we use
            self.ratioW = arg1
            self.ratioH = arg2
            self.outW = self.outH = None

            # The reduction ratio must be between 0 and 1
            assert self.ratioW > 0 and self.ratioW < 1
            assert self.ratioH > 0 and self.ratioH < 1

    def _getBufferSize(self, input):
        assert input.ndimension() == 4
        batchSize = input.size(0)
        planeSize = input.size(1)

        return torch.Size([batchSize, planeSize, 2])

    def _initSampleBuffer(self, input):
        sampleBufferSize = self._getBufferSize(input)

        if self.randomSamples is None:
            self.randomSamples = input.new().resize_(sampleBufferSize).uniform_()
        elif self.randomSamples.size(0) != sampleBufferSize[0] or self.randomSamples.size(1) != sampleBufferSize[1]:
            self.randomSamples.resize_(sampleBufferSize).uniform_()
        elif not self.newRandomPool:
            # Create new pooling windows, since this is a subsequent call
            self.randomSamples.uniform_()

    def _getOutputSizes(self, input):
        outW = self.outW
        outH = self.outH
        if self.ratioW is not None and self.ratioH is not None:
            assert input.ndimension() == 4
            outW = int(math.floor(input.size(3) * self.ratioW))
            outH = int(math.floor(input.size(2) * self.ratioH))

            # Neither can be smaller than 1
            assert outW > 0
            assert outH > 0
        else:
            assert outW is not None and outH is not None

        return outW, outH

    # Call this to turn off regeneration of random pooling regions each
    # updateOutput call.
    def fixPoolingRegions(self, val=True):
        self.newRandomPool = val
        return self

    def updateOutput(self, input):
        if self.indices is None:
            self.indices = input.new()
        self.indices = self.indices.long()
        self._initSampleBuffer(input)
        outW, outH = self._getOutputSizes(input)

        self._backend.SpatialFractionalMaxPooling_updateOutput(
            self._backend.library_state,
            input,
            self.output,
            outW, outH, self.poolSizeW, self.poolSizeH,
            self.indices, self.randomSamples)
        return self.output

    def updateGradInput(self, input, gradOutput):
        assert self.randomSamples is not None
        outW, outH = self._getOutputSizes(input)

        self._backend.SpatialFractionalMaxPooling_updateGradInput(
            self._backend.library_state,
            input,
            gradOutput,
            self.gradInput,
            outW, outH, self.poolSizeW, self.poolSizeH,
            self.indices)
        return self.gradInput

    # backward compat
    def empty(self):
        self.clearState()

    def clearState(self):
        self.indices = None
        self.randomSamples = None
        return super(SpatialFractionalMaxPooling, self).clearState()

    def __repr__(self):
        return super(SpatialFractionalMaxPooling, self).__repr__() + \
            '({}x{}, {}, {})'.format(self.outW or self.ratioW,
                                     self.outH or self.ratioH,
                                     self.poolSizeW, self.poolSizeH)
