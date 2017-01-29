####################################
# DepthConcat
# Concatenates the output of Convolutions along the depth dimension
# (nOutputFrame). This is used to implement the DepthConcat layer
# of the Going deeper with convolutions paper :
# http.//arxiv.org/pdf/1409.4842v1.pdf
# The normal Concat Module can't be used since the spatial dimensions
# of tensors to be concatenated may have different values. To deal with
# this, we select the largest spatial dimensions and add zero-padding
# around the smaller dimensions.
####################################

import math
import torch
from .Concat import Concat


class DepthConcat(Concat):

    def windowNarrow(self, output, currentOutput, offset):
        outputWindow = output.narrow(self.dimension, offset, currentOutput.size(self.dimension))
        for dim in range(len(self.outputSize)):
            currentSize = currentOutput.size(dim)
            if dim != self.dimension and self.outputSize[dim] != currentSize:
                # 5x5 vs 3x3 -> start = [(5-3)/2] + 1 = 2 (1 pad each side)
                # 9x9 vs 5x5 -> start = [(9-5)/2] + 1 = 3 (2 pad each side)
                # 9x9 vs 4x4 -> start = [(9-4)/2] + 1 = 3.5 (2 pad, 3 pad)
                start = int(math.floor(((self.outputSize[dim] - currentSize) / 2)))
                outputWindow = outputWindow.narrow(dim, start, currentSize)
        return outputWindow

    def updateOutput(self, input):
        outs = []
        for i in range(len(self.modules)):
            currentOutput = self.modules[i].updateOutput(input)
            outs.append(currentOutput)
            if i == 0:
                size = list(currentOutput.size())
            else:
                size[self.dimension] += currentOutput.size(self.dimension)
                for dim in range(len(self.outputSize)):
                    if dim != self.dimension:
                        # take the maximum size (shouldn't change anything for batch dim)
                        size[dim] = max(size[dim], currentOutput.size(dim))

        self.outputSize = torch.Size(size)
        self.output.resize_(self.outputSize).zero_()  # zero for padding

        offset = 0
        for i, module in enumerate(self.modules):
            currentOutput = outs[i]
            outputWindow = self.windowNarrow(self.output, currentOutput, offset)
            outputWindow.copy_(currentOutput)
            offset = offset + currentOutput.size(self.dimension)

        return self.output

    def updateGradInput(self, input, gradOutput):
        self.gradInput.resize_as_(input)

        offset = 0
        for i, module in enumerate(self.modules):
            currentOutput = module.output
            gradOutputWindow = self.windowNarrow(gradOutput, currentOutput, offset)
            currentGradInput = module.updateGradInput(input, gradOutputWindow)
            if i == 0:
                self.gradInput.copy_(currentGradInput)
            else:
                self.gradInput.add_(currentGradInput)

            offset += currentOutput.size(self.dimension)

        return self.gradInput

    def accGradParameters(self, input, gradOutput, scale=1):
        offset = 0
        for i, module in enumerate(self.modules):
            currentOutput = module.output
            gradOutputWindow = self.windowNarrow(gradOutput, currentOutput, offset)
            module.accGradParameters(input, gradOutputWindow, scale)
            offset += currentOutput.size(self.dimension)

    def backward(self, input, gradOutput, scale=1):
        self.gradInput.resize_as_(input)

        offset = 0
        for i, module in enumerate(self.modules):
            currentOutput = module.output
            gradOutputWindow = self.windowNarrow(gradOutput, currentOutput, offset)
            currentGradInput = module.backward(input, gradOutputWindow)
            if i == 0:
                self.gradInput.copy_(currentGradInput)
            else:
                self.gradInput.add_(currentGradInput)

            offset = offset + currentOutput.size(self.dimension)

        return self.gradInput

    def accUpdateGradParameters(self, input, gradOutput, lr):
        offset = 0
        for i, module in enumerate(self.modules):
            currentOutput = module.output
            gradOutputWindow = self.windowNarrow(gradOutput, currentOutput, offset)
            module.accUpdateGradParameters(input, gradOutputWindow, lr)
            offset = offset + currentOutput.size(self.dimension)
