import torch
from torch.legacy import nn

class Padding(nn.Module):
    # pad puts in [pad] amount of [value] over dimension [dim], starting at index [index] in that dimension. If pad<0, index counts from the left.  If pad>0 index counts from the right
    # index = 1 pads before index 1.  index = 2 pads starting before index 2 and after index 1 in dimension [dim]

    def __init__(self, dim, pad, value=0, index=0):
        self.value = value
        self.index = index
        self.dim = dim
        self.pad = pad
        self.outputSize = torch.LongStorage()
        super(Padding, self).__init__()


    def updateOutput(self, input):
        self.outputSize.resize(input.dim())
        self.outputSize.copy(input.size())
        dim = self.dim

        self.outputSize[dim] = self.outputSize[dim] + abs(self.pad)
        self.output.resize(self.outputSize)
        self.output.fill(self.value)
        index = self.index
        pad = self.pad
        if pad > 0:
           index = input.size(dim) - index
        else:
           pad = -pad

        if index == 0:
           self.output.narrow(dim, pad, input.size(dim)).copy(input)
        elif index == input.size(dim):
           self.output.narrow(dim, 0, input.size(dim)).copy(input)
        else:
           self.output.narrow(dim, 0, index).copy(input.narrow(dim, 0, index))
           self.output.narrow(dim, index + pad, input.size(dim) - index).copy(input.narrow(dim, index, input.size(dim) - index))

        return self.output


    def updateGradInput(self, input, gradOutput):
        self.gradInput.resizeAs(input)
        dim = self.dim

        index = self.index
        pad = self.pad
        if pad > 0:
           index = input.size(dim) - index
        else:
           pad = -pad

        if index == 0:
           self.gradInput.copy(gradOutput.narrow(dim, pad, input.size(dim)))
        elif index == input.size(dim):
           self.gradInput.copy(gradOutput.narrow(dim, 0, input.size(dim)))
        else:
           self.gradInput.narrow(dim, 0, index).copy(gradOutput.narrow(dim, 0, index))
           self.gradInput.narrow(dim, index, input.size(dim) - index).copy(gradOutput.narrow(dim, index + pad, input.size(dim) - index))

        return self.gradInput

