import torch
from .Module import Module


class Padding(Module):
    # pad puts in [pad] amount of [value] over dimension [dim], starting at
    # index [index] in that dimension. If pad<0, index counts from the left.
    # If pad>0 index counts from the right index = 1 pads before index 1.
    # index = 2 pads starting before index 2 and after index 1 in dimension [dim]

    def __init__(self, dim, pad, value=0, index=0):
        self.value = value
        self.index = index
        self.dim = dim
        self.pad = pad
        self.outputSize = torch.Size()
        super(Padding, self).__init__()

    def updateOutput(self, input):
        outputSize = list(input.size())
        outputSize[self.dim] += abs(self.pad)
        self.outputSize = torch.Size(outputSize)
        dim = self.dim

        self.output.resize_(self.outputSize)
        self.output.fill_(self.value)
        index = self.index
        pad = self.pad
        if pad > 0:
            index = input.size(dim) - index
        else:
            pad = -pad

        if index == 0:
            self.output.narrow(dim, pad, input.size(dim)).copy_(input)
        elif index == input.size(dim):
            self.output.narrow(dim, 0, input.size(dim)).copy_(input)
        else:
            self.output.narrow(dim, 0, index).copy_(input.narrow(dim, 0, index))
            self.output.narrow(dim, index + pad, input.size(dim) -
                               index).copy_(input.narrow(dim, index, input.size(dim) - index))

        return self.output

    def updateGradInput(self, input, gradOutput):
        self.gradInput.resize_as_(input)
        dim = self.dim

        index = self.index
        pad = self.pad
        if pad > 0:
            index = input.size(dim) - index
        else:
            pad = -pad

        if index == 0:
            self.gradInput.copy_(gradOutput.narrow(dim, pad, input.size(dim)))
        elif index == input.size(dim):
            self.gradInput.copy_(gradOutput.narrow(dim, 0, input.size(dim)))
        else:
            self.gradInput.narrow(dim, 0, index).copy_(gradOutput.narrow(dim, 0, index))
            self.gradInput.narrow(dim, index, input.size(
                dim) - index).copy_(gradOutput.narrow(dim, index + pad, input.size(dim) - index))

        return self.gradInput
