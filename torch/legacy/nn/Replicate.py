import torch
from .Module import Module


class Replicate(Module):

    def __init__(self, nf, dim=0):
        super(Replicate, self).__init__()
        self.nfeatures = nf
        self.dim = dim
        assert self.dim >= 0

    def updateOutput(self, input):
        assert self.dim < input.dim()

        size = list(input.size())
        size.insert(self.dim, self.nfeatures)

        stride = list(input.stride())
        stride.insert(self.dim, 0)

        self.output.set_(input.storage(), input.storage_offset(),
                         torch.Size(size), tuple(stride))
        return self.output

    def updateGradInput(self, input, gradOutput):
        self.gradInput.resize_as_(input).zero_()
        size = list(input.size())
        size.insert(self.dim, 1)

        gradInput = self.gradInput.view(*size)
        torch.sum(gradOutput, self.dim, out=gradInput)
        return self.gradInput
