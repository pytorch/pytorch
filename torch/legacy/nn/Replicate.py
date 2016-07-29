import torch
from torch.legacy import nn

class Replicate(nn.Module):

    def __init__(self, nf, dim=0):
        super(Replicate, self).__init__()
        self.nfeatures = nf
        self.dim = dim
        assert self.dim >= 0

    def updateOutput(self, input):
        assert self.dim < input.dim()

        sz = torch.LongStorage(input.dim()+1)
        sz[self.dim] = self.nfeatures
        for i in range(input.dim()):
            offset = 1 if i >= self.dim else 0
            sz[i+offset] = input.size(i)

        st = torch.LongStorage(input.dim()+1)
        st[self.dim] = 0
        for i in range(input.dim()):
            offset = 1 if i >= self.dim else 0
            st[i+offset] = input.stride(i)

        self.output.set(input.storage(), input.storageOffset(), sz, st)
        return self.output

    def updateGradInput(self, input, gradOutput):
        self.gradInput.resizeAs(input).zero()
        sz = torch.LongStorage(input.dim()+1)
        sz[self.dim] = 1
        for i in range(input.dim()):
           offset = 1 if i >= self.dim else 0
           sz[i+offset] = input.size(i)

        gradInput = self.gradInput.view(sz)
        gradInput.sum(gradOutput, self.dim)
        return self.gradInput

