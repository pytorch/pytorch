import torch
from .Module import Module
from .utils import clear

class Normalize(Module):

    def __init__(self, p, eps=1e-10):
        super(Normalize, self).__init__()
        assert p > 0
        self.p = p
        self.eps = eps

        self._output = None
        self.norm = None
        self.buffer = None
        self._indices = None
        self.normp = None
        self._gradInput = None
        self.cross = None
        self.buffer2 = None


    def updateOutput(self, input):
        assert input.dim() == 2
        input_size = input.size()

        self._output = self._output or input.new()
        self.norm = self.norm or input.new()
        self.buffer = self.buffer or input.new()

        self._output.resizeAs_(input)

        # specialization for the infinity norm
        if self.p == float('inf'):
            if not self._indices:
                self._indices = torch.cuda.FloatTensor() if torch.typename(self.output) == 'torch.cuda.FloatTensor' \
                    else torch.LongTensor()

            torch.abs(self.buffer, input)
            torch.max(self.norm, self._indices, self.buffer, 1)
            self.norm.add_(self.eps)
        else:
            self.normp = self.normp or input.new()
            if self.p % 2 != 0:
                torch.abs(self.buffer, input).pow_(self.p)
            else:
                torch.pow(self.buffer, input, self.p)

            torch.sum(self.normp, self.buffer, 1).add_(self.eps)
            torch.pow(self.norm, self.normp, 1./self.p)

        torch.div(self._output, input, self.norm.view(-1, 1).expandAs(input))

        self.output = self._output.view(input_size)
        return self.output

    def updateGradInput(self, input, gradOutput):
        assert input.dim() == 2
        assert gradOutput.dim() == 2

        input_size = input.size()
        n = input.size(0) # batch size
        d = input.size(1) # dimensionality of vectors

        self._gradInput = self._gradInput or input.new()
        self.cross = self.cross or input.new()
        # compute diagonal term with gradOutput
        self._gradInput.resize_(n, d)
        if self.p == float('inf'):
                # specialization for the inf case
                torch.mul(self._gradInput, self.norm.view(n, 1,1).expand(n, d,1), gradOutput)
                self.buffer.resizeAs_(input).zero_()
                self.cross.resize_(n, 1)
                torch.gather(self.cross, input, 1, self._indices)
                self.cross.div_(self.norm)
                self.buffer.scatter_(1, self._indices, self.cross)
        else:
                torch.mul(self._gradInput, self.normp.view(n, 1).expand(n, d), gradOutput)
                # small optimizations for different p
                # buffer = input*|input|^(p-2)
                # for non-even p, need to add absolute value
                if self.p % 2 != 0:
                    if self.p < 2:
                        # add eps to avoid possible division by 0
                        torch.abs(self.buffer, input).add_(self.eps).pow_(self.p-2).mul_(input)
                    else:
                        torch.abs(self.buffer, input).pow_(self.p-2).mul_(input)
                # special case for p == 2, pow(x, 0) = 1
                elif self.p == 2:
                    self.buffer.copy_(input)
                else:
                    # p is even and > 2, pow(x, p) is always positive
                    torch.pow(self.buffer, input, self.p-2).mul_(input)

        # compute cross term in two steps
        self.cross.resize_(n, 1)

        # instead of having a huge temporary matrix (b1*b2),
        #: the computations as b1*(b2*gradOutput). This avoids redundant
        # computation and also a huge buffer of size n*d^2
        self.buffer2 = self.buffer2 or input.new() # nxd
        torch.mul(self.buffer2, input, gradOutput)
        torch.sum(self.cross, self.buffer2, 1)

        self.buffer.mul_(self.cross.expandAs(self.buffer))
        self._gradInput.add_(-1, self.buffer)

        # reuse cross buffer for normalization
        if self.p == float('inf'):
            torch.mul(self.cross, self.norm, self.norm)
        else:
            torch.mul(self.cross, self.normp, self.norm)

        self._gradInput.div_(self.cross.expand(n, d))

        self.gradInput = self._gradInput.view(input_size)
        return self.gradInput

    def __repr__(self):
        return super(Normalize, self).__repr__() + '({})'.format(self.p)

    def type(self, type, tensorCache=None):
        if not type:
            return self._type
        # torch.max expects a LongTensor as indices, whereas cutorch.max expects a CudaTensor.
        if type == 'torch.cuda.FloatTensor':
            super(Normalize, self).type(type, tensorCache)
        else:
            # self._indices must be a LongTensor. Setting it to nil temporarily avoids
            # unnecessary memory allocations.
            indices, self._indices = self._indices, None
            super(Normalize, self).type(type, tensorCache)
            self._indices = indices.long() if indices else None

        return self

    def clearState(self):
        clear(self, [
           '_output',
           '_indices',
           '_gradInput',
           'buffer',
           'norm',
           'normp',
           'cross',
        ])
        return super(Normalize, self).clearState()

