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

        if self._output is None:
            self._output = input.new()
        if self.norm is None:
            self.norm = input.new()
        if self.buffer is None:
            self.buffer = input.new()

        self._output.resize_as_(input)

        # specialization for the infinity norm
        if self.p == float('inf'):
            if not self._indices:
                self._indices = torch.cuda.FloatTensor() if torch.typename(self.output) == 'torch.cuda.FloatTensor' \
                    else torch.LongTensor()

            torch.abs(input, out=self.buffer)
            torch.max(self._indices, self.buffer, 1, out=self.norm, keepdim=True)
            self.norm.add_(self.eps)
        else:
            if self.normp is None:
                self.normp = input.new()
            if self.p % 2 != 0:
                torch.abs(input, out=self.buffer).pow_(self.p)
            else:
                torch.pow(input, self.p, out=self.buffer)

            torch.sum(self.buffer, 1, out=self.normp, keepdim=True).add_(self.eps)
            torch.pow(self.normp, 1. / self.p, out=self.norm)

        torch.div(input, self.norm.view(-1, 1).expand_as(input), out=self._output)

        self.output = self._output.view(input_size)
        return self.output

    def updateGradInput(self, input, gradOutput):
        assert input.dim() == 2
        assert gradOutput.dim() == 2

        input_size = input.size()
        n = input.size(0)  # batch size
        d = input.size(1)  # dimensionality of vectors

        if self._gradInput is None:
            self._gradInput = input.new()
        if self.cross is None:
            self.cross = input.new()
        # compute diagonal term with gradOutput
        self._gradInput.resize_(n, d)
        if self.p == float('inf'):
                # specialization for the inf case
            torch.mul(self.norm.view(n, 1, 1).expand(n, d, 1), gradOutput, out=self._gradInput)
            self.buffer.resize_as_(input).zero_()
            self.cross.resize_(n, 1)
            torch.gather(input, 1, self._indices, out=self.cross)
            self.cross.div_(self.norm)
            self.buffer.scatter_(1, self._indices, self.cross)
        else:
            torch.mul(self.normp.view(n, 1).expand(n, d), gradOutput, out=self._gradInput)
            # small optimizations for different p
            # buffer = input*|input|^(p-2)
            # for non-even p, need to add absolute value
            if self.p % 2 != 0:
                if self.p < 2:
                    # add eps to avoid possible division by 0
                    torch.abs(input, out=self.buffer).add_(self.eps).pow_(self.p - 2).mul_(input)
                else:
                    torch.abs(input, out=self.buffer).pow_(self.p - 2).mul_(input)
            # special case for p == 2, pow(x, 0) = 1
            elif self.p == 2:
                self.buffer.copy_(input)
            else:
                # p is even and > 2, pow(x, p) is always positive
                torch.pow(input, self.p - 2, out=self.buffer).mul_(input)

        # compute cross term in two steps
        self.cross.resize_(n, 1)

        # instead of having a huge temporary matrix (b1*b2),
        #: the computations as b1*(b2*gradOutput). This avoids redundant
        # computation and also a huge buffer of size n*d^2
        if self.buffer2 is None:
            self.buffer2 = input.new()  # nxd
        torch.mul(input, gradOutput, out=self.buffer2)
        torch.sum(self.buffer2, 1, out=self.cross, keepdim=True)

        self.buffer.mul_(self.cross.expand_as(self.buffer))
        self._gradInput.add_(-1, self.buffer)

        # reuse cross buffer for normalization
        if self.p == float('inf'):
            torch.mul(self.norm, self.norm, out=self.cross)
        else:
            torch.mul(self.normp, self.norm, out=self.cross)

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
