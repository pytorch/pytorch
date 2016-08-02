import torch
from torch.legacy import nn

class Normalize(nn.Module):

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

        self._output.resizeAs(input)

        # specialization for the infinity norm
        if self.p == float('inf'):
            if not self._indices:
                self._indices = torch.cuda.FloatTensor() if torch.typename(self.output) == 'torch.cuda.FloatTensor' \
                    else torch.LongTensor()

            self.buffer.abs(input)
            torch.max(self.norm, self._indices, self.buffer, 1)
            self.norm.add(self.eps)
        else:
            self.normp = self.normp or input.new()
            if self.p % 2 != 0:
                self.buffer.abs(input).pow(self.p)
            else:
                self.buffer.pow(input, self.p)

            self.normp.sum(self.buffer, 1).add(self.eps)
            self.norm.pow(self.normp, 1/self.p)

        self._output.cdiv(input, self.norm.view(-1, 1).expandAs(input))

        self.output.view(self._output, input_size)
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
        self._gradInput.resize(n, d)
        if self.p == float('inf'):
                # specialization for the inf case
                self._gradInput.cmul(self.norm.view(n, 1,1).expand(n, d,1), gradOutput)
                self.buffer.resizeAs(input).zero()
                self.cross.resize(n, 1)
                self.cross.gather(input, 1, self._indices)
                self.cross.cdiv(self.norm)
                self.buffer.scatter(1, self._indices, self.cross)
        else:
                self._gradInput.cmul(self.normp.view(n, 1).expand(n, d), gradOutput)
                # small optimizations for different p
                # buffer = input*|input|^(p-2)
                # for non-even p, need to add absolute value
                if self.p % 2 != 0:
                    if self.p < 2:
                        # add eps to avoid possible division by 0
                        self.buffer.abs(input).add(self.eps).pow(self.p-2).cmul(input)
                    else:
                        self.buffer.abs(input).pow(self.p-2).cmul(input)
                # special case for p == 2, pow(x, 0) = 1
                elif self.p == 2:
                    self.buffer.copy(input)
                else:
                    # p is even and > 2, pow(x, p) is always positive
                    self.buffer.pow(input, self.p-2).cmul(input)

        # compute cross term in two steps
        self.cross.resize(n, 1)

        # instead of having a huge temporary matrix (b1*b2),
        #: the computations as b1*(b2*gradOutput). This avoids redundant
        # computation and also a huge buffer of size n*d^2
        self.buffer2 = self.buffer2 or input.new() # nxd
        self.buffer2.cmul(input, gradOutput)
        self.cross.sum(self.buffer2, 1)

        self.buffer.cmul(self.cross.expandAs(self.buffer))
        self._gradInput.add(-1, self.buffer)

        # reuse cross buffer for normalization
        if self.p == float('inf'):
            self.cross.cmul(self.norm, self.norm)
        else:
            self.cross.cmul(self.normp, self.norm)

        self._gradInput.cdiv(self.cross.expand(n, d))

        self.gradInput.view(self._gradInput, input_size)
        return self.gradInput


    def __repr__(self):
        return super(Normalize, self).__repr__() + '({})'.format(self.p)


    def type(self, type, tensorCache):
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
        nn.utils.clear(self, [
           '_output',
           '_indices',
           '_gradInput',
           'buffer',
           'norm',
           'normp',
           'cross',
        ])
        return super(Normalize, self).clearState()

