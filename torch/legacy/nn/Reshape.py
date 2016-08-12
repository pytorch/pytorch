import torch
from torch.legacy import nn

class Reshape(nn.Module):

    def __init__(self, *args):
        super(Reshape, self).__init__()

        self.size = torch.LongStorage()
        self.batchsize = torch.LongStorage()

        n = len(args)
        if n == 0 and isinstance(args[0], torch.LongStorage):
            self.size.resize_(args[0].size()).copy_(args[0])
        else:
            self.size.resize_(n)
            for i in range(n):
                self.size[i] = args[i]

        self.nelement = 1
        self.batchsize.resize_(self.size.size() + 1)
        for i, s in enumerate(self.size):
           self.nelement *= s
           self.batchsize[i+1] = self.size[i]

        self._input = None
        self._gradOutput = None

    def updateOutput(self, input):
        if not input.isContiguous():
           self._input = self._input or input.new()
           self._input.resizeAs_(input)
           self._input.copy_(input)
           input = self._input

        self.batchsize[0] = input.size(0)
        self.output = input.view(self.batchsize)

        return self.output

    def updateGradInput(self, input, gradOutput):
        if not gradOutput.isContiguous():
           self._gradOutput = self._gradOutput or gradOutput.new()
           self._gradOutput.resizeAs_(gradOutput)
           self._gradOutput.copy_(gradOutput)
           gradOutput = self._gradOutput

        self.gradInput = gradOutput.viewAs(input)
        return self.gradInput

    def __repr__(self):
        return super(Reshape, self).__repr__() + \
                '({})'.format('x'.join(map(lambda x: str(x), self.size)))

    def clearState(self):
        nn.utils.clear(self, '_input', '_gradOutput')
        return super(Reshape, self).clearState()

