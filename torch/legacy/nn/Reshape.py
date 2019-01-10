import torch
from .Module import Module
from .utils import clear


class Reshape(Module):

    def __init__(self, *args):
        super(Reshape, self).__init__()

        if len(args) == 0 and isinstance(args[0], torch.Size):
            self.size = args[0]
        else:
            self.size = torch.Size(args)

        self.nelement = 1
        for s in self.size:
            self.nelement *= s

        self._input = None
        self._gradOutput = None

    def updateOutput(self, input):
        if not input.is_contiguous():
            if self._input is None:
                self._input = input.new()
            self._input.resize_as_(input)
            self._input.copy_(input)
            input = self._input

        batchsize = [input.size(0)] + list(self.size)
        self.output = input.view(torch.Size(batchsize))

        return self.output

    def updateGradInput(self, input, gradOutput):
        if not gradOutput.is_contiguous():
            if self._gradOutput is None:
                self._gradOutput = gradOutput.new()
            self._gradOutput.resize_as_(gradOutput)
            self._gradOutput.copy_(gradOutput)
            gradOutput = self._gradOutput

        self.gradInput = gradOutput.view_as(input)
        return self.gradInput

    def __repr__(self):
        return super(Reshape, self).__repr__() + \
            '({})'.format('x'.join(map(lambda x: str(x), self.size)))

    def clearState(self):
        clear(self, '_input', '_gradOutput')
        return super(Reshape, self).clearState()
