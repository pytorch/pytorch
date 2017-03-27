import torch
from .Module import Module
from .utils import clear


class MaskedSelect(Module):

    def __init__(self):
        super(MaskedSelect, self).__init__()
        self._maskIndices = torch.LongTensor()
        self._maskIndexBuffer = torch.LongTensor()
        self._maskIndexBufferCPU = torch.FloatTensor()
        self._gradBuffer = torch.Tensor()
        self._gradMask = torch.ByteTensor()

    def updateOutput(self, input):
        input, mask = input
        torch.masked_select(input, mask, out=self.output)
        return self.output

    def updateGradInput(self, input, gradOutput):
        input, mask = input
        if input.type() == 'torch.cuda.FloatTensor':
            torch.range(0, mask.nelement() - 1, out=self._maskIndexBufferCPU).resize_(mask.size())
            self._maskIndexBuffer.resize_(self._maskIndexBufferCPU.size()).copy_(self._maskIndexBufferCPU)
        else:
            torch.range(0, mask.nelement() - 1, out=self._maskIndexBuffer).resize_(mask.size())

        torch.masked_select(self._maskIndexBuffer, mask, out=self._maskIndices)
        self._gradBuffer.resize_(input.nelement()).zero_()
        self._gradBuffer.scatter_(0, self._maskIndices, gradOutput)
        self._gradBuffer.resize_(input.size())
        self.gradInput = [self._gradBuffer, self._gradMask.resize_(mask.size()).fill_(0)]
        return self.gradInput

    def type(self, type=None, tensorCache=None):
        if type is None:
            return self._type

        self._gradBuffer = self._gradBuffer.type(type)
        self.gradInput = self.gradInput.type(type)
        self.output = self.output.type(type)

        # These casts apply when switching between cuda/non-cuda types
        if type != 'torch.cuda.FloatTensor':
            self._maskIndexBuffer = self._maskIndexBuffer.long()
            self._maskIndices = self._maskIndices.long()
            self._gradMask = self._gradMask.byte()
        else:
            self._maskIndexBuffer = self._maskIndexBuffer.cuda()
            self._maskIndices = self._maskIndices.cuda()
            self._gradMask = self._gradMask.cuda()

        self._type = type
        return self

    def clearState(self):
        return clear(self, ['output',
                            'gradInput',
                            '_maskIndexBuffer',
                            '_maskIndexBufferCPU',
                            '_maskIndices',
                            '_gradBuffer',
                            '_gradMask'])
