import torch
from torch.legacy import nn

class MaskedSelect(nn.Module):

    def __init__(self):
        super(MaskedSelect, self).__init__()
        self._maskIndices = torch.LongTensor()
        self._maskIndexBuffer = torch.LongTensor()
        self._maskIndexBufferCPU = torch.FloatTensor()
        self._gradBuffer = torch.Tensor()
        self._gradMask = torch.ByteTensor()

    def updateOutput(self, input):
        input, mask = input
        torch.maskedSelect(self.output, input, mask)
        return self.output

    def updateGradInput(self, input, gradOutput):
        input, mask = input
        if input.type() == 'torch.cuda.FloatTensor':
            torch.range(self._maskIndexBufferCPU, 0, mask.nElement()-1).resize_(mask.size())
            self._maskIndexBuffer.resize_(self._maskIndexBufferCPU.size()).copy_(self._maskIndexBufferCPU)
        else:
            torch.range(self._maskIndexBuffer, 0, mask.nElement()-1).resize_(mask.size())

        torch.maskedSelect(self._maskIndices, self._maskIndexBuffer, mask)
        self._gradBuffer.resize_(input.nElement()).zero_()
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
        return nn.utils.clear(self, ['output',
                                    'gradInput',
                                    '_maskIndexBuffer',
                                    '_maskIndexBufferCPU',
                                    '_maskIndices',
                                    '_gradBuffer',
                                    '_gradMask'])

