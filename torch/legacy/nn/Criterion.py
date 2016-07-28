import torch
from torch.legacy import nn
from torch.legacy.nn import ffi

class Criterion(object):

    def __init__(self):
        self.gradInput = torch.Tensor()
        self.output = 0
        self._backend = nn._backends.THNNDoubleBackend

    def updateOutput(self, input, target):
        raise NotImplementedError

    def forward(self, input, target):
        return self.updateOutput(input, target)

    def backward(self, input, target):
        return self.updateGradInput(input, target)

    def updateGradInput(self, input, target):
        raise NotImplementedError

    def clone(self):
        raise NotImplementedError

    def type(self, type, tensorCache=None):
        # find all tensors and convert them
        for key, param in self.__dict__.items():
            setattr(self, key, nn.utils.recursiveType(param, type, tensorCache or {}))

        self._backend = ffi.type2backend[type]
        return self

    def float(self):
        return self.type('torch.FloatTensor')

    def double(self):
        return self.type('torch.DoubleTensor')

    def cuda(self):
        return self.type('torch.CudaTensor')

