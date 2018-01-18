import torch
from .Module import Module
from .utils import recursiveType
import torch._thnn


class Criterion(object):

    def __init__(self):
        self.gradInput = torch.Tensor()
        self.output = 0
        self._backend = torch._thnn.type2backend[self.gradInput.type()]

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
            setattr(self, key, recursiveType(param, type, tensorCache or {}))

        self._backend = torch._thnn.type2backend[type]
        return self

    def float(self):
        return self.type('torch.FloatTensor')

    def double(self):
        return self.type('torch.DoubleTensor')

    def cuda(self):
        return self.type('torch.cuda.FloatTensor')
