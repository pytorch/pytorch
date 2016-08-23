import torch
from torch.autograd import Variable

from .module import Module


class Threshold(Module):

    def __init__(self, threshold, value, inplace=False):
        super(Threshold, self).__init__()
        self.threshold = threshold
        self.value = value
        self.inplace = inplace
        # TODO: check in THNN (if inplace == True, then assert value <= threshold)

    def _forward(self, input):
        return self._backend.Threshold(self.threshold, self.value, self.inplace)(input)


class ReLU(Threshold):

    def __init__(self, inplace=False):
        super(ReLU, self).__init__(0, 0, inplace)


class HardTanh(Module):

    def __init__(self, min_value=-1, max_value=1, inplace=False):
        super(HardTanh, self).__init__()
        self.min_val = min_value
        self.max_val = max_value
        self.inplace = inplace
        assert self.max_val > self.min_val

    def _forward(self, input):
        return self._backend.HardTanh(self.min_val, self.max_val, self.inplace)(input)


class ReLU6(HardTanh):

    def __init__(self, inplace=False):
        super(ReLU6, self).__init__(0, 6, inplace)

class Sigmoid(Module):

    def _forward(self, input):
        return self._backend.Sigmoid()(input)

class Tanh(Module):

    def _forward(self, input):
        return self._backend.Tanh()(input)
