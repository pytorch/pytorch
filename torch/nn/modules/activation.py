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

    def forward(self, input):
        return self._backend.Threshold(self.threshold, self.value, self.inplace)(input)


class ReLU(Threshold):

    def __init__(self, inplace=False):
        super(ReLU, self).__init__(0, 0, inplace)


class Hardtanh(Module):

    def __init__(self, min_value=-1, max_value=1, inplace=False):
        super(Hardtanh, self).__init__()
        self.min_val = min_value
        self.max_val = max_value
        self.inplace = inplace
        assert self.max_val > self.min_val

    def forward(self, input):
        return self._backend.Hardtanh(self.min_val, self.max_val, self.inplace)(input)


class ReLU6(Hardtanh):

    def __init__(self, inplace=False):
        super(ReLU6, self).__init__(0, 6, inplace)


class Sigmoid(Module):

    def forward(self, input):
        return self._backend.Sigmoid()(input)


class Tanh(Module):

    def forward(self, input):
        return self._backend.Tanh()(input)


class Softmax(Module):

    def forward(self, input):
        assert input.dim() == 2, 'Softmax requires a 2D tensor as input'
        return self._backend.Softmax()(input)


class Softmax2d(Module):

    def forward(self, input):
        assert input.dim() == 4, 'Softmax2d requires a 4D tensor as input'
        return self._backend.Softmax()(input)

class LogSoftmax(Module):

    def forward(self, input):
        return self._backend.LogSoftmax()(input)


class ELU(Module):

    def __init__(self, alpha=1., inplace=False):
        super(ELU, self).__init__()
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, input):
        return self._backend.ELU(self.alpha, self.inplace)(input)


class Hardshrink(Module):

    def __init__(self, lambd=0.5):
        super(Hardshrink, self).__init__()
        self.lambd = lambd

    def forward(self, input):
        return self._backend.Hardshrink(self.lambd)(input)


class LeakyReLU(Module):

    def __init__(self, negative_slope=1e-2, inplace=False):
        super(LeakyReLU, self).__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace

    def forward(self, input):
        return self._backend.LeakyReLU(self.negative_slope, self.inplace)(input)


class LogSigmoid(Module):

    def forward(self, input):
        return self._backend.LogSigmoid()(input)


class Softplus(Module):

    def __init__(self, beta=1):
        super(Softplus, self).__init__()
        self.beta = beta
        self.threshold = 20

    def forward(self, input):
        return self._backend.Softplus(self.beta, self.threshold)(input)


class Softshrink(Module):

    def __init__(self, lambd=0.5):
        super(Softshrink, self).__init__()
        self.lambd = lambd

    def forward(self, input):
        return self._backend.Softshrink(self.lambd)(input)


class PReLU(Module):

    def __init__(self, num_parameters=1):
        super(PReLU, self).__init__()
        self.num_parameters = num_parameters
        self.weight = Variable(torch.Tensor(num_parameters).fill_(0.25))

    def forward(self, input):
        return self._backend.PReLU()(input, self.weight)


class Softsign(Module):

    def forward(self, input):
        return self._backend.Softsign()(input)


class Softmin(Module):

    def forward(self, input):
        return self._backend.Softmin()(input)


class Tanhshrink(Module):

    def forward(self, input):
        tanh = self._backend.Tanh()(input)
        return input - tanh


# TODO: RReLU

