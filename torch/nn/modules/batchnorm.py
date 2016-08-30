import torch
from torch.autograd import Variable

from .module import Module

# TODO: check contiguous in THNN
class BatchNorm(Module):
    expected_dim = 2

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(BatchNorm, self).__init__()

        self.affine = affine
        self.eps = eps
        self.momentum = momentum
        self.running_mean = torch.zeros(num_features)
        self.running_var = torch.ones(num_features)

        if self.affine:
            self.weight = Variable(torch.Tensor(num_features))
            self.bias = Variable(torch.Tensor(num_features))
            self.reset_parameters()
        else:
            self.weight = None
            self.bias = None

    def reset_parameters(self):
        if self.weight:
            self.weight.data.uniform_()
        if self.bias:
            self.bias.data.zero_()

        self.running_mean.zero_()
        self.running_var.fill_(1)

    def _checkInputDim(self, input):
        if input.dim() != self.expected_dim:
            raise RuntimeError('only mini-batch supported ({}D tensor), got {}D tensor instead'.format(self.expected_dim, input.dim()))
        if input.size(1) != self.running_mean.nElement():
            raise RuntimeError('got {}-feature tensor, expected {}'.format(input.size(1), self.running_mean.nElement()))

    def __call__(self, input):
        self._checkInputDim(input)
        args = (input,)
        if self.weight is not None:
            args = args + (self.weight, self.bias)
        return self._backend.BatchNorm(self.running_mean,
                self.running_var, self.train, self.momentum, self.eps)(*args)[0]


class BatchNorm2d(BatchNorm):
    expected_dim = 4
