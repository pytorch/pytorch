import torch

from ..function import Function


class Diag(Function):

    def __init__(self, diagonal_idx=0):
        super(Diag, self).__init__()
        self.diagonal_idx = diagonal_idx

    def forward(self, input):
        return input.diag()

    def backward(self, grad_output):
        return grad_output.diag()


class Tril(Function):

    def __init__(self, diagonal_idx=0):
        super(Tril, self).__init__()
        self.diagonal_idx = diagonal_idx

    def forward(self, input):
        return input.tril(self.diagonal_idx)

    def backward(self, grad_output):
        return grad_output.tril(self.diagonal_idx)


class Triu(Function):

    def __init__(self, diagonal_idx=0):
        super(Triu, self).__init__()
        self.diagonal_idx = diagonal_idx

    def forward(self, input):
        return input.triu(self.diagonal_idx)

    def backward(self, grad_output):
        return grad_output.triu(self.diagonal_idx)

# TODO: trace
