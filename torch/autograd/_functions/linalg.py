import torch

from ..function import Function
from ..variable import Variable


class Diag(Function):

    @staticmethod
    def forward(ctx, input, diagonal_idx=0):
        ctx.diagonal_idx = diagonal_idx
        return input.diag(ctx.diagonal_idx)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.diag(ctx.diagonal_idx), None


class Tril(Function):

    @staticmethod
    def forward(ctx, input, diagonal_idx=0):
        ctx.diagonal_idx = diagonal_idx
        return input.tril(ctx.diagonal_idx)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.tril(ctx.diagonal_idx), None


class Triu(Function):

    @staticmethod
    def forward(ctx, input, diagnoal_idx=0):
        ctx.diagonal_idx = diagnoal_idx
        return input.triu(ctx.diagonal_idx)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.triu(ctx.diagonal_idx), None


class Trace(Function):

    @staticmethod
    def forward(ctx, input):
        ctx.isize = input.size()
        return input.new((input.trace(), ))

    @staticmethod
    def backward(ctx, grad_output):
        isize = ctx.isize
        min_size = min(isize)
        grad_input = Variable(grad_output.data.new(isize).zero_()).view(-1)
        grad_input[::(isize[1] + 1)] = grad_output.expand(min_size)
        return grad_input.view(isize)


class Cross(Function):

    @staticmethod
    def forward(ctx, input, other, dim=-1):
        ctx.dim = dim
        ctx.save_for_backward(input, other)
        return torch.cross(input, other, ctx.dim)

    @staticmethod
    def backward(ctx, grad_output):
        input, other = ctx.saved_variables
        grad_input = other.cross(grad_output, ctx.dim)
        grad_other = grad_output.cross(input, ctx.dim)
        return grad_input, grad_other, None


class Inverse(Function):

    @staticmethod
    def forward(ctx, input):
        inverse = torch.inverse(input)
        ctx.save_for_backward(inverse)
        return inverse

    @staticmethod
    def backward(ctx, grad_output):
        inverse, = ctx.saved_variables
        return -torch.mm(inverse.t(), torch.mm(grad_output, inverse.t()))


class Gesv(Function):

    @staticmethod
    def forward(ctx, b, a):
        # TODO see if one can backprop through LU
        X, LU = torch.gesv(b, a)
        ctx.save_for_backward(X, a)
        ctx.mark_non_differentiable(LU)
        return X, LU

    @staticmethod
    def backward(ctx, grad_output, grad_LU=None):
        X, a = ctx.saved_variables
        grad_b, _ = torch.gesv(grad_output, a.t())
        grad_a = -torch.mm(grad_b, X.t())
        return grad_b, grad_a
