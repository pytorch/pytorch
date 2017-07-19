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


class Symeig(Function):

    @staticmethod
    def forward(ctx, input, eigenvectors=False, upper=True):
        ctx.eigenvectors = eigenvectors
        ctx.upper = upper
        e, V = torch.symeig(input, eigenvectors=ctx.eigenvectors, upper=ctx.upper)
        ctx.save_for_backward(input, e, V)
        return e, V

    @staticmethod
    def backward(ctx, grad_e, grad_V):
        x, e, V, = ctx.saved_variables

        N = x.size(0)

        if ctx.upper:
            tri0 = torch.triu
            tri1 = lambda a: torch.tril(a, -1)
        else:
            tri0 = torch.tril
            tri1 = lambda a: torch.triu(a, 1)

        def G(n):
            return sum([V[:, m] * grad_V.t()[n].matmul(V[:, m]) / (e[n] - e[m])
                       for m in range(N) if m != n])

        g = sum([torch.ger(V[:, n], V[:, n] * grad_e[n] + G(n))
                 for n in range(N)])

        out = tri0(g) + tri1(g).t()

        return out, None, None
