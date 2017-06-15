import torch

from ..function import Function, InplaceFunction
from .utils import maybe_view, maybe_unexpand, maybe_unexpand_or_view

# TODO: no need to save all args if the grad w.r.t. some of them is not needed
def _get_output(ctx, arg, inplace=False):
    if inplace:
        ctx.mark_dirty(arg)
        return arg
    else:
        return arg.new().resize_as_(arg)


class Addmm(InplaceFunction):

    @staticmethod
    def forward(ctx, add_matrix, matrix1, matrix2, alpha=1, beta=1, inplace=False):
        ctx.alpha = alpha
        ctx.beta = beta
        ctx.add_matrix_size = add_matrix.size()
        ctx.save_for_backward(matrix1, matrix2)
        output = _get_output(ctx, add_matrix, inplace=inplace)
        return torch.addmm(alpha, add_matrix, beta,
                           matrix1, matrix2, out=output)

    @staticmethod
    def backward(ctx, grad_output):
        matrix1, matrix2 = ctx.saved_variables
        grad_add_matrix = grad_matrix1 = grad_matrix2 = None

        if ctx.needs_input_grad[0]:
            grad_add_matrix = maybe_unexpand(grad_output, ctx.add_matrix_size)
            if ctx.alpha != 1:
                grad_add_matrix = grad_add_matrix.mul(ctx.alpha)

        if ctx.needs_input_grad[1]:
            grad_matrix1 = torch.mm(grad_output, matrix2.t())
            if ctx.beta != 1:
                grad_matrix1 *= ctx.beta

        if ctx.needs_input_grad[2]:
            grad_matrix2 = torch.mm(matrix1.t(), grad_output)
            if ctx.beta != 1:
                grad_matrix2 *= ctx.beta

        return grad_add_matrix, grad_matrix1, grad_matrix2, None, None, None


class Addbmm(InplaceFunction):

    @staticmethod
    def forward(ctx, add_matrix, batch1, batch2, alpha=1, beta=1, inplace=False):
        ctx.alpha = alpha
        ctx.beta = beta
        ctx.add_matrix_size = add_matrix.size()
        ctx.save_for_backward(batch1, batch2)
        output = _get_output(ctx, add_matrix, inplace=inplace)
        return torch.addbmm(alpha, add_matrix, beta,
                            batch1, batch2, out=output)

    @staticmethod
    def backward(ctx, grad_output):
        batch1, batch2 = ctx.saved_variables
        grad_add_matrix = grad_batch1 = grad_batch2 = None

        if ctx.needs_input_grad[0]:
            grad_add_matrix = maybe_unexpand(grad_output, ctx.add_matrix_size)
            if ctx.alpha != 1:
                grad_add_matrix = grad_add_matrix.mul(ctx.alpha)

        if any(ctx.needs_input_grad[1:]):
            batch_grad_output = (grad_output
                                 .unsqueeze(0)
                                 .expand(batch1.size(0), batch1.size(1), batch2.size(2)))

        if ctx.needs_input_grad[1]:
            grad_batch1 = torch.bmm(batch_grad_output, batch2.transpose(1, 2))
            if ctx.beta != 1:
                grad_batch1 *= ctx.beta

        if ctx.needs_input_grad[2]:
            grad_batch2 = torch.bmm(batch1.transpose(1, 2), batch_grad_output)
            if ctx.beta != 1:
                grad_batch2 *= ctx.beta

        return grad_add_matrix, grad_batch1, grad_batch2, None, None, None


class Baddbmm(InplaceFunction):

    @staticmethod
    def forward(ctx, add_batch, batch1, batch2, alpha=1, beta=1, inplace=False):
        ctx.alpha = alpha
        ctx.beta = beta
        ctx.add_batch_size = add_batch.size()
        ctx.save_for_backward(batch1, batch2)
        output = _get_output(ctx, add_batch, inplace=inplace)
        return torch.baddbmm(alpha, add_batch, beta,
                             batch1, batch2, out=output)

    @staticmethod
    def backward(ctx, grad_output):
        batch1, batch2 = ctx.saved_variables
        grad_add_batch = grad_batch1 = grad_batch2 = None

        if ctx.needs_input_grad[0]:
            grad_add_batch = maybe_unexpand(grad_output, ctx.add_batch_size)
            if ctx.alpha != 1:
                grad_add_batch = grad_add_batch.mul(ctx.alpha)

        if ctx.needs_input_grad[1]:
            grad_batch1 = torch.bmm(grad_output, batch2.transpose(1, 2))
            if ctx.beta != 1:
                grad_batch1 *= ctx.beta

        if ctx.needs_input_grad[2]:
            grad_batch2 = torch.bmm(batch1.transpose(1, 2), grad_output)
            if ctx.beta != 1:
                grad_batch2 *= ctx.beta

        return grad_add_batch, grad_batch1, grad_batch2, None, None, None


class Addmv(InplaceFunction):

    @staticmethod
    def forward(ctx, add_vector, matrix, vector, alpha=1, beta=1, inplace=False):
        ctx.alpha = alpha
        ctx.beta = beta
        ctx.add_vector_size = add_vector.size()
        ctx.save_for_backward(matrix, vector)
        output = _get_output(ctx, add_vector, inplace=inplace)
        return torch.addmv(alpha, add_vector, beta,
                           matrix, vector, out=output)

    @staticmethod
    def backward(ctx, grad_output):
        matrix, vector = ctx.saved_variables
        grad_add_vector = grad_matrix = grad_vector = None

        if ctx.needs_input_grad[0]:
            grad_add_vector = maybe_unexpand(grad_output, ctx.add_vector_size)
            if ctx.alpha != 1:
                grad_add_vector = grad_add_vector.mul(ctx.alpha)

        if ctx.needs_input_grad[1]:
            grad_matrix = torch.ger(grad_output, vector)
            if ctx.beta != 1:
                grad_matrix *= ctx.beta

        if ctx.needs_input_grad[2]:
            grad_vector = torch.mv(matrix.t(), grad_output)
            if ctx.beta != 1:
                grad_vector *= ctx.beta

        return grad_add_vector, grad_matrix, grad_vector, None, None, None


class Addr(InplaceFunction):

    @staticmethod
    def forward(ctx, add_matrix, vector1, vector2, alpha=1, beta=1, inplace=False):
        ctx.alpha = alpha
        ctx.beta = beta
        ctx.add_matrix_size = add_matrix.size()
        ctx.save_for_backward(vector1, vector2)
        output = _get_output(ctx, add_matrix, inplace=inplace)
        return torch.addr(alpha, add_matrix, beta,
                          vector1, vector2, out=output)

    @staticmethod
    def backward(ctx, grad_output):
        vector1, vector2 = ctx.saved_variables
        grad_add_matrix = grad_vector1 = grad_vector2 = None

        if ctx.needs_input_grad[0]:
            grad_add_matrix = maybe_unexpand(grad_output, ctx.add_matrix_size)
            if ctx.alpha != 1:
                grad_add_matrix = grad_add_matrix.mul(ctx.alpha)

        if ctx.needs_input_grad[1]:
            grad_vector1 = torch.mv(grad_output, vector2)
            if ctx.beta != 1:
                grad_vector1 *= ctx.beta

        if ctx.needs_input_grad[2]:
            # TODO: maybe it's better to do transpose + mv + transpose
            grad_vector2 = torch.mm(vector1.unsqueeze(0), grad_output).squeeze(0)
            if ctx.beta != 1:
                grad_vector2 *= ctx.beta

        return grad_add_matrix, grad_vector1, grad_vector2, None, None, None


class Dot(Function):

    @staticmethod
    def forward(ctx, vector1, vector2):
        ctx.save_for_backward(vector1, vector2)
        ctx.sizes = (vector1.size(), vector2.size())
        return vector1.new((vector1.dot(vector2),))

    @staticmethod
    def backward(ctx, grad_output):
        vector1, vector2 = ctx.saved_variables
        grad_vector1 = grad_vector2 = None

        if ctx.needs_input_grad[0]:
            grad_vector1 = vector2.mul(grad_output.expand(ctx.sizes[1])).view(ctx.sizes[0])

        if ctx.needs_input_grad[1]:
            grad_vector2 = vector1.mul(grad_output.expand(ctx.sizes[0])).view(ctx.sizes[1])

        return grad_vector1, grad_vector2
