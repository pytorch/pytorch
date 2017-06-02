from functools import reduce

from ..function import Function
from ..variable import Variable


class Sum(Function):

    @staticmethod
    def forward(ctx, input, dim=None, keepdim=True):
        ctx.dim = dim
        ctx.keepdim = keepdim
        ctx.input_size = input.size()
        if dim is None:
            return input.new((input.sum(),))
        else:
            return input.sum(dim, keepdim)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.dim is None:
            return grad_output.expand(ctx.input_size), None, None
        else:
            if ctx.keepdim is False:
                grad_output = grad_output.unsqueeze(ctx.dim)

            repeats = [1 for _ in ctx.input_size]
            repeats[ctx.dim] = ctx.input_size[ctx.dim]
            return grad_output.repeat(*repeats), None, None


class Prod(Function):

    @staticmethod
    def forward(ctx, input, dim=None, keepdim=True):
        ctx.dim = dim
        ctx.keepdim = keepdim
        ctx.input_size = input.size()
        if dim is None:
            ctx.result = input.prod()
            ctx.save_for_backward(input)
            return input.new((ctx.result,))
        else:
            output = input.prod(dim, keepdim)
            ctx.save_for_backward(input, output)
            return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.dim is None:
            input, = ctx.saved_variables
            zero_idx = (input.data == 0).nonzero()
            if zero_idx.dim() == 0:
                return grad_output.mul(ctx.result).expand_as(input).div(input), None, None
            elif zero_idx.size(0) > 1:
                return (grad_output * 0).expand_as(input), None, None
            else:
                grad_input = Variable(grad_output.data.new(ctx.input_size).zero_())
                zero_idx = tuple(zero_idx[0].cpu())
                to_add = input.data.new(ctx.input_size).zero_()
                to_add[zero_idx] = 1.
                grad_input[zero_idx] = grad_output * (input + Variable(to_add)).prod()
                return grad_input, None, None
        else:
            input, output = ctx.saved_variables
            dim = ctx.dim if ctx.dim >= 0 else ctx.dim + input.dim()
            if ctx.keepdim is False:
                grad_output = grad_output.unsqueeze(dim)

            zero_mask = input == 0
            slice_zero_count = zero_mask.sum(dim, True)
            total_zeros = slice_zero_count.sum()
            grad_input = grad_output.mul(output).expand_as(input).div(input)
            if total_zeros == 0:
                return grad_input, None, None

            some_zeros = slice_zero_count.gt(0).expand_as(grad_input)
            grad_input[some_zeros] = 0

            single_zero_idx = slice_zero_count.eq(1).nonzero()

            if len(single_zero_idx) == 0:
                return grad_input, None, None

            for idx in single_zero_idx:
                idx_tuple = tuple(idx.cpu())
                input_idx_tuple = idx_tuple[:dim] + (slice(0, None),) + idx_tuple[dim + 1:]

                # slice_mask and input_copy are 1D
                slice_mask = zero_mask[input_idx_tuple]
                input_copy = input[input_idx_tuple].clone()
                zero_idx = slice_mask.nonzero()[0, 0]
                input_copy[zero_idx] = 1.

                grad_idx_tuple = idx_tuple[:dim] + (zero_idx,) + idx_tuple[dim + 1:]
                grad_input[grad_idx_tuple] = grad_output[idx_tuple] * input_copy.prod()

            return grad_input, None, None


class Mean(Function):

    @staticmethod
    def forward(ctx, input, dim=None, keepdim=True):
        ctx.dim = dim
        ctx.keepdim = keepdim
        ctx.input_size = input.size()
        if dim is None:
            return input.new((input.mean(),))
        else:
            return input.mean(dim, keepdim)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.dim is None:
            grad_input_val = grad_output / reduce(lambda x, y: x * y, ctx.input_size, 1)
            return grad_input_val.expand(ctx.input_size), None, None
        else:
            if ctx.keepdim is False:
                grad_output = grad_output.unsqueeze(ctx.dim)

            repeats = [1 for _ in ctx.input_size]
            dim_size = ctx.input_size[ctx.dim]
            repeats[ctx.dim] = dim_size
            return grad_output.repeat(*repeats).div_(dim_size), None, None


class _SelectionFunction(Function):
    has_all_reduce = True
    # additional_args is prepended before dim when calling the tensor
    # function. It's a no-op for subclasses other than kthvalue.
    # kthvalue not only requires us to pass a dim, but also preceed it with k.

    @classmethod
    def forward(cls, ctx, input, dim=None, keepdim=True, additional_args=tuple()):
        fn = getattr(input, cls.__name__.lower())
        ctx.dim = dim
        ctx.keepdim = keepdim
        ctx.additional_args = additional_args
        ctx.input_size = input.size()
        if ctx.dim is None and cls.has_all_reduce:
            value = fn(*additional_args)
            ctx.indices_tuple = tuple(input.eq(value).nonzero()[0])
            return input.new((value,))
        else:
            if ctx.dim is None:
                dim = input.dim() - 1
            else:
                dim = ctx.dim
            args = (dim, keepdim)
            if additional_args:
                args = additional_args + args
            output, indices = fn(*args)
            ctx.save_for_backward(indices)
            ctx.mark_non_differentiable(indices)
            return output, indices

    @classmethod
    def backward(cls, ctx, grad_output, grad_indices=None):
        grad_input = Variable(grad_output.data.new(*ctx.input_size).zero_())
        if ctx.dim is None and cls.has_all_reduce:
            grad_input[ctx.indices_tuple] = grad_output.data[0]
        else:
            if ctx.dim is None:
                dim = len(ctx.input_size) - 1
            else:
                dim = ctx.dim

            indices, = ctx.saved_variables
            if ctx.keepdim is False:
                grad_output = grad_output.unsqueeze(dim)
                grad_indices = grad_indices.unsqueeze(dim)
                indices = indices.unsqueeze(dim)

            grad_input.scatter_(dim, indices, grad_output)
        return grad_input, None, None, None


class Max(_SelectionFunction):
    pass


class Min(_SelectionFunction):
    pass


class Mode(_SelectionFunction):
    has_all_reduce = False


class Median(_SelectionFunction):
    has_all_reduce = False


class Kthvalue(_SelectionFunction):
    has_all_reduce = False

    @classmethod
    def forward(cls, ctx, input, k, dim=None, keepdim=True):
        return super(Kthvalue, cls).forward(ctx, input, dim, keepdim, (k,))


class Norm(Function):

    @staticmethod
    def forward(ctx, input, p=2, dim=None, keepdim=True):
        ctx.p = p
        ctx.dim = dim
        ctx.keepdim = keepdim

        if dim is None:
            ctx.norm = input.norm(p)
            ctx.save_for_backward(input)
            return input.new((ctx.norm,))
        else:
            output = input.norm(p, dim, keepdim)
            ctx.save_for_backward(input, output)
            return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.dim is None:
            input, = ctx.saved_variables
            if ctx.p == 2:
                scale_v = (grad_output / ctx.norm).expand_as(input)
                return input.mul(scale_v), None, None, None
            else:
                pow = input.abs().pow(ctx.p - 2)
                scale_v = (grad_output / ctx.norm ** (ctx.p - 1)).expand_as(input)
                return input.mul(pow).mul(scale_v), None, None, None
        else:
            input, output = ctx.saved_variables

            if ctx.keepdim is False:
                grad_output = grad_output.unsqueeze(ctx.dim)
                output = output.unsqueeze(ctx.dim)

            big_grad_output = grad_output.expand_as(input)
            if ctx.p == 2:
                big_output = output.expand_as(input)
                return input.mul(big_grad_output).div(big_output), None, None, None
            else:
                pow = input.abs().pow(ctx.p - 2)
                big_output = output.pow(ctx.p - 1).expand_as(input)
                return input.mul(pow).mul(big_grad_output).div(big_output), None, None, None


# TODO: renorm
# TODO: std
# TODO: var
