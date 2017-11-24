from functools import reduce
import torch

from ..function import Function
from ..variable import Variable


class Type(Function):

    @staticmethod
    def forward(ctx, i, dest_type):
        ctx.input_type = type(i)
        ctx.input_device = -1 if not i.is_cuda else i.get_device()
        return i.type(dest_type)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.input_device == -1:
            return grad_output.type(ctx.input_type), None
        else:
            with torch.cuda.device(ctx.input_device):
                return grad_output.type(ctx.input_type), None


class CudaTransfer(Function):

    @staticmethod
    def forward(ctx, i, device=None, async=False):
        ctx.source_device = -1 if not i.is_cuda else i.get_device()
        ctx.source_was_cuda = i.is_cuda
        if device is not None:
            return i.cuda(device, async=async)
        else:
            return i.cuda(async=async)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.source_device != -1:
            return grad_output.cuda(ctx.source_device), None, None
        elif ctx.source_was_cuda:
            return grad_output, None, None
        else:
            return grad_output.cpu(), None, None


# TODO: deprecate this
class Resize(Function):

    @staticmethod
    def forward(ctx, tensor, sizes):
        ctx.sizes = sizes
        ctx.numel = reduce(lambda x, y: x * y, sizes, 1)
        if tensor.numel() != ctx.numel:
            raise RuntimeError(("requested resize to {} ({} elements in total), "
                                "but the given tensor has a size of {} ({} elements). "
                                "autograd's resize can only change the shape of a given "
                                "tensor, while preserving the number of elements. ").format(
                'x'.join(map(str, sizes)), ctx.numel,
                'x'.join(map(str, tensor.size())), tensor.numel()))
        ctx.input_sizes = tensor.size()
        if tensor.is_contiguous():
            result = tensor.new(tensor).contiguous().view(*sizes)
            ctx.mark_shared_storage((tensor, result))
            return result
        else:
            return tensor.contiguous().view(*sizes)

    @staticmethod
    def backward(ctx, grad_output):
        assert grad_output.numel() == ctx.numel
        return grad_output.contiguous().view(ctx.input_sizes), None


class Repeat(Function):

    @staticmethod
    def forward(ctx, input, repeats):
        ctx.repeats = repeats
        ctx.input_dims = input.dim()
        return input.repeat(repeats)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output
        num_unsqueezed = grad_output.dim() - ctx.input_dims
        for _ in range(num_unsqueezed):
            grad_input = grad_input.sum(0, keepdim=False)
        for dim, repeat in enumerate(ctx.repeats[num_unsqueezed:]):
            if repeat == 1:
                continue
            grad_input = sum(grad_input.chunk(repeat, dim))
        return grad_input, None


def sum_scan_exclusive(x, dim):
    ret = torch.cumsum(-x, dim=dim)

    end_idx = ret.size(dim) - 1
    ret_sum = ret.narrow(dim, end_idx, 1).clone()
    ret -= ret_sum.expand_as(ret)
    ret += x
    return ret


class Prod(Function):

    @staticmethod
    def forward(ctx, input, dim=None, keepdim=None):
        ctx.dim = dim
        ctx.keepdim = False if keepdim is None else keepdim
        ctx.input_size = input.size()
        if dim is None:
            ctx.result = input.prod()
            ctx.save_for_backward(input)
            return input.new((ctx.result,))
        else:
            if keepdim is not None:
                output = input.prod(dim, keepdim=keepdim)
            else:
                output = input.prod(dim)
            ctx.save_for_backward(input, output)
            return output

    @staticmethod
    def backward(ctx, grad_output):
        def safe_zeros_backward(inp, dim):
            # note that the gradient is equivalent to:
            # cumprod(exclusive, normal) * cumprod(exclusive, reverse), e.g.:
            # input:                        [    a,     b,     c]
            # cumprod(exclusive, normal):   [1    ,     a, a * b]
            # cumprod(exclusive, reverse):  [b * c,     c,     1]
            # product:                      [b * c, a * c, a * b]
            # and this is safe under input with 0s.
            if inp.size(dim) == 1:
                return grad_output

            ones_size = torch.Size((inp.size()[:dim] + (1,) + inp.size()[dim + 1:]))
            ones = Variable(grad_output.data.new(ones_size).fill_(1))
            exclusive_normal_nocp = torch.cat((ones, inp.narrow(dim, 0, inp.size(dim) - 1)), dim)
            exclusive_normal = exclusive_normal_nocp.cumprod(dim)

            def reverse_dim(var, dim):
                index = Variable(torch.arange(var.size(dim) - 1, -1, -1, out=var.data.new().long()))
                return var.index_select(dim, index)

            narrow_reverse = reverse_dim(inp.narrow(dim, 1, inp.size(dim) - 1), dim)
            exclusive_reverse_nocp = torch.cat((ones, narrow_reverse), dim)
            exclusive_reverse = reverse_dim(exclusive_reverse_nocp.cumprod(dim), dim)

            grad_input = grad_output.expand_as(exclusive_normal).mul(exclusive_normal.mul(exclusive_reverse))
            return grad_input

        if ctx.dim is None:
            input, = ctx.saved_variables
            zero_idx = (input.data == 0).nonzero()
            if zero_idx.dim() == 0:
                return grad_output.mul(ctx.result).expand_as(input).div(input), None, None
            elif zero_idx.size(0) > 1:
                return (grad_output * 0).expand_as(input), None, None
            else:
                return safe_zeros_backward(input.contiguous().view(-1), 0).view_as(input), None, None

        else:
            input, output = ctx.saved_variables
            dim = ctx.dim if ctx.dim >= 0 else ctx.dim + input.dim()
            if ctx.keepdim is False and len(ctx.input_size) != 1:
                grad_output = grad_output.unsqueeze(dim)
                output = output.unsqueeze(dim)

            zero_mask = input == 0
            slice_zero_count = zero_mask.sum(dim, True)
            total_zeros = slice_zero_count.data.sum()
            if total_zeros == 0:
                grad_input = grad_output.mul(output).expand_as(input).div(input)
            else:
                grad_input = safe_zeros_backward(input, dim)

            return grad_input, None, None


class Cumsum(Function):

    @staticmethod
    def forward(ctx, input, dim):
        ctx.dim = dim
        return torch.cumsum(input, dim=ctx.dim)

    @staticmethod
    def backward(ctx, grad_output):
        return sum_scan_exclusive(grad_output, dim=ctx.dim), None


class Cumprod(Function):

    @staticmethod
    def forward(ctx, input, dim):
        ctx.dim = dim
        ctx.save_for_backward(input)
        return torch.cumprod(input, dim=ctx.dim)

    @staticmethod
    def backward(ctx, grad_output):
        '''
        There are two algorithms to do this. The first one
        is very efficient, but works only when there are no
        nonzero elements in the input.

        The second one is much more complex, but it doesn't
        assume anything on the input. The main downside is
        that it takes time O(n^2), where n = input.size(self.dim)
        (i.e. the length of the cumulative product). This is in
        contrast to the forward pass and the efficient algorithm,
        which are both O(n).

        The second algorithm is a simple application of the chain
        rule. If x is an n-dimensional vector, and y = cumprod(x),
        and F is the final cost, then

        dF / dx_k = sum_j (dF / dy_j) * (dy_j / dx_k)   (1)

        The term dF / dy_j is just grad_output[j] (assuming again
        everything is one-dimensional).

        The term (dy_j / dx_k) is easilly seen to be

        if j >= k
            dy_j / dx_k = prod_{1 <= i <= j, i != k} x_i
        else:
            dy_j / dx_k = 0

        Note that the indicator (j>=k) can be taken out
        by replacing the sum in (1) with a sum from
        j = k to n.

        Thus,
        df / dx_k = sum_{k <= j <= n} grad_output[j] * (dy_j / dx_k)

        with
        dy_j / dx_k = prod_{1 <= i <= j, i != k} x_i     (2)

        Note that this last term is just the cumulative product
        with k omitted. Thus, if x_k (the input) is nonzero, we can
        just express this as

        dy_j / dx_k = (prod_{1 <= i <= j} x_i) / x_k
                    = y_j / x_k

        So therefore,

        df / dx_k = sum_{k <= j <= n} grad_output[j] * y_j / x_k

        so

        grad_output = sum_scan_exclusiv(grad_output * output) / input

        If the input is nonzero, we need to calculate the dy_j / dx_k
        by using the formula (2), called in the code omitted_products.

        The way the code calculates it is simply by noting that

        prod_{1 <= i <= j, i != k} x_i
            = (prod_{1 <= i <= k} x_i) * (prod_{k + 1 <= i <= j} x_i)

        the first term is calculated as prods_until_k, which since
        doesn't depend in j is easy to vectorize.

        The second term (indexed by j) is the cumulative product of
        x_{k+1}, x_{k+2}, ..., x_n, and it's named in the code
        prods_from_k_pkus_1, and it's calculated as a cumprod.

        In order to vectorize this properly, we need to add to
        omitted_products the dimensions where k > j, and therefore
        dy_j / dx_k = 0, which is done right after the assert.
        '''

        input, = ctx.saved_variables
        dim_size = input.size(ctx.dim)
        if dim_size == 1:
            return grad_output, None

        #  Simple case with nonzero elements in the input
        if (input != 0).data.all():
            output = torch.cumprod(input, dim=ctx.dim)
            return sum_scan_exclusive(output * grad_output, dim=ctx.dim) / input, None

        positive_dim = ctx.dim if ctx.dim >= 0 else input.dim() + ctx.dim
        dim_padding = (slice(None, None),) * (positive_dim)

        ones_size = list(input.size())
        ones_size[ctx.dim] = 1
        ones = Variable(input.data.new([1]).expand(ones_size))
        grad_input = Variable(grad_output.data.new(input.size()).zero_())
        for k in range(dim_size):
            if k == 0:
                prods_from_k_plus_1 = torch.cumprod(
                    input[dim_padding + (slice(k + 1, None),)],
                    dim=ctx.dim
                )

                omitted_products = torch.cat(
                    (ones, prods_from_k_plus_1),
                    dim=ctx.dim
                )

            elif k == dim_size - 1:
                prods_until_k = torch.prod(
                    input[dim_padding + (slice(None, k),)],
                    dim=ctx.dim,
                    keepdim=True
                )

                omitted_products = prods_until_k

            else:
                prods_until_k = torch.prod(
                    input[dim_padding + (slice(None, k),)],
                    dim=ctx.dim,
                    keepdim=True
                )

                prods_from_k_plus_1 = torch.cumprod(
                    input[dim_padding + (slice(k + 1, None),)],
                    dim=ctx.dim
                )

                omitted_products = prods_until_k.expand_as(
                    prods_from_k_plus_1) * prods_from_k_plus_1

                omitted_products = torch.cat(
                    (prods_until_k, omitted_products), ctx.dim)

            # At this point omitted_products is the same size
            # as input, except on the dimension dim where it's
            # dim_size - k
            assert omitted_products.size(ctx.dim) == dim_size - k

            # should we implement copy_ or _set_item in variable?
            index = tuple(slice(None, None) for _ in range(positive_dim)) + (k,)
            grad_input[index] = torch.sum(
                grad_output[dim_padding + (slice(k, None),)] * omitted_products,
                dim=ctx.dim)

        return grad_input, None
