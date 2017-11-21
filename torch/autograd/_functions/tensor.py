from functools import reduce
import torch

from ..function import Function, InplaceFunction
from ..variable import Variable


def _preprocess_adv_index_seq(index):
    result = []
    for indexer in index:
        if isinstance(indexer, Variable):
            assert not indexer.requires_grad
            result.append(indexer.data)
        else:
            result.append(indexer)
    return result


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


class IndexAdd(InplaceFunction):

    @staticmethod
    def forward(ctx, tensor1, dim, index, tensor2, inplace=False):
        assert not ctx.needs_input_grad[2]
        ctx.dim = dim
        if ctx.needs_input_grad[3]:
            ctx.save_for_backward(index)
        if not inplace:
            tensor1 = tensor1.clone()
        else:
            ctx.mark_dirty(tensor1)
        return tensor1.index_add_(ctx.dim, index, tensor2)

    @staticmethod
    def backward(ctx, grad_output):
        grad_tensor1 = grad_tensor2 = None

        if ctx.needs_input_grad[0]:
            grad_tensor1 = grad_output

        if ctx.needs_input_grad[3]:
            index, = ctx.saved_variables
            grad_tensor2 = grad_output.index_select(ctx.dim, index)

        return grad_tensor1, None, None, grad_tensor2, None


class IndexCopy(InplaceFunction):

    @staticmethod
    def forward(ctx, tensor1, dim, index, tensor2, inplace=False):
        assert not ctx.needs_input_grad[2]
        ctx.dim = dim
        if any(ctx.needs_input_grad):
            ctx.save_for_backward(index)
        if not inplace:
            tensor1 = tensor1.clone()
        else:
            ctx.mark_dirty(tensor1)
        return tensor1.index_copy_(ctx.dim, index, tensor2)

    @staticmethod
    def backward(ctx, grad_output):
        grad_tensor1 = grad_tensor2 = None

        if any(ctx.needs_input_grad):
            index, = ctx.saved_variables

        if ctx.needs_input_grad[0]:
            grad_tensor1 = grad_output.clone().index_fill_(ctx.dim, index, 0)

        if ctx.needs_input_grad[3]:
            grad_tensor2 = grad_output.index_select(ctx.dim, index)

        return grad_tensor1, None, None, grad_tensor2, None


class IndexFill(InplaceFunction):

    @staticmethod
    def forward(ctx, tensor, dim, index, value, inplace=False):
        ctx.dim = dim
        assert not ctx.needs_input_grad[2]
        if ctx.needs_input_grad[0]:
            ctx.save_for_backward(index)
        if not inplace:
            tensor = tensor.clone()
        else:
            ctx.mark_dirty(tensor)
        return tensor.index_fill_(dim, index, value)

    @staticmethod
    def backward(ctx, grad_output):
        grad_tensor = None

        if ctx.needs_input_grad[0]:
            index, = ctx.saved_variables
            grad_tensor = grad_output.clone().index_fill_(ctx.dim, index, 0)

        return grad_tensor, None, None, None, None


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


class Clone(Function):

    @staticmethod
    def forward(ctx, input):
        return input.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class Unsqueeze(Function):

    @staticmethod
    def forward(ctx, input, dim):
        ctx.dim = dim
        result = input.unsqueeze(dim)
        ctx.mark_shared_storage((input, result))
        return result

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.squeeze(ctx.dim), None


class _MultiSelectionFunction(Function):

    @staticmethod
    def forward(ctx, input, dim, return_indices, args):
        fn = getattr(input, ctx._forward_cls.__name__.lower())
        ctx.return_indices = return_indices
        ctx.input_size = input.size()
        ctx.dim = dim
        output, indices = fn(*args)
        if return_indices:
            ctx.save_for_backward(indices)
            ctx.mark_non_differentiable(indices)
            return output, indices
        else:
            ctx.indices = indices
            return output

    @staticmethod
    def backward(ctx, grad_output, grad_indices=None):
        grad_input = Variable(grad_output.data.new(ctx.input_size).zero_())
        if ctx.return_indices:
            indices, = ctx.saved_variables
        else:
            indices = ctx.indices
        dim = ctx.dim if ctx.dim is not None else grad_output.dim() - 1
        return (grad_input.scatter(dim, indices, grad_output),) + (None,) * ctx.num_flags


class Sort(_MultiSelectionFunction):

    @staticmethod
    def forward(ctx, input, dim=None, descending=False, return_indices=True):
        ctx.dim = dim if dim is not None else input.dim() - 1
        args = (ctx.dim, descending)
        ctx.num_flags = 3
        return _MultiSelectionFunction.forward(ctx, input, dim, return_indices, args)


class Topk(_MultiSelectionFunction):

    @staticmethod
    def forward(ctx, input, k, dim=None, largest=True, sort=True, return_indices=True):
        ctx.dim = dim if dim is not None else input.dim() - 1
        args = (k, ctx.dim, largest, sort)
        ctx.num_flags = 5
        return _MultiSelectionFunction.forward(ctx, input, dim, return_indices, args)


class Gather(Function):

    @staticmethod
    def forward(ctx, input, dim, index):
        assert not ctx.needs_input_grad[2], "Gather can't differentiate the index"
        ctx.input_size = input.size()
        ctx.save_for_backward(index)
        ctx.dim = dim
        return input.gather(dim, index)

    @staticmethod
    def backward(ctx, grad_output):
        index, = ctx.saved_variables
        grad_input = Variable(grad_output.data.new(ctx.input_size).zero_())
        return grad_input.scatter_add_(ctx.dim, index, grad_output), None, None


class Scatter(InplaceFunction):

    @staticmethod
    def forward(ctx, input, dim, index, source, inplace=False):
        assert not ctx.needs_input_grad[2], "Scatter can't differentiate the index"
        ctx.dim = dim
        if inplace:
            ctx.mark_dirty(input)
        else:
            input = input.clone()
        ctx.save_for_backward(index)
        return input.scatter_(ctx.dim, index, source)

    @staticmethod
    def backward(ctx, grad_output):
        index, = ctx.saved_variables
        grad_input = grad_source = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.clone()
            grad_input.scatter_(ctx.dim, index, 0)
        if ctx.needs_input_grad[3]:
            grad_source = grad_output.gather(ctx.dim, index)
        return grad_input, None, None, grad_source, None


class ScatterAdd(InplaceFunction):

    @staticmethod
    def forward(ctx, input, dim, index, source, inplace=False):
        assert not ctx.needs_input_grad[2], "ScatterAdd can't differentiate the index"
        ctx.dim = dim
        if inplace:
            ctx.mark_dirty(input)
        else:
            input = input.clone()
        ctx.save_for_backward(index)
        return input.scatter_add_(ctx.dim, index, source)

    @staticmethod
    def backward(ctx, grad_output):
        index, = ctx.saved_variables
        grad_input = grad_source = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output
        if ctx.needs_input_grad[3]:
            grad_source = grad_output.gather(ctx.dim, index)
        return grad_input, None, None, grad_source, None


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


class Unfold(Function):

    @staticmethod
    def forward(ctx, input, dim, size, step):
        ctx.input_size = input.size()
        ctx.input_numel = input.numel()
        ctx.dim = dim
        ctx.size = size
        ctx.step = step
        result = input.unfold(dim, size, step)
        ctx.mark_shared_storage((input, result))
        return result

    @staticmethod
    def backward(ctx, grad_output):
        idx = grad_output.data.new().long()
        torch.arange(0, ctx.input_numel, out=idx)
        idx = idx.view(ctx.input_size)
        idx_unfolded = idx.unfold(ctx.dim, ctx.size, ctx.step)
        idx_unfolded = idx_unfolded.contiguous().view(-1)
        grad_input = Variable(grad_output.data.new(ctx.input_numel).zero_())
        grad_output = grad_output.contiguous().view(-1)
        grad_input = grad_input.index_add(0, Variable(idx_unfolded), grad_output)
        return grad_input.view(ctx.input_size), None, None, None
