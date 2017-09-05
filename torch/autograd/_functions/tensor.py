from functools import reduce
import torch
from torch._utils import _accumulate

from ..function import Function, InplaceFunction, once_differentiable
from ..variable import Variable
from .utils import maybe_unexpand


class Index(Function):

    @staticmethod
    def forward(ctx, i, index):
        ctx.input_size = i.size()
        ctx.index = index
        result = i.index(ctx.index)
        ctx.advanced_indexing = i._check_advanced_indexing(index)
        if not ctx.advanced_indexing:
            ctx.mark_shared_storage((i, result))
        return result

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.data.new(ctx.input_size).zero_()
        grad_input = Variable(grad_input)
        if ctx.advanced_indexing:
            grad_input._advanced_index_add(ctx.index, grad_output)
        else:
            grad_input[ctx.index] = grad_output
        return grad_input, None


class SetItem(InplaceFunction):

    @staticmethod
    def forward(ctx, i, index, value):
        assert not isinstance(index, Variable)
        ctx.mark_dirty(i)
        ctx.index = index
        ctx.tensor_value = torch.is_tensor(value)
        if ctx.tensor_value:
            ctx.value_size = value.size()
        i._set_index(ctx.index, value)
        return i

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        grad_input[ctx.index] = 0
        grad_value = None
        if ctx.tensor_value:
            grad_value = grad_output[ctx.index].contiguous().view(ctx.value_size)
        return grad_input, None, grad_value


# TODO: how to do NoGrad in new style
class NoGrad(Function):

    def forward(self, i):
        result = i.new(i)
        self.mark_non_differentiable(result)
        self.mark_shared_storage((i, result))
        return result

    def backward(self, grad_output):
        assert False, "backward of NoGrad should never be called"

    def _do_forward(self, *args, **kwargs):
        result = super(NoGrad, self)._do_forward(*args, **kwargs)
        self.requires_grad = False
        return result

    __call__ = _do_forward


class Transpose(Function):

    @staticmethod
    def forward(ctx, i, dim1, dim2):
        result = i.transpose(dim1, dim2)
        ctx.dims = (dim1, dim2)
        ctx.mark_shared_storage((i, result))
        return result

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.transpose(*ctx.dims), None, None


class View(Function):

    @staticmethod
    def forward(ctx, i, sizes):
        ctx.new_sizes = sizes
        ctx.old_size = i.size()
        result = i.view(*sizes)
        ctx.mark_shared_storage((i, result))
        return result

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.contiguous().view(ctx.old_size), None


class Expand(Function):

    @staticmethod
    # NOTE: new_size can be a tuple of any arguments that expand accepts, including a single-element
    # tuple containing torch.Size or a list
    def forward(ctx, i, new_size):
        result = i.expand(*new_size)
        ctx.num_unsqueezed = result.dim() - i.dim()
        ctx.expanded_dims = [dim for dim, (expanded, original)
                             in enumerate(zip(result.size()[ctx.num_unsqueezed:], i.size()))
                             if expanded != original]

        ctx.mark_shared_storage((i, result))
        return result

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output
        for i in range(ctx.num_unsqueezed):
            grad_input = grad_input.sum(0)
        for dim in ctx.expanded_dims:
            grad_input = grad_input.sum(dim, True)
        return grad_input, None


class Type(Function):

    @staticmethod
    def forward(ctx, i, dest_type):
        ctx.input_type = type(i)
        return i.type(dest_type)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.type(ctx.input_type), None


class CudaTransfer(Function):

    @staticmethod
    def forward(ctx, i, device_id=None, async=False):
        ctx.source_device = -1 if not i.is_cuda else i.get_device()
        ctx.source_was_cuda = i.is_cuda
        if device_id is not None:
            return i.cuda(device_id, async=async)
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


class Permute(Function):

    @staticmethod
    def forward(ctx, input, dim_indices):
        ctx.rev_dim_indices = [None for _ in range(len(dim_indices))]
        for i, dim_idx in enumerate(dim_indices):
            ctx.rev_dim_indices[dim_idx] = i
        result = input.permute(*dim_indices)
        ctx.mark_shared_storage((input, result))
        return result

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.permute(*ctx.rev_dim_indices), None


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


class AdvancedIndexAdd(InplaceFunction):

    @staticmethod
    def forward(ctx, tensor1, adv_index, tensor2):
        assert not ctx.needs_input_grad[1]
        if ctx.needs_input_grad[2]:
            ctx.adv_index = adv_index
        ctx.mark_dirty(tensor1)
        ctx.tensor2_size = tensor2.size()
        return tensor1._advanced_index_add(adv_index, tensor2)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        grad_tensor1 = grad_tensor2 = None

        if ctx.needs_input_grad[0]:
            grad_tensor1 = grad_output

        if ctx.needs_input_grad[2]:
            grad_tensor2 = grad_output._advanced_index_select(ctx.adv_index).contiguous().view(ctx.tensor2_size)
        return grad_tensor1, None, grad_tensor2


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


class IndexSelect(Function):

    @staticmethod
    def forward(ctx, tensor, dim, index):
        ctx.dim = dim
        assert not ctx.needs_input_grad[2]

        if ctx.needs_input_grad[0]:
            ctx.save_for_backward(index)
            ctx.input_size = tensor.size()

        return tensor.index_select(dim, index)

    @staticmethod
    def backward(ctx, grad_output):
        grad_tensor = None

        if ctx.needs_input_grad[0]:
            index, = ctx.saved_variables
            grad_tensor = Variable(grad_output.data.new(*ctx.input_size).zero_())
            grad_tensor = grad_tensor.index_add(ctx.dim, index, grad_output)

        return grad_tensor, None, None


class Concat(Function):

    @staticmethod
    def forward(ctx, dim, *inputs):
        ctx.dim = dim
        ctx.input_sizes = [i.size(dim) for i in inputs]
        return torch.cat(inputs, dim)

    @staticmethod
    def backward(ctx, grad_output):
        return (None,) + tuple(grad_output.narrow(ctx.dim, end - size, size) for size, end
                               in zip(ctx.input_sizes, _accumulate(ctx.input_sizes)))


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


class Squeeze(InplaceFunction):

    @staticmethod
    def forward(ctx, input, dim=None, inplace=False):
        ctx.dim = dim
        ctx.input_size = input.size()
        if inplace:
            ctx.mark_dirty(input)
            if dim is not None:
                return input.squeeze_(dim)
            else:
                return input.squeeze_()
        else:
            if dim is not None:
                result = input.squeeze(dim)
            else:
                result = input.squeeze()

            ctx.mark_shared_storage((input, result))
            return result

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.contiguous().view(ctx.input_size), None, None


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


class MaskedScatter(InplaceFunction):

    @staticmethod
    def forward(ctx, tensor1, mask, tensor2, inplace=False):
        assert not ctx.needs_input_grad[1], "MaskedScatter can't differentiate the mask"
        ctx.tensor1_size = tensor1.size()
        ctx.tensor2_size = tensor2.size()
        if not inplace:
            tensor1 = tensor1.clone()
        else:
            ctx.mark_dirty(tensor1)
        ctx.save_for_backward(mask)
        return tensor1.masked_scatter_(mask, tensor2)

    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_variables
        grad_tensor1 = grad_tensor2 = None
        if ctx.needs_input_grad[0]:
            grad_tensor1 = maybe_unexpand(grad_output.clone().masked_fill_(mask, 0), ctx.tensor1_size)
        if ctx.needs_input_grad[2]:
            grad_tensor2 = Variable(grad_output.data.new(ctx.tensor2_size).zero_())
            mask_selected = grad_output.masked_select(mask)
            diff_nelem = grad_tensor2.nelement() - mask_selected.nelement()
            if diff_nelem > 0:
                # because mask_selected returns a 1-d tensor with size of masked elements that are 1,
                # we need to fill out the rest with zeros then reshape back to tensor2's size.
                zeros_fillin = Variable(grad_output.data.new(diff_nelem).zero_())
                mask_selected = torch.cat((mask_selected, zeros_fillin), 0)

            mask_selected = mask_selected.view(ctx.tensor2_size)
            grad_tensor2 = maybe_unexpand(mask_selected, ctx.tensor2_size)
        return grad_tensor1, None, grad_tensor2, None


class MaskedFill(InplaceFunction):

    @staticmethod
    def forward(ctx, tensor, mask, value, inplace=False):
        assert not ctx.needs_input_grad[1], "MaskedFill can't differentiate the mask"
        ctx.tensor_size = tensor.size()
        if not inplace:
            tensor = tensor.clone()
        else:
            ctx.mark_dirty(tensor)
        ctx.save_for_backward(mask)
        return tensor.masked_fill_(mask, value)

    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_variables
        grad_tensor = None
        if ctx.needs_input_grad[0]:
            grad_tensor = maybe_unexpand(grad_output.clone().masked_fill_(mask, 0), ctx.tensor_size)
        return grad_tensor, None, None, None


class MaskedSelect(Function):

    @staticmethod
    def forward(ctx, tensor, mask):
        assert not ctx.needs_input_grad[1], "MaskedSelect can't differentiate the mask"
        ctx.input_size = tensor.size()
        ctx.save_for_backward(mask)
        return tensor.masked_select(mask)

    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_variables
        grad_tensor = None
        if ctx.needs_input_grad[0]:
            # determine the actual broadcasted sizes used
            try:
                new_size = torch._C._infer_size(ctx.input_size, mask.size())
            except RuntimeError:
                new_size = None

            # we need to potentially expand grad_tensor, since it is passed to Variable.masked_scatter, which
            # eventually is in-place (so can't rely on automatically broadcasting)
            grad_tensor = Variable(grad_output.data.new(new_size if new_size is not None else ctx.input_size).zero_())
            grad_tensor = grad_tensor.masked_scatter(mask, grad_output)
            grad_tensor = maybe_unexpand(grad_tensor, ctx.input_size)
        return grad_tensor, None


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


class Chunk(Function):

    @staticmethod
    def forward(ctx, i, num_chunks, dim=0):
        ctx.dim = dim
        result = i.chunk(num_chunks, dim)
        ctx.mark_shared_storage(*((i, chunk) for chunk in result))
        return result

    @staticmethod
    def backward(ctx, *grad_output):
        grad_input = torch.cat(grad_output, ctx.dim)
        return grad_input, None, None


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
