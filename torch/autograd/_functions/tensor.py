from functools import reduce
import torch
from torch._utils import _accumulate

from ..function import Function, InplaceFunction


class Index(Function):

    def __init__(self, index):
        super(Index, self).__init__()
        self.index = index

    def forward(self, i):
        self.input_size = i.size()
        result = i.index(self.index)
        self.mark_shared_storage((i, result))
        return result

    def backward(self, grad_output):
        grad_input = grad_output.new(self.input_size).zero_()
        grad_input._set_index(self.index, grad_output)
        return grad_input


class SetItem(InplaceFunction):

    def __init__(self, index, value=None):
        super(SetItem, self).__init__(True)
        self.index = index
        self.value = value

    def forward(self, i, value=None):
        self.mark_dirty(i)
        if value is None:
            value = self.value
        i._set_index(self.index, value)
        return i

    def backward(self, grad_output):
        if self.value is None:
            grad_input = grad_output.clone()
            grad_input._set_index(self.index, 0)
            grad_value = grad_output.index(self.index).clone()
            return grad_input, grad_value
        else:
            grad_input = grad_output.clone()
            grad_input._set_index(self.index, 0)
            return grad_input


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

    def __init__(self, *dims):
        super(Transpose, self).__init__()
        assert len(dims) == 2
        self.dims = dims

    def forward(self, i):
        result = i.transpose(*self.dims)
        self.mark_shared_storage((i, result))
        return result

    def backward(self, grad_output):
        return grad_output.transpose(*self.dims)


class View(Function):

    def __init__(self, *sizes):
        super(View, self).__init__()
        self.sizes = sizes

    def forward(self, i):
        self.input_size = i.size()
        result = i.view(*self.sizes)
        self.mark_shared_storage((i, result))
        return result

    def backward(self, grad_output):
        # TODO: not sure if this clone is necessary
        return grad_output.contiguous().view(self.input_size)


class Expand(Function):

    def __init__(self, sizes):
        super(Expand, self).__init__()
        self.sizes = sizes
        self.expanded_dims = []

    def forward(self, i):
        result = i.expand(*self.sizes)
        unsqueezed = (1,) * (len(self.sizes) - len(i.size()))
        self.expanded_dims = [dim for dim, (expanded, original)
                              in enumerate(zip(self.sizes, unsqueezed + i.size()))
                              if expanded != original]
        self.mark_shared_storage((i, result))
        return result

    def backward(self, grad_output):
        grad_input = grad_output
        for dim in self.expanded_dims:
            grad_input = grad_input.sum(dim)
        return grad_input


class Type(Function):

    def __init__(self, dest_type):
        super(Type, self).__init__()
        self.dest_type = dest_type

    def forward(self, i):
        assert self.dest_type != type(i)
        self.input_type = type(i)
        return i.type(self.dest_type)

    def backward(self, grad_output):
        return grad_output.type(self.input_type)


class CudaTransfer(Function):

    def __init__(self, device_id=None, async=False):
        super(CudaTransfer, self).__init__()
        self.device_id = device_id
        self.async = async

    def forward(self, i):
        self.source_device = -1 if not i.is_cuda else i.get_device()
        self.source_was_cuda = i.is_cuda
        if self.device_id:
            return i.cuda(self.device_id, async=self.async)
        else:
            return i.cuda(async=self.async)

    def backward(self, grad_output):
        if self.source_device != -1:
            return grad_output.cuda(self.source_device)
        elif self.source_was_cuda:
            return grad_output
        else:
            return grad_output.cpu()


class Permute(Function):

    def __init__(self, dim_indices):
        super(Permute, self).__init__()
        self.dim_indices = dim_indices
        self.rev_dim_indices = [None for _ in range(len(dim_indices))]
        for i, dim_idx in enumerate(self.dim_indices):
            self.rev_dim_indices[dim_idx] = i

    def forward(self, i):
        result = i.permute(*self.dim_indices)
        self.mark_shared_storage((i, result))
        return result

    def backward(self, grad_output):
        return grad_output.permute(*self.rev_dim_indices)


class IndexAdd(InplaceFunction):

    def __init__(self, dim, inplace=False):
        super(IndexAdd, self).__init__(inplace)
        self.dim = dim

    def forward(self, tensor1, index, tensor2):
        assert not self.needs_input_grad[1]
        if self.needs_input_grad[2]:
            self.save_for_backward(index)
        if not self.inplace:
            tensor1 = tensor1.clone()
        else:
            self.mark_dirty(tensor1)
        return tensor1.index_add_(self.dim, index, tensor2)

    def backward(self, grad_output):
        grad_tensor1 = grad_tensor2 = None

        if self.needs_input_grad[0]:
            grad_tensor1 = grad_output

        if self.needs_input_grad[2]:
            index, = self.saved_tensors
            grad_tensor2 = grad_output.index_select(self.dim, index)

        return grad_tensor1, None, grad_tensor2


class IndexCopy(InplaceFunction):

    def __init__(self, dim, inplace=False):
        super(IndexCopy, self).__init__(inplace)
        self.dim = dim

    def forward(self, tensor1, index, tensor2):
        assert not self.needs_input_grad[1]
        if any(self.needs_input_grad):
            self.save_for_backward(index)
        if not self.inplace:
            tensor1 = tensor1.clone()
        else:
            self.mark_dirty(tensor1)
        return tensor1.index_copy_(self.dim, index, tensor2)

    def backward(self, grad_output):
        grad_tensor1 = grad_tensor2 = None

        if any(self.needs_input_grad):
            index, = self.saved_tensors

        if self.needs_input_grad[0]:
            grad_tensor1 = grad_output.clone().index_fill_(self.dim, index, 0)

        if self.needs_input_grad[2]:
            grad_tensor2 = grad_output.index_select(self.dim, index)

        return grad_tensor1, None, grad_tensor2


class IndexFill(InplaceFunction):

    def __init__(self, dim, value, inplace=False):
        super(IndexFill, self).__init__(inplace)
        self.dim = dim
        self.value = value

    def forward(self, tensor, index):
        assert not self.needs_input_grad[1]
        if self.needs_input_grad[0]:
            self.save_for_backward(index)
        if not self.inplace:
            tensor = tensor.clone()
        else:
            self.mark_dirty(tensor)
        return tensor.index_fill_(self.dim, index, self.value)

    def backward(self, grad_output):
        grad_tensor = None

        if self.needs_input_grad[0]:
            index, = self.saved_tensors
            grad_tensor = grad_output.clone().index_fill_(self.dim, index, 0)

        return grad_tensor, None


class IndexSelect(Function):

    def __init__(self, dim):
        super(IndexSelect, self).__init__()
        self.dim = dim

    def forward(self, tensor, index):
        assert not self.needs_input_grad[1]

        if self.needs_input_grad[0]:
            self.save_for_backward(index)
            self.input_size = tensor.size()

        return tensor.index_select(self.dim, index)

    def backward(self, grad_output):
        grad_tensor = None

        if self.needs_input_grad[0]:
            index, = self.saved_tensors
            grad_tensor = grad_output.new(*self.input_size).zero_()
            grad_tensor.index_add_(self.dim, index, grad_output)

        return grad_tensor, None


class Concat(Function):

    def __init__(self, dim):
        super(Concat, self).__init__()
        self.dim = dim

    def forward(self, *inputs):
        self.input_sizes = [i.size(self.dim) for i in inputs]
        return torch.cat(inputs, self.dim)

    def backward(self, grad_output):
        return tuple(grad_output.narrow(self.dim, end - size, size) for size, end
                     in zip(self.input_sizes, _accumulate(self.input_sizes)))


class Resize(Function):

    def __init__(self, *sizes):
        super(Resize, self).__init__()
        self.sizes = sizes
        self.numel = reduce(lambda x, y: x * y, sizes, 1)

    def forward(self, tensor):
        if tensor.numel() != self.numel:
            raise RuntimeError(("requested resize to {} ({} elements in total), "
                                "but the given tensor has a size of {} ({} elements). "
                                "autograd's resize can only change the shape of a given "
                                "tensor, while preserving the number of elements. ").format(
                'x'.join(map(str, self.sizes)), self.numel,
                'x'.join(map(str, tensor.size())), tensor.numel()))
        self.input_sizes = tensor.size()
        result = tensor.new(tensor).resize_(*self.sizes)
        self.mark_shared_storage((tensor, result))
        return result

    def backward(self, grad_output):
        assert grad_output.numel() == self.numel
        return grad_output.new(grad_output).resize_(self.input_sizes)


class Clone(Function):

    def forward(self, input):
        return input.clone()

    def backward(self, grad_output):
        return grad_output


class Squeeze(Function):

    def __init__(self, dim=None):
        super(Squeeze, self).__init__()
        self.dim = dim

    def forward(self, input):
        self.input_size = input.size()
        self.numel = input.numel()
        if self.dim is not None:
            result = input.squeeze(self.dim)
        else:
            result = input.squeeze()
        self.mark_shared_storage((input, result))
        return result

    def backward(self, grad_output):
        assert grad_output.numel() == self.numel
        return grad_output.contiguous().view(self.input_size)


class Unsqueeze(Function):

    def __init__(self, dim):
        super(Unsqueeze, self).__init__()
        self.dim = dim

    def forward(self, input):
        result = input.unsqueeze(self.dim)
        self.mark_shared_storage((input, result))
        return result

    def backward(self, grad_output):
        return grad_output.squeeze(self.dim)


class MaskedCopy(InplaceFunction):

    def forward(self, tensor1, mask, tensor2):
        assert not self.needs_input_grad[1], "MaskedCopy can't differentiate " \
            "the mask"
        if not self.inplace:
            tensor1 = tensor1.clone()
        else:
            self.mark_dirty(tensor1)
        self.save_for_backward(mask)
        return tensor1.masked_copy_(mask, tensor2)

    def backward(self, grad_output):
        mask, = self.saved_tensors
        grad_tensor1 = grad_tensor2 = None
        if self.needs_input_grad[0]:
            grad_tensor1 = grad_output.clone().masked_fill_(mask, 0)
        if self.needs_input_grad[2]:
            grad_tensor2 = grad_output.masked_select(mask)
        return grad_tensor1, None, grad_tensor2


class MaskedFill(InplaceFunction):

    def __init__(self, value, inplace=False):
        super(MaskedFill, self).__init__(inplace)
        self.value = value

    def forward(self, tensor, mask):
        assert not self.needs_input_grad[1], "MaskedFill can't differentiate " \
            "the mask"
        if not self.inplace:
            tensor = tensor.clone()
        else:
            self.mark_dirty(tensor)
        self.save_for_backward(mask)
        return tensor.masked_fill_(mask, self.value)

    def backward(self, grad_output):
        mask, = self.saved_tensors
        grad_tensor = None
        if self.needs_input_grad[0]:
            grad_tensor = grad_output.clone().masked_fill_(mask, 0)
        return grad_tensor, None


class MaskedSelect(Function):

    def forward(self, tensor, mask):
        assert not self.needs_input_grad[1], "MaskedSelect can't differentiate " \
            "the mask"
        self.input_size = tensor.size()
        self.save_for_backward(mask)
        return tensor.masked_select(mask)

    def backward(self, grad_output):
        mask, = self.saved_tensors
        grad_tensor = None
        if self.needs_input_grad[0]:
            # TODO: remove zero
            grad_tensor = grad_output.new(self.input_size).zero_()
            grad_tensor.masked_copy_(mask, grad_output)
        return grad_tensor, None


class _MultiSelectionFunction(Function):

    def __init__(self, dim, return_indices):
        super(_MultiSelectionFunction, self).__init__()
        self.dim = dim
        self.return_indices = return_indices

    def forward(self, input):
        fn = getattr(input, self.__class__.__name__.lower())
        self.input_size = input.size()
        output, indices = fn(*self.args)
        if self.return_indices:
            self.save_for_backward(indices)
            self.mark_non_differentiable(indices)
            return output, indices
        else:
            self.indices = indices
            return output

    def backward(self, grad_output, grad_indices=None):
        grad_input = grad_output.new(self.input_size).zero_()
        if self.return_indices:
            indices, = self.saved_tensors
        else:
            indices = self.indices
        dim = self.dim if self.dim is not None else grad_output.dim() - 1
        return grad_input.scatter_(dim, indices, grad_output)


class Sort(_MultiSelectionFunction):

    def __init__(self, dim=None, descending=False, return_indices=True):
        super(Sort, self).__init__(dim, return_indices)
        self.descending = descending

    def forward(self, input):
        dim = self.dim if self.dim is not None else input.dim() - 1
        self.args = (dim, self.descending)
        return super(Sort, self).forward(input)


class Topk(_MultiSelectionFunction):

    def __init__(self, k, dim=None, largest=True, sort=True, return_indices=True):
        super(Topk, self).__init__(dim, return_indices)
        self.k = k
        self.largest = largest
        self.sort = sort

    def forward(self, input):
        dim = self.dim if self.dim is not None else input.dim() - 1
        self.args = (self.k, dim, self.largest, self.sort)
        return super(Topk, self).forward(input)


class Chunk(Function):

    def __init__(self, num_chunks, dim=0):
        super(Chunk, self).__init__()
        self.num_chunks = num_chunks
        self.dim = dim

    def forward(self, i):
        self.input_size = i.size()
        result = i.chunk(self.num_chunks, self.dim)
        self.mark_shared_storage(*((i, chunk) for chunk in result))
        return result

    def backward(self, *grad_output):
        grad_input = grad_output[0].new(self.input_size)
        offset = 0
        for grad in grad_output:
            grad_size = grad.size(self.dim)
            grad_input.narrow(self.dim, offset, grad_size).copy_(grad)
            offset += grad_size
        return grad_input


class Gather(Function):

    def __init__(self, dim):
        super(Gather, self).__init__()
        self.dim = dim

    def forward(self, input, index):
        assert not self.needs_input_grad[1], "Gather can't differentiate " \
            "the index"
        self.input_size = input.size()
        self.save_for_backward(index)
        return input.gather(self.dim, index)

    def backward(self, grad_output):
        index, = self.saved_tensors
        grad_input = grad_output.new(self.input_size).zero_()
        return grad_input.scatter_(self.dim, index, grad_output), None


class Scatter(InplaceFunction):

    def __init__(self, dim, inplace=False):
        super(Scatter, self).__init__(inplace)
        self.dim = dim

    def forward(self, input, index, source):
        assert not self.needs_input_grad[1], "Scatter can't differentiate " \
            "the index"
        if self.inplace:
            self.mark_dirty(input)
        else:
            input = input.clone()
        self.save_for_backward(index)
        return input.scatter_(self.dim, index, source)

    def backward(self, grad_output):
        index, = self.saved_tensors
        grad_input = grad_source = None
        if self.needs_input_grad[0]:
            grad_input = grad_output.clone()
            grad_input.scatter_(self.dim, index, 0)
        if self.needs_input_grad[2]:
            grad_source = grad_output.gather(self.dim, index)
        return grad_input, None, grad_source


class Repeat(Function):

    def __init__(self, repeats):
        super(Repeat, self).__init__()
        self.repeats = repeats

    def forward(self, input):
        return input.repeat(self.repeats)

    def backward(self, grad_output):
        grad_input = grad_output
        for dim, repeat in enumerate(self.repeats):
            if repeat == 1:
                continue
            grad_input = sum(grad_input.chunk(repeat, dim))
        return grad_input


# TODO: unfold
