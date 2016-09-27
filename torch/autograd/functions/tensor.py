import torch

from ..function import Function
from ..variable import Variable


class Index(Function):

    def __init__(self, *index):
        super(Index, self).__init__()
        self.index = index

    def forward(self, i):
        self.input_size = i.size()
        return i[self.index]

    def backward(self, grad_output):
        # TODO: this won't have to be zeroed
        grad_input = grad_output.new(self.input_size).zero_()
        grad_input[self.index].copy_(grad_output)
        return grad_input


class Transpose(Function):

    def __init__(self, *dims):
        super(Transpose, self).__init__()
        assert len(dims) == 2
        self.dims = dims

    def forward(self, i):
        return i.transpose(*self.dims)

    def backward(self, grad_output):
        return grad_output.transpose(*self.dims)


class View(Function):

    def __init__(self, *sizes):
        super(View, self).__init__()
        self.sizes = sizes

    def forward(self, i):
        self.input_size = i.size()
        return i.view(*self.sizes)

    def backward(self, grad_output):
        # TODO: not sure if this clone is necessary
        return grad_output.clone().view(self.input_size)


class Expand(Function):
    def __init__(self, *sizes):
        super(Expand, self).__init__()
        self.sizes = sizes
        self.expanded_dims = []

    def forward(self, i):
        self.expanded_dims = [dim for dim, (expanded, original)
                in enumerate(zip(self.sizes, i.size()))
                if expanded != original]
        return i.expand(*self.sizes)

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

    def __init__(self, device_id=None):
        super(CudaTransfer, self).__init__()
        self.device_id = device_id

    def forward(self, i):
        self.source_device = -1 if not i.is_cuda else i.get_device()
        if self.device_id:
            return i.cuda(self.device_id)
        else:
            return i.cuda()

    def backward(self, grad_output):
        if self.source_device != -1:
            return grad_output.cuda(self.source_device)
        else:
            return grad_output.cpu()


class Permute(Function):

    def __init__(self, *dim_indices):
        super(Permute, self).__init__()
        self.dim_indices = dim_indices
        self.rev_dim_indices = [None for _ in range(len(dim_indices))]
        for i, dim_idx in enumerate(self.dim_indices):
            self.rev_dim_indices[dim_idx] = i

    def forward(self, i):
        return i.permute(*self.dim_indices)

    def backward(self, grad_output):
        return grad_output.permute(*self.rev_dim_indices)


class IndexAdd(Function):

    def __init__(self, dim):
        super(IndexAdd, self).__init__()
        self.dim = dim

    def forward(self, tensor1, index, tensor2):
        assert not self.needs_input_grad[1]
        if self.needs_input_grad[2]:
            self.save_for_backward(index)
        return tensor1.clone().index_add_(self.dim, index, tensor2)

    def backward(self, grad_output):
        grad_tensor1 = grad_tensor2 = None

        if self.needs_input_grad[0]:
            grad_tensor1 = grad_output

        if self.needs_input_grad[2]:
            index, = self.saved_tensors
            grad_tensor2 = grad_output.index_select(self.dim, index)

        return grad_tensor1, None, grad_tensor2


class IndexCopy(Function):

    def __init__(self, dim):
        super(IndexCopy, self).__init__()
        self.dim = dim

    def forward(self, tensor1, index, tensor2):
        assert not self.needs_input_grad[1]
        if any(self.needs_input_grad):
            self.save_for_backward(index)
        return tensor1.clone().index_copy_(self.dim, index, tensor2)

    def backward(self, grad_output):
        grad_tensor1 = grad_tensor2 = None

        if any(self.needs_input_grad):
            index, = self.saved_tensors

        if self.needs_input_grad[0]:
            grad_tensor1 = grad_output.clone().index_fill_(self.dim, index, 0)

        if self.needs_input_grad[2]:
            grad_tensor2 = grad_output.index_select(self.dim, index)

        return grad_tensor1, None, grad_tensor2


class IndexFill(Function):

    def __init__(self, dim, value):
        super(IndexFill, self).__init__()
        self.dim = dim
        self.value = value

    def forward(self, tensor, index):
        assert not self.needs_input_grad[1]
        if self.needs_input_grad[0]:
            self.save_for_backward(index)
        return tensor.clone().index_fill_(self.dim, index, self.value)

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
            grad_tensor.index_copy_(self.dim, index, grad_output)

        return grad_tensor, None


# TODO: cat
# TODO: chunk
# TODO: copy
# TODO: gather
# TODO: kthvalue
# TODO: repeat
# TODO: sort
# TODO: split
# TODO: squeeze
# TODO: topk
# TODO: unfold
