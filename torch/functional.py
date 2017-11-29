import torch
from ._utils import _range
from operator import mul
from functools import reduce

__all__ = [
    'split', 'chunk', 'stack', 'unbind', 'btriunpack', 'matmul', 'det',
]


def split(tensor, split_size, dim=0):
    """Splits the tensor into equally sized chunks (if possible).

    Last chunk will be smaller if the tensor size along a given dimension
    is not divisible by ``split_size``.

    Arguments:
        tensor (Tensor): tensor to split.
        split_size (int): size of a single chunk.
        dim (int): dimension along which to split the tensor.
    """
    if dim < 0:
        dim += tensor.dim()
    dim_size = tensor.size(dim)
    num_splits = (dim_size + split_size - 1) // split_size
    last_split_size = split_size - (split_size * num_splits - dim_size)

    def get_split_size(i):
        return split_size if i < num_splits - 1 else last_split_size
    return tuple(tensor.narrow(int(dim), int(i * split_size), int(get_split_size(i))) for i
                 in _range(0, num_splits))


def chunk(tensor, chunks, dim=0):
    """Splits a tensor into a number of chunks along a given dimension.

    Arguments:
        tensor (Tensor): tensor to split.
        chunks (int): number of chunks to return.
        dim (int): dimension along which to split the tensor.
    """
    if dim < 0:
        dim += tensor.dim()
    split_size = (tensor.size(dim) + chunks - 1) // chunks
    return split(tensor, split_size, dim)


def stack(sequence, dim=0, out=None):
    """Concatenates sequence of tensors along a new dimension.

    All tensors need to be of the same size.

    Arguments:
        sequence (Sequence): sequence of tensors to concatenate.
        dim (int): dimension to insert. Has to be between 0 and the number
            of dimensions of concatenated tensors (inclusive).
    """
    if len(sequence) == 0:
        raise ValueError("stack expects a non-empty sequence of tensors")
    if dim < 0:
        dim += sequence[0].dim() + 1
    inputs = [t.unsqueeze(dim) for t in sequence]
    if out is None:
        return torch.cat(inputs, dim)
    else:
        return torch.cat(inputs, dim, out=out)


def unbind(tensor, dim=0):
    """Removes a tensor dimension.

    Returns a tuple of all slices along a given dimension, already without it.

    Arguments:
        tensor (Tensor): tensor to unbind.
        dim (int): dimension to remove.
    """
    return tuple(tensor.select(dim, i) for i in _range(tensor.size(dim)))


def btriunpack(LU_data, LU_pivots, unpack_data=True, unpack_pivots=True):
    """Unpacks the data and pivots from a batched LU factorization (btrifact) of a tensor.

    Returns a tuple indexed by:
      0: The pivots.
      1: The L tensor.
      2: The U tensor.

    Arguments:
        LU_data (Tensor): The packed LU factorization data.
        LU_pivots (Tensor): The packed LU factorization pivots.
        unpack_data (bool): Flag indicating if the data should be unpacked.
        unpack_pivots (bool): Flag indicating if the pivots should be unpacked.
    """

    nBatch, sz, _ = LU_data.size()

    if unpack_data:
        I_U = torch.triu(torch.ones(sz, sz)).type_as(LU_data).byte().unsqueeze(0).expand(nBatch, sz, sz)
        I_L = 1 - I_U
        L = LU_data.new(LU_data.size()).zero_()
        U = LU_data.new(LU_data.size()).zero_()
        I_diag = torch.eye(sz).type_as(LU_data).byte().unsqueeze(0).expand(nBatch, sz, sz)
        L[I_diag] = 1.0
        L[I_L] = LU_data[I_L]
        U[I_U] = LU_data[I_U]
    else:
        L = U = None

    if unpack_pivots:
        P = torch.eye(sz).type_as(LU_data).unsqueeze(0).repeat(nBatch, 1, 1)
        for i in range(nBatch):
            for j in range(sz):
                k = LU_pivots[i, j] - 1
                t = P[i, :, j].clone()
                P[i, :, j] = P[i, :, k]
                P[i, :, k] = t
    else:
        P = None

    return P, L, U


def matmul(tensor1, tensor2, out=None):
    """Matrix product of two tensors.

    The behavior depends on the dimensionality of the tensors as follows:

    - If both tensors are 1-dimensional, the dot product (scalar) is returned.
    - If both arguments are 2-dimensional, the matrix-matrix product is returned.
    - If the first argument is 1-dimensional and the second argument is 2-dimensional,
      a 1 is prepended to its dimension for the purpose of the matrix multiply.
      After the matrix multiply, the prepended dimension is removed.
    - If the first argument is 2-dimensional and the second argument is 1-dimensional,
      the matrix-vector product is returned.
    - If both arguments are at least 1-dimensional and at least one argument is
      N-dimensional (where N > 2), then a batched matrix multiply is returned.  If the first
      argument is 1-dimensional, a 1 is prepended to its dimension for the purpose of the
      batched matrix multiply and removed after.  If the second argument is 1-dimensional, a
      1 is appended to its dimension for the purpose of the batched matrix multiple and removed after.
      The non-matrix (i.e. batch) dimensions are :ref:`broadcasted <broadcasting-semantics>` (and thus
      must be broadcastable).  For example, if :attr:`tensor1` is a `j x 1 x n x m` Tensor
      and :attr:`tensor2` is a `k x m x p` Tensor, :attr:`out` will be an `j x k x n x p` Tensor.

    .. note::

        The 1-dimensional dot product version of this function does not support an :attr:`out` parameter.

    Arguments:
        tensor1 (Tensor): First tensor to be multiplied
        tensor2 (Tensor): Second tensor to be multiplied
        out (Tensor, optional): Output tensor
    """
    dim_tensor1 = tensor1.dim()
    dim_tensor2 = tensor2.dim()
    if dim_tensor1 == 1 and dim_tensor2 == 1:
        if out is None:
            return torch.dot(tensor1, tensor2)
        else:
            raise ValueError("out must be None for 1-d tensor matmul, returns a scalar")
    if dim_tensor1 == 2 and dim_tensor2 == 1:
        if out is None:
            return torch.mv(tensor1, tensor2)
        else:
            return torch.mv(tensor1, tensor2, out=out)
    elif dim_tensor1 == 1 and dim_tensor2 == 2:
        if out is None:
            return torch.mm(tensor1.unsqueeze(0), tensor2).squeeze_(0)
        else:
            return torch.mm(tensor1.unsqueeze(0), tensor2, out=out).squeeze_(0)
    elif dim_tensor1 == 2 and dim_tensor2 == 2:
        if out is None:
            return torch.mm(tensor1, tensor2)
        else:
            return torch.mm(tensor1, tensor2, out=out)
    elif dim_tensor1 >= 3 and (dim_tensor2 == 1 or dim_tensor2 == 2):
        # optimization: use mm instead of bmm by folding tensor1's batch into
        # its leading matrix dimension.

        if dim_tensor2 == 1:
            tensor2 = tensor2.unsqueeze(-1)

        size1 = tensor1.size()
        size2 = tensor2.size()
        output_size = size1[:-1] + size2[-1:]

        # fold the batch into the first dimension
        tensor1 = tensor1.contiguous().view(-1, size1[-1])

        if out is None or not out.is_contiguous():
            output = torch.mm(tensor1, tensor2)
        else:
            output = torch.mm(tensor1, tensor2, out=out)

        output = output.view(output_size)

        if dim_tensor2 == 1:
            output = output.squeeze(-1)

        if out is not None:
            out.set_(output)
            return out

        return output
    elif (dim_tensor1 >= 1 and dim_tensor2 >= 1) and (dim_tensor1 >= 3 or dim_tensor2 >= 3):
        # ensure each tensor size is at least 3-dimensional
        tensor1_exp_size = torch.Size((1,) * max(3 - tensor1.dim(), 0) + tensor1.size())
        # rhs needs to be a separate case since we can't freely expand 1s on the rhs, but can on lhs
        if dim_tensor2 == 1:
            tensor2 = tensor2.unsqueeze(1)
        tensor2_exp_size = torch.Size((1,) * max(3 - tensor2.dim(), 0) + tensor2.size())

        # expand the batch portion (i.e. cut off matrix dimensions and expand rest)
        expand_batch_portion = torch._C._infer_size(tensor1_exp_size[:-2], tensor2_exp_size[:-2])

        # flatten expanded batches
        tensor1_expanded = tensor1.expand(*(expand_batch_portion + tensor1_exp_size[-2:])) \
            .contiguous().view(reduce(mul, expand_batch_portion), *tensor1_exp_size[-2:])
        tensor2_expanded = tensor2.expand(*(expand_batch_portion + tensor2_exp_size[-2:])) \
            .contiguous().view(reduce(mul, expand_batch_portion), *tensor2_exp_size[-2:])

        # reshape batches back into result
        total_expansion = expand_batch_portion + (tensor1_exp_size[-2], tensor2_exp_size[-1])

        def maybeSqueeze(tensor):
            if dim_tensor1 == 1:
                return tensor.squeeze(-2)
            elif dim_tensor2 == 1:
                return tensor.squeeze(-1)
            else:
                return tensor

        if out is None or not out.is_contiguous():
            output = torch.bmm(tensor1_expanded, tensor2_expanded)
        else:
            output = torch.bmm(tensor1_expanded, tensor2_expanded, out=out)

        output = maybeSqueeze(output.view(total_expansion))

        if out is not None:
            out.set_(output)
            return out

        return output

    raise ValueError("both arguments to __matmul__ need to be at least 1D, "
                     "but they are {}D and {}D".format(dim_tensor1, dim_tensor2))


def det(var):
    """Calculates determinant of a 2D square Variable.

    .. note::
        Backward through `det` internally uses SVD results. So double backward
        through `det` will need to backward through :meth:`~Tensor.svd`. This
        can be unstable in certain cases. Please see :meth:`~torch.svd` for
        details.

    Arguments:
        var (Variable): The input 2D square Variable.
    """
    if torch.is_tensor(var):
        raise ValueError("det is currently only supported on Variable")
    return var.det()
