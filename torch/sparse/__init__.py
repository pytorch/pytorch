import torch
from torch import _C
from ..tensor import _TensorBase

_sparse_tensor_classes = set()


class _SparseBase(object):
    is_cuda = False
    is_sparse = True

    def cpu(self):
        return self.type(getattr(torch.sparse, self.__class__.__name__))

    def is_pinned(self):
        raise NotImplementedError

    def pin_memory(self):
        raise NotImplementedError

    def share_memory_(self):
        raise NotImplementedError

    def is_shared(self):
        raise NotImplementedError

    def __deepcopy__(self, _memo):
        memo = _memo.setdefault('torch', {})
        if self._cdata in memo:
            return memo[self._cdata]
        new_tensor = self.clone()
        memo[self._cdata] = new_tensor
        return new_tensor

    def __reduce__(self):
        raise NotImplementedError

    def __getstate__(self):
        raise NotImplementedError

    def __setstate__(self, state):
        raise NotImplementedError

    def __bool__(self):
        # TODO (easy) implement numel and remove this override
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError

    def split(self, split_size, dim=0):
        raise NotImplementedError

    def chunk(self, n_chunks, dim=0):
        raise NotImplementedError

    def tolist(self):
        raise NotImplementedError

    def view_as(self, tensor):
        raise NotImplementedError

    def permute(self, *dims):
        raise NotImplementedError

    def expand(self, *sizes):
        raise NotImplementedError

    def expand_as(self, tensor):
        raise NotImplementedError

    def repeat(self, *sizes):
        raise NotImplementedError

    def __rsub__(self, other):
        raise NotImplementedError

    def __matmul__(self, other):
        raise NotImplementedError

    def __pow__(self, other):
        raise NotImplementedError

    def __ipow__(self, other):
        raise NotImplementedError

    def __rdiv__(self, other):
        raise NotImplementedError

    def __idiv__(self, other):
        raise NotImplementedError

    def __mod__(self, other):
        raise NotImplementedError

    def __neg__(self):
        raise NotImplementedError

    def __eq__(self, other):
        raise NotImplementedError

    def __ne__(self, other):
        raise NotImplementedError

    def __lt__(self, other):
        raise NotImplementedError

    def __le__(self, other):
        raise NotImplementedError

    def __gt__(self, other):
        raise NotImplementedError

    def __ge__(self, other):
        raise NotImplementedError

    def __and__(self, other):
        raise NotImplementedError

    def __or__(self, other):
        raise NotImplementedError

    def __xor__(self, other):
        raise NotImplementedError

    def __iand__(self, other):
        raise NotImplementedError

    def __ior__(self, other):
        raise NotImplementedError

    def __ixor__(self, other):
        raise NotImplementedError

    def __str__(self):
        self.contiguous()  # to make sure the output is consistent
        return '{} with indices:\n{}and values:\n{}'.format(
            self.__class__.__name__, self.indices(), self.values())


class SparseDoubleTensor(_SparseBase, _C.SparseDoubleTensorBase, _TensorBase):
    def is_signed(self):
        return True

# NB: Define the class with Sparse in the name, so we have better error
# messages.
DoubleTensor = SparseDoubleTensor

class SparseFloatTensor(_SparseBase, _C.SparseFloatTensorBase, _TensorBase):
    def is_signed(self):
        return True

FloatTensor = SparseFloatTensor

class SparseLongTensor(_SparseBase, _C.SparseLongTensorBase, _TensorBase):
    def is_signed(self):
        return True

LongTensor = SparseLongTensor

class SparseIntTensor(_SparseBase, _C.SparseIntTensorBase, _TensorBase):
    def is_signed(self):
        return True

IntTensor = SparseIntTensor

class SparseShortTensor(_SparseBase, _C.SparseShortTensorBase, _TensorBase):
    def is_signed(self):
        return True

ShortTensor = SparseShortTensor

class SparseCharTensor(_SparseBase, _C.SparseCharTensorBase, _TensorBase):
    def is_signed(self):
        # TODO
        return False

CharTensor = SparseCharTensor

class SparseByteTensor(_SparseBase, _C.SparseByteTensorBase, _TensorBase):
    def is_signed(self):
        return False

ByteTensor = SparseByteTensor

_sparse_tensor_classes.add(DoubleTensor)
_sparse_tensor_classes.add(FloatTensor)
_sparse_tensor_classes.add(LongTensor)
_sparse_tensor_classes.add(IntTensor)
_sparse_tensor_classes.add(ShortTensor)
_sparse_tensor_classes.add(CharTensor)
_sparse_tensor_classes.add(ByteTensor)
_sparse_tensor_classes.add(SparseDoubleTensor)
_sparse_tensor_classes.add(SparseFloatTensor)
_sparse_tensor_classes.add(SparseLongTensor)
_sparse_tensor_classes.add(SparseIntTensor)
_sparse_tensor_classes.add(SparseShortTensor)
_sparse_tensor_classes.add(SparseCharTensor)
_sparse_tensor_classes.add(SparseByteTensor)
torch._tensor_classes.update(_sparse_tensor_classes)

_C._sparse_init()
