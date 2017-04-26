import torch
from torch import _C
from ..tensor import _TensorBase
from torch.sparse import _SparseBase, _sparse_tensor_classes
from . import _lazy_init, device, _dummy_type


if not hasattr(torch._C, 'CudaSparseDoubleTensorBase'):
    # Define dummy base classes
    for t in ['Double', 'Float', 'Long', 'Int', 'Short', 'Char', 'Byte', 'Half']:
        tensor_name = 'CudaSparse{0}TensorBase'.format(t)

        torch._C.__dict__[tensor_name] = _dummy_type(tensor_name)


class _CudaSparseBase(object):
    is_cuda = True
    is_sparse = True

    def type(self, *args, **kwargs):
        with device(self.get_device()):
            return super(_CudaSparseBase, self).type(*args, **kwargs)

    def __new__(cls, *args, **kwargs):
        _lazy_init()
        # We need this method only for lazy init, so we can remove it
        del _CudaSparseBase.__new__
        return super(_CudaSparseBase, cls).__new__(cls, *args, **kwargs)


class SparseDoubleTensor(_CudaSparseBase, torch._C.CudaSparseDoubleTensorBase, _SparseBase, _TensorBase):

    def is_signed(self):
        return True

DoubleTensor = SparseDoubleTensor

class SparseFloatTensor(_CudaSparseBase, torch._C.CudaSparseFloatTensorBase, _SparseBase, _TensorBase):

    def is_signed(self):
        return True

FloatTensor = SparseFloatTensor

class SparseLongTensor(_CudaSparseBase, torch._C.CudaSparseLongTensorBase, _SparseBase, _TensorBase):

    def is_signed(self):
        return True

LongTensor = SparseLongTensor

class SparseIntTensor(_CudaSparseBase, torch._C.CudaSparseIntTensorBase, _SparseBase, _TensorBase):

    def is_signed(self):
        return True

IntTensor = SparseIntTensor

class SparseShortTensor(_CudaSparseBase, torch._C.CudaSparseShortTensorBase, _SparseBase, _TensorBase):

    def is_signed(self):
        return True

ShortTensor = SparseShortTensor

class SparseCharTensor(_CudaSparseBase, torch._C.CudaSparseCharTensorBase, _SparseBase, _TensorBase):

    def is_signed(self):
        # TODO
        return False

CharTensor = SparseCharTensor

class SparseByteTensor(_CudaSparseBase, torch._C.CudaSparseByteTensorBase, _SparseBase, _TensorBase):

    def is_signed(self):
        return False

ByteTensor = SparseByteTensor

class SparseHalfTensor(_CudaSparseBase, torch._C.CudaSparseHalfTensorBase, _SparseBase, _TensorBase):

    def is_signed(self):
        return True

HalfTensor = SparseHalfTensor

_sparse_tensor_classes.add(DoubleTensor)
_sparse_tensor_classes.add(FloatTensor)
_sparse_tensor_classes.add(LongTensor)
_sparse_tensor_classes.add(IntTensor)
_sparse_tensor_classes.add(ShortTensor)
_sparse_tensor_classes.add(CharTensor)
_sparse_tensor_classes.add(ByteTensor)
_sparse_tensor_classes.add(HalfTensor)
_sparse_tensor_classes.add(SparseDoubleTensor)
_sparse_tensor_classes.add(SparseFloatTensor)
_sparse_tensor_classes.add(SparseLongTensor)
_sparse_tensor_classes.add(SparseIntTensor)
_sparse_tensor_classes.add(SparseShortTensor)
_sparse_tensor_classes.add(SparseCharTensor)
_sparse_tensor_classes.add(SparseByteTensor)
_sparse_tensor_classes.add(SparseHalfTensor)
torch._tensor_classes.update(_sparse_tensor_classes)
