import torch
from torch import _C
from torch.sparse import _SparseTensorBase, _sparse_tensor_classes
from . import _lazy_init, device


if not hasattr(torch._C, 'CudaSparseDoubleTensorBase'):
    # Define dummy base classes
    for t in ['Double', 'Float', 'Long', 'Int', 'Short', 'Char', 'Byte', 'Half']:
        tensor_name = 'CudaSparse{0}TensorBase'.format(t)
        torch._C.__dict__[tensor_name] = type(tensor_name, (object,), {})


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


class DoubleTensor(_CudaSparseBase, torch._C.CudaSparseDoubleTensorBase, _SparseTensorBase):

    def is_signed(self):
        return True


class FloatTensor(_CudaSparseBase, torch._C.CudaSparseFloatTensorBase, _SparseTensorBase):

    def is_signed(self):
        return True


class LongTensor(_CudaSparseBase, torch._C.CudaSparseLongTensorBase, _SparseTensorBase):

    def is_signed(self):
        return True


class IntTensor(_CudaSparseBase, torch._C.CudaSparseIntTensorBase, _SparseTensorBase):

    def is_signed(self):
        return True


class ShortTensor(_CudaSparseBase, torch._C.CudaSparseShortTensorBase, _SparseTensorBase):

    def is_signed(self):
        return True


class CharTensor(_CudaSparseBase, torch._C.CudaSparseCharTensorBase, _SparseTensorBase):

    def is_signed(self):
        # TODO
        return False


class ByteTensor(_CudaSparseBase, torch._C.CudaSparseByteTensorBase, _SparseTensorBase):

    def is_signed(self):
        return False


class HalfTensor(_CudaSparseBase, torch._C.CudaSparseHalfTensorBase, _SparseTensorBase):

    def is_signed(self):
        return True


_sparse_tensor_classes.add(DoubleTensor)
_sparse_tensor_classes.add(FloatTensor)
_sparse_tensor_classes.add(LongTensor)
_sparse_tensor_classes.add(IntTensor)
_sparse_tensor_classes.add(ShortTensor)
_sparse_tensor_classes.add(CharTensor)
_sparse_tensor_classes.add(ByteTensor)
_sparse_tensor_classes.add(HalfTensor)
torch._tensor_classes.update(_sparse_tensor_classes)
