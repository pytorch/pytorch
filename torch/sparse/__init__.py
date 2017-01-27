import torch
from torch import _C

import sys

_sparse_tensor_classes = set()


class DoubleTensor(_C.SparseDoubleTensorBase):

    def is_signed(self):
        return True


class FloatTensor(_C.SparseFloatTensorBase):

    def is_signed(self):
        return True


class LongTensor(_C.SparseLongTensorBase):

    def is_signed(self):
        return True


class IntTensor(_C.SparseIntTensorBase):

    def is_signed(self):
        return True


class ShortTensor(_C.SparseShortTensorBase):

    def is_signed(self):
        return True


class CharTensor(_C.SparseCharTensorBase):

    def is_signed(self):
        # TODO
        return False


class ByteTensor(_C.SparseByteTensorBase):

    def is_signed(self):
        return False

_sparse_tensor_classes.add(DoubleTensor)
_sparse_tensor_classes.add(FloatTensor)
_sparse_tensor_classes.add(LongTensor)
_sparse_tensor_classes.add(IntTensor)
_sparse_tensor_classes.add(ShortTensor)
_sparse_tensor_classes.add(CharTensor)
_sparse_tensor_classes.add(ByteTensor)
torch._tensor_classes.update(_sparse_tensor_classes)

_C._sparse_init()
