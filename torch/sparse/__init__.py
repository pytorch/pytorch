import torch
from torch import _C
from .._utils import _type, _cuda

_sparse_tensor_classes = set()


class _SparseTensorBase(object):
    is_cuda = False
    is_sparse = True

    def double(self):
        """Casts this tensor to double type"""
        return self.type(type(self).__module__ + '.DoubleTensor')

    def float(self):
        """Casts this tensor to float type"""
        return self.type(type(self).__module__ + '.FloatTensor')

    def half(self):
        """Casts this tensor to half-precision float type"""
        return self.type(type(self).__module__ + '.HalfTensor')

    def long(self):
        """Casts this tensor to long type"""
        return self.type(type(self).__module__ + '.LongTensor')

    def int(self):
        """Casts this tensor to int type"""
        return self.type(type(self).__module__ + '.IntTensor')

    def short(self):
        """Casts this tensor to short type"""
        return self.type(type(self).__module__ + '.ShortTensor')

    def char(self):
        """Casts this tensor to char type"""
        return self.type(type(self).__module__ + '.CharTensor')

    def byte(self):
        """Casts this tensor to byte type"""
        return self.type(type(self).__module__ + '.ByteTensor')

    def __deepcopy__(self, _memo):
        memo = _memo.setdefault('torch', {})
        if self._cdata in memo:
            return memo[self._cdata]
        new_tensor = self.clone()
        memo[self._cdata] = new_tensor
        return new_tensor

    def __add__(self, other):
        return self.add(other)
    __radd__ = __add__

    def __iadd__(self, other):
        return self.add_(other)

    def __sub__(self, other):
        return self.sub(other)

    def __isub__(self, other):
        return self.sub_(other)

    def __mul__(self, other):
        return self.mul(other)
    __rmul__ = __mul__

    def __imul__(self, other):
        return self.mul_(other)

    def __div__(self, other):
        return self.div(other)
    __truediv__ = __div__

    def __idiv__(self, other):
        return self.div_(other)


class DoubleTensor(_SparseTensorBase, _C.SparseDoubleTensorBase):
    def is_signed(self):
        return True


class FloatTensor(_SparseTensorBase, _C.SparseFloatTensorBase):
    def is_signed(self):
        return True


class LongTensor(_SparseTensorBase, _C.SparseLongTensorBase):
    def is_signed(self):
        return True


class IntTensor(_SparseTensorBase, _C.SparseIntTensorBase):
    def is_signed(self):
        return True


class ShortTensor(_SparseTensorBase, _C.SparseShortTensorBase):
    def is_signed(self):
        return True


class CharTensor(_SparseTensorBase, _C.SparseCharTensorBase):
    def is_signed(self):
        # TODO
        return False


class ByteTensor(_SparseTensorBase, _C.SparseByteTensorBase):
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

_SparseTensorBase.type = _type
_SparseTensorBase.cuda = _cuda

_C._sparse_init()
