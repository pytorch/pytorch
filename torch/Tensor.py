import torch
from . import TensorPrinting
from functools import reduce
from itertools import chain
import sys
import math


def _infer_sizes(sizes, total):
    to_infer = -1
    total_sizes = 1
    for i, size in enumerate(sizes):
        total_sizes *= size
        if size == -1:
            if to_infer >= 0:
                raise RuntimeError
            to_infer = i
    if to_infer >= 0:
        assert total % total_sizes == 0, "Can't make sizes have exactly %d elements" % total
        sizes[to_infer] = -total / total_sizes
    return sizes


class _TensorBase(object):
    def new(self, *args, **kwargs):
        return self.__class__(*args, **kwargs)

    def type(self, t=None):
        current = "torch." + self.__class__.__name__
        if t is None:
            return current
        if t == current:
            return self
        _, _, typename = t.partition('.')
        return torch._import_dotted_name(t)(self.size()).copy(self)

    def typeAs(self, t):
        return self.type(t.type())

    def double(self):
        return self.type('torch.DoubleTensor')

    def float(self):
        return self.type('torch.FloatTensor')

    def long(self):
        return self.type('torch.LongTensor')

    def int(self):
        return self.type('torch.IntTensor')

    def short(self):
        return self.type('torch.ShortTensor')

    def char(self):
        return self.type('torch.CharTensor')

    def byte(self):
        return self.type('torch.ByteTensor')

    def copy(self, other):
        torch._C._tensorCopy(self, other)
        return self

    def __repr__(self):
        return str(self)

    def __str__(self):
        return TensorPrinting.printTensor(self)

    def __iter__(self):
        return iter(map(lambda i: self.select(0, i), torch._pyrange(self.size(0))))

    def split(self, split_size, dim=0):
        result = []
        dim_size = self.size(dim)
        num_splits = int(math.ceil(float(dim_size) / split_size))
        last_split_size = split_size - (split_size * num_splits - dim_size)
        def get_split_size(i):
            return split_size if i < num_splits-1 else last_split_size
        return [self.narrow(int(dim), int(i*split_size), int(get_split_size(i))) for i
                in torch._pyrange(0, num_splits)]

    def chunk(self, n_chunks, dim=0):
        split_size = math.ceil(float(self.size(dim)) / n_chunks)
        return self.split(split_size, dim)

    def tolist(self):
        dim = self.dim()
        if dim == 1:
            return [v for v in self]
        elif dim > 0:
            return [subt.tolist() for subt in self]
        return []

    def view(self, *args):
        dst = self.new()
        if len(args) == 1 and torch.isStorage(args[0]):
            sizes = args[0]
        else:
            sizes = torch.LongStorage(args)
        sizes = _infer_sizes(sizes, self.nElement())

        if reduce(lambda a,b: a * b, sizes) != self.nElement():
            raise RuntimeError('Invalid size for view. Input size: ' +
                    'x'.join(map(lambda v: str(v), self.size())) +
                    ', output size: ' +
                    'x'.join(map(lambda v: str(v), sizes)) + '.')

        assert self.isContiguous(), "expecting a contiguous tensor"
        dst.set_(self.storage(), self.storageOffset(), sizes)
        return dst

    def viewAs(self, tensor):
        return self.view(tensor.size())

    def permute(self, *args):
        perm = list(args)
        tensor = self
        n_dims = tensor.dim()
        assert len(perm) == n_dims, 'Invalid permutation'
        for i, p in enumerate(perm):
            if p != i and p != -1:
                j = i
                while True:
                    assert 0 <= perm[j] and perm[j] < n_dims, 'Invalid permutation'
                    tensor = tensor.transpose(j, perm[j])
                    perm[j], j = -1, perm[j]
                    if perm[j] == i:
                        break
                perm[j] = -1
        return tensor

    def expandAs(self, tensor):
        return self.expand(tensor.size())

    def expand(self, *args):
        result = self.new()
        sizes = args[0] if len(args) == 1 and torch.isLongStorage(args[0]) else torch.LongStorage(args)
        src = self

        src_dim = src.dim()
        src_stride = src.stride()
        src_size = src.size()

        if sizes.size() != src_dim:
            raise ValueError('the number of dimensions provided must equal tensor.dim()')

        # create a new geometry for tensor:
        for i, size in enumerate(src_size):
            if size == 1:
                src_size[i] = sizes[i]
                src_stride[i] = 0
            elif size != sizes[i]:
                raise ValueError('incorrect size: only supporting singleton expansion (size=1)')

        result.set_(src.storage(), src.storageOffset(),
                                src_size, src_stride)
        return result

    def repeatTensor(self, *args):
        # If args == (torch.LongStorage,), then we need to unpack the tuple
        repeats = list(args[0] if len(args) == 1 else args)
        result = self.new()
        src = self.contiguous()

        if len(repeats) < src.dim():
            raise ValueError('Number of dimensions of repeat dims can not be smaller than number of dimensions of tensor')

        xtensor = src.new().set_(src)
        xsize = xtensor.size().tolist()
        for i in torch._pyrange(len(repeats)-src.dim()):
            xsize = [1] + xsize

        size = torch.LongStorage([a * b for a, b in zip(xsize, repeats)])
        xtensor.resize_(torch.LongStorage(xsize))
        result.resize_(size)
        urtensor = result.new(result)
        for i in torch._pyrange(xtensor.dim()):
            urtensor = urtensor.unfold(i,xtensor.size(i),xtensor.size(i))
        for i in torch._pyrange(urtensor.dim()-xtensor.dim()):
            xsize = [1] + xsize
        xtensor.resize_(torch.LongStorage(xsize))
        xxtensor = xtensor.expandAs(urtensor)
        urtensor.copy(xxtensor)
        return result

    def __add__(self, other):
        return self.clone().add(other)
    __radd__ = __add__

    def __sub__(self, other):
        return self.clone().sub(other)
    __rsub__ = __sub__

    def __mul__(self, other):
        if torch.isTensor(other):
            dim_self = self.dim()
            dim_other = other.dim()
            if dim_self == 1 and dim_other == 1:
                return self.dot(other)
            elif dim_self == 2 and dim_other == 1:
                return torch.mv(self, other)
            elif dim_self == 2 and dim_other == 2:
                return torch.mm(self, other)
        else:
            return self.clone().mul(other)

    def __rmul__(self, other):
        # No need to check for tensor on lhs - it would execute it's __mul__
        return self.clone().mul(other)

    def __div__(self, other):
        return self.clone().div(other)
    __rdiv__ = __div__

    def __mod__(self, other):
        return self.clone().remainder(other)

    def __neg__(self):
        return self.clone().mul(-1)
