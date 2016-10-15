import torch
from . import _tensor_str
from ._utils import _type, _cuda, _range
from functools import reduce
from itertools import chain
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
        sizes[to_infer] = -total // total_sizes
    return sizes


class _TensorBase(object):
    is_cuda = False

    def new(self, *args, **kwargs):
        return self.__class__(*args, **kwargs)

    def type_as(self, t, async=False):
        return self.type(t.type())

    def cpu(self):
        return self.type(getattr(torch, self.__class__.__name__))

    def double(self, async=False):
        return self.type(type(self).__module__ + '.DoubleTensor')

    def float(self, async=False):
        return self.type(type(self).__module__ + '.FloatTensor')

    def long(self, async=False):
        return self.type(type(self).__module__ + '.LongTensor')

    def int(self, async=False):
        return self.type(type(self).__module__ + '.IntTensor')

    def short(self, async=False):
        return self.type(type(self).__module__ + '.ShortTensor')

    def char(self, async=False):
        return self.type(type(self).__module__ + '.CharTensor')

    def byte(self, async=False):
        return self.type(type(self).__module__ + '.ByteTensor')

    def copy_(self, source, async=False):
        if async:
            torch._C._tensorCopyAsync(self, source)
        else:
            torch._C._tensorCopy(self, source)
        return self

    def __deepcopy__(self, _memo):
        memo = _memo.setdefault('torch', {})
        if self._cdata in memo:
            return memo[self._cdata]
        new_storage = self.storage().__deepcopy__(_memo)
        new_tensor = self.new()
        new_tensor.set_(new_storage, self.storage_offset(), self.size(), self.stride())
        memo[self._cdata] = new_tensor
        return new_tensor

    def __reduce__(self):
        return type(self), (self.tolist(),)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return _tensor_str._str(self)

    def __iter__(self):
        return iter(map(lambda i: self.select(0, i), _range(self.size(0))))

    def split(self, split_size, dim=0):
        dim_size = self.size(dim)
        num_splits = int(math.ceil(float(dim_size) / split_size))
        last_split_size = split_size - (split_size * num_splits - dim_size)
        def get_split_size(i):
            return split_size if i < num_splits-1 else last_split_size
        return tuple(self.narrow(int(dim), int(i*split_size), int(get_split_size(i))) for i
                in _range(0, num_splits))

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
        if len(args) == 1 and torch.is_storage(args[0]):
            sizes = args[0]
        else:
            sizes = torch.LongStorage(args)
        sizes = _infer_sizes(sizes, self.nelement())
        numel = reduce(lambda a,b: a * b, sizes) if len(sizes) > 0 else 0

        if numel != self.nelement():
            def format_size(size):
                return 'x'.join(str(v) for v in size) if len(size) > 0 else '0'
            raise ValueError(
                "view of size '{0}' is invalid for input of size '{1}'"
                .format(format_size(sizes), format_size(self.size())))
        if not self.is_contiguous():
            raise ValueError("input should be contiguous")
        if self.storage() is not None:
            dst.set_(self.storage(), self.storage_offset(), sizes)
        return dst

    def view_as(self, tensor):
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

    def expand_as(self, tensor):
        return self.expand(tensor.size())

    def expand(self, *args):
        result = self.new()
        sizes = args[0] if len(args) == 1 and isinstance(args[0], torch.LongStorage) else torch.LongStorage(args)
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

        result.set_(src.storage(), src.storage_offset(),
                                src_size, src_stride)
        return result

    def repeat(self, *args):
        # If args == (torch.LongStorage,), then we need to unpack the tuple
        if len(args) == 1 and isinstance(args[0], torch.LongStorage):
            args = args[0]
        repeats = list(args)
        result = self.new()
        src = self.contiguous()

        if len(repeats) < src.dim():
            raise ValueError('Number of dimensions of repeat dims can not be smaller than number of dimensions of tensor')

        xtensor = src.new().set_(src)
        xsize = xtensor.size().tolist()
        for i in _range(len(repeats)-src.dim()):
            xsize = [1] + xsize

        size = torch.LongStorage([a * b for a, b in zip(xsize, repeats)])
        xtensor.resize_(torch.LongStorage(xsize))
        result.resize_(size)
        urtensor = result.new(result)
        for i in _range(xtensor.dim()):
            urtensor = urtensor.unfold(i,xtensor.size(i),xtensor.size(i))
        for i in _range(urtensor.dim()-xtensor.dim()):
            xsize = [1] + xsize
        xtensor.resize_(torch.LongStorage(xsize))
        xxtensor = xtensor.expand_as(urtensor)
        urtensor.copy_(xxtensor)
        return result

    def unsqueeze(self, dim):
        return self.new(self).unsqueeze_(dim)

    def unsqueeze_(self, dim):
        sizes = self.size().tolist()
        sizes.insert(dim, 1)
        strides = self.stride().tolist()
        strides.insert(dim, 0)
        return self.set_(self.storage(), self.storage_offset(),
                torch.LongStorage(sizes), torch.LongStorage(strides))

    #TODO: add tests for operators
    def __add__(self, other):
        return self.add(other)
    __radd__ = __add__

    def __iadd__(self, other):
        return self.add_(other)

    def __sub__(self, other):
        return self.sub(other)

    def __rsub__(self, other):
        return self.new().resize_as_(self).fill_(other).add_(-1, self)

    def __isub__(self, other):
        return self.sub_(other)

    def __mul__(self, other):
        return self.mul(other)
    __rmul__ = __mul__

    def __imul__(self, other):
        return self.mul_(other)

    def __matmul__(self, other):
        dim_self = self.dim()
        dim_other = other.dim()
        # TODO: should this really be dot product?
        # if dim_self == 1 and dim_other == 1:
            # return self.dot(other)
        if dim_self == 2 and dim_other == 1:
            return torch.mv(self, other)
        elif dim_self == 2 and dim_other == 2:
            return torch.mm(self, other)

    def __div__(self, other):
        return self.div(other)
    __truediv__ = __div__

    def __rdiv__(self, other):
        return self.new().resize_as_(self).fill_(other).div_(self)
    __rtruediv__ = __rdiv__

    def __idiv__(self, other):
        return self.div_(other)

    def __mod__(self, other):
        return self.remainder(other)

    def __neg__(self):
        return self.neg()


_TensorBase.type = _type
_TensorBase.cuda = _cuda
