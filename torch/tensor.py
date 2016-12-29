import torch
from . import _tensor_str
from ._utils import _type, _cuda, _range
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
        sizes = list(sizes)
        sizes[to_infer] = -total // total_sizes
        return torch.Size(sizes)
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

    def half(self, async=False):
        return self.type(type(self).__module__ + '.HalfTensor')

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

    def is_pinned(self):
        storage = self.storage()
        return storage.is_pinned() if storage else False

    def pin_memory(self):
        if self.is_cuda:
            raise TypeError("cannot pin '{0}' only CPU memory can be pinned"
                            .format(self.type()))
        storage = self.storage()
        if storage is None:
            storage = (self.storage_type())()
        return type(self)().set_(storage.pin_memory()).view_as(self)

    def share_memory_(self):
        """Moves the underlying storage to shared memory.

        This is a no-op if the underlying storage is already in shared memory
        and for CUDA tensors. Tensors in shared memory cannot be resized.
        """
        self.storage().share_memory_()
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
        # All strings are unicode in Python 3, while we have to encode unicode
        # strings in Python2. If we can't, let python decide the best
        # characters to replace unicode characters with.
        if sys.version_info > (3,):
            return _tensor_str._str(self)
        else:
            if hasattr(sys.stdout, 'encoding'):
                return _tensor_str._str(self).encode(
                    sys.stdout.encoding or 'UTF-8', 'replace')
            else:
                return _tensor_str._str(self).encode('UTF-8', 'replace')

    def __bool__(self):
        if self.numel() == 0:
            return False
        raise RuntimeError("bool value of non-empty " + torch.typename(self) +
                " objects is ambiguous")

    __nonzero__ = __bool__

    def __iter__(self):
        return iter(map(lambda i: self.select(0, i), _range(self.size(0))))

    def split(self, split_size, dim=0):
        if dim < 0:
            dim += self.dim()
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
        if len(args) == 1 and isinstance(args[0], torch.Size):
            sizes = args[0]
        else:
            sizes = torch.Size(args)
        sizes = _infer_sizes(sizes, self.nelement())
        numel = reduce(lambda a, b: a * b, sizes) if len(sizes) > 0 else 0

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
        if len(args) == 1 and isinstance(args[0], torch.Size):
            sizes = args[0]
        else:
            sizes = torch.Size(args)
        src = self

        src_dim = src.dim()
        src_stride = list(src.stride())
        src_size = list(src.size())

        if len(sizes) != src_dim:
            raise ValueError('the number of dimensions provided must equal tensor.dim()')

        # create a new geometry for tensor:
        for i, size in enumerate(src_size):
            if size == 1:
                src_size[i] = sizes[i]
                src_stride[i] = 0
            elif size != sizes[i]:
                raise ValueError('incorrect size: only supporting singleton expansion (size=1)')

        result.set_(src.storage(), src.storage_offset(), torch.Size(src_size),
                    tuple(src_stride))
        return result

    def repeat(self, *args):
        # If args == (torch.Size,), then we need to unpack the tuple
        if len(args) == 1 and isinstance(args[0], torch.Size):
            args = args[0]
        repeats = list(args)
        result = self.new()
        src = self.contiguous()

        if len(repeats) < src.dim():
            raise ValueError('Number of dimensions of repeat dims can not be smaller than number of dimensions of tensor')

        xtensor = src.new().set_(src)
        xsize = list(xtensor.size())
        for i in _range(len(repeats)-src.dim()):
            xsize = [1] + xsize

        size = torch.Size([a * b for a, b in zip(xsize, repeats)])
        xtensor.resize_(torch.Size(xsize))
        result.resize_(size)
        urtensor = result.new(result)
        for i in _range(xtensor.dim()):
            urtensor = urtensor.unfold(i,xtensor.size(i),xtensor.size(i))
        for i in _range(urtensor.dim()-xtensor.dim()):
            xsize = [1] + xsize
        xtensor.resize_(torch.Size(xsize))
        xxtensor = xtensor.expand_as(urtensor)
        urtensor.copy_(xxtensor)
        return result

    def unsqueeze(self, dim):
        return self.new(self).unsqueeze_(dim)

    def unsqueeze_(self, dim):
        sizes = list(self.size())
        sizes.insert(dim, 1)
        strides = list(self.stride())
        strides.insert(dim, strides[dim] if len(strides) < dim else 1)
        return self.set_(self.storage(), self.storage_offset(),
                         torch.Size(sizes), tuple(strides))

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

    def __eq__(self, other):
        return self.eq(other)

    def __ne__(self, other):
        return self.ne(other)

    def __lt__(self, other):
        return self.lt(other)

    def __le__(self, other):
        return self.le(other)

    def __gt__(self, other):
        return self.gt(other)

    def __ge__(self, other):
        return self.ge(other)

    # TODO: add native add or and xor in the libs
    def __and__(self, other):
        if (type(self).__name__ != 'ByteTensor' or
                type(other).__name__ != 'ByteTensor'):
            raise RuntimeError('logical operations are supported on ByteTensors only')
        return (self + other).eq(2)

    def __or__(self, other):
        if (type(self).__name__ != 'ByteTensor' or
                type(other).__name__ != 'ByteTensor'):
            raise RuntimeError('logical operations are supported on ByteTensors only')
        return (self + other).gt(0)

    def __xor__(self, other):
        if (type(self).__name__ != 'ByteTensor' or
                type(other).__name__ != 'ByteTensor'):
            raise RuntimeError('logical operations are supported on ByteTensors only')
        return (self + other).eq(1)

    def __iand__(self, other):
        if (type(self).__name__ != 'ByteTensor' or
                type(other).__name__ != 'ByteTensor'):
            raise RuntimeError('logical operations are supported on ByteTensors only')
        return self.mul_(other)

    def __ior__(self, other):
        if (type(self).__name__ != 'ByteTensor' or
                type(other).__name__ != 'ByteTensor'):
            raise RuntimeError('logical operations are supported on ByteTensors only')
        return self.copy_((self + other).gt(0))

    def __ixor__(self, other):
        if (type(self).__name__ != 'ByteTensor' or
                type(other).__name__ != 'ByteTensor'):
            raise RuntimeError('logical operations are supported on ByteTensors only')
        return self.copy_((self + other).eq(1))

    def __hash__(self):
        return id(self)


_TensorBase.type = _type
_TensorBase.cuda = _cuda
