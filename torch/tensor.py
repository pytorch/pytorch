import torch
from . import _tensor_str
from ._utils import _type, _cuda, _range
import sys


class _TensorBase(object):
    #: bool: True if this is a CUDA tensor
    is_cuda = False

    def new(self, *args, **kwargs):
        """Constructs a new tensor of the same data type."""
        return self.__class__(*args, **kwargs)

    def type_as(self, tensor):
        """Returns this tensor cast to the type of the given tensor.

        This is a no-op if the tensor is already of the correct type. This is
        equivalent to::

            self.type(tensor.type())

        Params:
            tensor (Tensor): the tensor which has the desired type
        """
        return self.type(tensor.type())

    def cpu(self):
        """Returns a CPU copy of this tensor if it's not already on the CPU"""
        return self.type(getattr(torch, self.__class__.__name__))

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

    def is_pinned(self):
        """Returns true if this tensor resides in pinned memory"""
        storage = self.storage()
        return storage.is_pinned() if storage else False

    def pin_memory(self):
        """Copies the tensor to pinned memory, if it's not already pinned."""
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

    def is_shared(self):
        """Checks if tensor is in shared memory.

        This is always ``True`` for CUDA tensors.
        """
        return self.storage().is_shared()

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
        return (type(self), (), self.__getstate__())

    def __getstate__(self):
        return (self.storage(),
                self.storage_offset(),
                tuple(self.size()),
                self.stride())

    def __setstate__(self, state):
        self.set_(*state)

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
        """Splits this tensor into a list of tensors.

        See :func:`torch.split`.
        """
        return torch.split(self, split_size, dim)

    def chunk(self, n_chunks, dim=0):
        """Splits this tensor into a list of tensors.

        See :func:`torch.chunk`.
        """
        return torch.chunk(self, n_chunks, dim)

    def tolist(self):
        """Returns a nested list represenation of this tensor."""
        dim = self.dim()
        if dim == 1:
            return [v for v in self]
        elif dim > 0:
            return [subt.tolist() for subt in self]
        return []

    def view_as(self, tensor):
        """Returns this tensor viewed as the size as the specified tensor.

        This is equivalent to::

                self.view(tensor.size())
        """
        return self.view(tensor.size())

    def permute(self, *dims):
        """Permute the dimensions of this tensor.

        Args:
            *dims (int...): The desired ordering of dimensions

        Example:
            >>> x = torch.randn(2, 3, 5)
            >>> x.size()
            torch.Size([2, 3, 5])
            >>> x.permute(2, 0, 1).size()
            torch.Size([5, 2, 3])
        """
        perm = list(dims)
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

    def expand(self, *sizes):
        """Returns a new view of the tensor with singleton dimensions expanded
        to a larger size.

        Tensor can be also expanded to a larger number of dimensions, and the
        new ones will be appended at the front.

        Expanding a tensor does not allocate new memory, but only creates a
        new view on the existing tensor where a dimension of size one is
        expanded to a larger size by setting the ``stride`` to 0. Any dimension
        of size 1 can be expanded to an arbitrary value without allocating new
        memory.

        Args:
            *sizes (torch.Size or int...): The desired expanded size

        Example:
            >>> x = torch.Tensor([[1], [2], [3]])
            >>> x.size()
            torch.Size([3, 1])
            >>> x.expand(3, 4)
             1  1  1  1
             2  2  2  2
             3  3  3  3
            [torch.FloatTensor of size 3x4]
        """
        result = self.new()
        if len(sizes) == 1 and isinstance(sizes[0], torch.Size):
            sizes = sizes[0]
        else:
            sizes = torch.Size(sizes)
        src = self

        num_unsqueezed = len(sizes) - src.dim()
        if src.dim() == 0:
            raise ValueError('can\'t expand an empty tensor')
        if num_unsqueezed < 0:
            raise ValueError('the number of dimensions provided must be greater or equal tensor.dim()')

        src_stride = [0] * num_unsqueezed + list(src.stride())
        src_size = [1] * num_unsqueezed + list(src.size())
        for i in range(num_unsqueezed - 1, -1, -1):
            # to be consistent with .unsqueeze()
            src_stride[i] = src_size[i + 1] * src_stride[i + 1]

        # create a new geometry for tensor:
        for i, (size, target_size) in enumerate(zip(src_size, sizes)):
            if size == 1:
                if target_size == 1:
                    continue
                src_size[i] = target_size
                src_stride[i] = 0
            elif size != target_size:
                raise ValueError('incorrect size: only supporting singleton expansion (size=1)')

        result.set_(src.storage(), src.storage_offset(), torch.Size(src_size),
                    tuple(src_stride))
        return result

    def expand_as(self, tensor):
        """Expands this tensor to the size of the specified tensor.

        This is equivalent to::

            self.expand(tensor.size())
        """
        return self.expand(tensor.size())

    def repeat(self, *sizes):
        """Repeats this tensor along the specified dimensions.

        Unlike :meth:`expand`, this function copies the tensor's data.

        Args:
            *sizes (torch.Size or int...): The number of times to repeat this tensor along each dimension

        Example:
            >>> x = torch.Tensor([1, 2, 3])
            >>> x.repeat(4, 2)
             1  2  3  1  2  3
             1  2  3  1  2  3
             1  2  3  1  2  3
             1  2  3  1  2  3
            [torch.FloatTensor of size 4x6]
            >>> x.repeat(4, 2, 1).size()
            torch.Size([4, 2, 3])
        """
        # If args == (torch.Size,), then we need to unpack the tuple
        if len(sizes) == 1 and isinstance(sizes[0], torch.Size):
            sizes = sizes[0]
        repeats = list(sizes)
        result = self.new()
        src = self.contiguous()

        if len(repeats) < src.dim():
            raise ValueError('Number of dimensions of repeat dims can not be '
                             'smaller than number of dimensions of tensor')

        xtensor = src.new().set_(src)
        xsize = list(xtensor.size())
        for i in _range(len(repeats) - src.dim()):
            xsize = [1] + xsize

        size = torch.Size([a * b for a, b in zip(xsize, repeats)])
        xtensor.resize_(torch.Size(xsize))
        result.resize_(size)
        urtensor = result.new(result)
        for i in _range(xtensor.dim()):
            urtensor = urtensor.unfold(i, xtensor.size(i), xtensor.size(i))
        for i in _range(urtensor.dim() - xtensor.dim()):
            xsize = [1] + xsize
        xtensor.resize_(torch.Size(xsize))
        xxtensor = xtensor.expand_as(urtensor)
        urtensor.copy_(xxtensor)
        return result

    # TODO: add tests for operators
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
        try:
            dim_other = other.dim()
        except AttributeError:  # not a tensor
            return NotImplemented
        if dim_self == 1 and dim_other == 1:
            return self.dot(other)
        if dim_self == 2 and dim_other == 1:
            return self.mv(other)
        if dim_self == 1 and dim_other == 2:
            return self.unsqueeze(0).mm(other).squeeze(0)
        elif dim_self == 2 and dim_other == 2:
            return self.mm(other)
        raise ValueError("both arguments to __matmul__ need to be 1D or 2D, "
                         "but they are {}D and {}D".format(dim_self, dim_other))

    def __pow__(self, other):
        return self.pow(other)

    def __ipow__(self, other):
        return self.pow_(other)

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
