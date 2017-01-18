import torch
from ._utils import _type, _cuda, _range


class _StorageBase(object):
    is_cuda = False

    def __str__(self):
        content = ' ' + '\n '.join(str(self[i]) for i in _range(len(self)))
        return content + '\n[{} of size {}]'.format(torch.typename(self), len(self))

    def __repr__(self):
        return str(self)

    def __iter__(self):
        return iter(map(lambda i: self[i], _range(self.size())))

    def __copy__(self):
        return self.clone()

    def __deepcopy__(self, memo):
        memo = memo.setdefault('torch', {})
        if self._cdata in memo:
            return memo[self._cdata]
        new_storage = self.clone()
        memo[self._cdata] = new_storage
        return new_storage

    def __reduce__(self):
        return type(self), (self.tolist(),)

    def clone(self):
        """Returns a copy of this storage"""
        return type(self)(self.size()).copy_(self)

    def tolist(self):
        """Returns a list containing the elements of this storage"""
        return [v for v in self]

    def cpu(self):
        """Returns a CPU copy of this storage if it's not already on the CPU"""
        return self.type(getattr(torch, self.__class__.__name__))

    def double(self):
        """Casts this storage to double type"""
        return self.type(type(self).__module__ + '.DoubleStorage')

    def float(self):
        """Casts this storage to float type"""
        return self.type(type(self).__module__ + '.FloatStorage')

    def half(self):
        """Casts this storage to half type"""
        return self.type(type(self).__module__ + '.HalfStorage')

    def long(self):
        """Casts this storage to long type"""
        return self.type(type(self).__module__ + '.LongStorage')

    def int(self):
        """Casts this storage to int type"""
        return self.type(type(self).__module__ + '.IntStorage')

    def short(self):
        """Casts this storage to short type"""
        return self.type(type(self).__module__ + '.ShortStorage')

    def char(self):
        """Casts this storage to char type"""
        return self.type(type(self).__module__ + '.CharStorage')

    def byte(self):
        """Casts this storage to byte type"""
        return self.type(type(self).__module__ + '.ByteStorage')

    def pin_memory(self):
        """Copies the storage to pinned memory, if it's not already pinned."""
        if self.is_cuda:
            raise TypeError("cannot pin '{0}' only CPU memory can be pinned"
                            .format(self.type()))
        import torch.cuda
        allocator = torch.cuda._host_allocator()
        return type(self)(self.size(), allocator=allocator).copy_(self)

    def share_memory_(self):
        """Moves the storage to shared memory.

        This is a no-op for storages already in shared memory and for CUDA
        storages, which do not need to be moved for sharing across processes.
        Storages in shared memory cannot be resized.

        Returns: self
        """
        from torch.multiprocessing import get_sharing_strategy
        if self.is_cuda:
            pass  # CUDA doesn't use POSIX shared memory
        elif get_sharing_strategy() == 'file_system':
            self._share_filename_()
        else:
            self._share_fd_()
        return self


_StorageBase.type = _type
_StorageBase.cuda = _cuda
