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
        return type(self)(self.size()).copy_(self)

    def tolist(self):
        return [v for v in self]

    def cpu(self):
        return self.type(getattr(torch, self.__class__.__name__))

    def double(self, async=False):
        return self.type(type(self).__module__ + '.DoubleStorage', async)

    def float(self, async=False):
        return self.type(type(self).__module__ + '.FloatStorage', async)

    def half(self, async=False):
        return self.type(type(self).__module__ + '.HalfStorage', async)

    def long(self, async=False):
        return self.type(type(self).__module__ + '.LongStorage', async)

    def int(self, async=False):
        return self.type(type(self).__module__ + '.IntStorage', async)

    def short(self, async=False):
        return self.type(type(self).__module__ + '.ShortStorage', async)

    def char(self, async=False):
        return self.type(type(self).__module__ + '.CharStorage', async)

    def byte(self, async=False):
        return self.type(type(self).__module__ + '.ByteStorage', async)

    def pin_memory(self):
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
