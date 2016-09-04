import weakref
import torch

_shared_cache = weakref.WeakValueDictionary()

class _StorageBase(object):
    def __str__(self):
        content = ' ' + '\n '.join(str(self[i]) for i in torch._pyrange(len(self)))
        return content + '\n[{} of size {}]'.format(torch.typename(self), len(self))

    def __repr__(self):
        return str(self)

    def __iter__(self):
        return iter(map(lambda i: self[i], torch._pyrange(self.size())))

    def copy_(self, other):
        torch._C._storageCopy(self, other)
        return self

    def share(self):
        handle, weak_storage = self._share()
        manager_handle, object_handle, size = handle
        _shared_cache[object_handle] = weak_storage
        return handle

    @classmethod
    def new_shared(cls, args):
        manager_handle, object_handle, size = args
        try:
            weak_storage = _shared_cache[object_handle]
            # Try to momentarily convert a weak reference into a strong one
            weak_storage.retain()
            if weak_storage._cdata != 0:
                # Success, we managed to retain the storage before it was freed
                new_storage = type(weak_storage)(cdata=weak_storage._cdata)
                return new_storage
        except KeyError:
            pass
        new_storage, weak_storage = cls._new_shared(manager_handle, object_handle, size)
        _shared_cache[object_handle] = weak_storage
        return new_storage

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
