import torch
from torch.utils.weak import WeakTensorKeyDictionary


def try_get_fake_mode(obj):
    from torch._subclasses.fake_tensor import FakeTensor
    from torch._subclasses.functional_tensor import mb_unwrap_functional_tensor

    from torch.nested._internal.metadata_cache import MetadataCache

    if isinstance(obj, dict):
        for v in obj.values():
            if v is not None:
                fake_mode = try_get_fake_mode(v)
                if fake_mode is not None:
                    return fake_mode
    elif isinstance(obj, torch.Tensor):
        t = mb_unwrap_functional_tensor(obj)
        if isinstance(t, FakeTensor):
            return t.fake_mode
        else:
            return None
    elif isinstance(obj, MetadataCache):
        # TODO(soulitzer): revisit assumptions
        # Assume that I have a cache that is registered somewhere.
        # Assume that anything in the cache has been registered.
        # Every MetadataCache must contain at least one tensor
        return try_get_fake_mode(obj.data)
    else:
        assert False, f"get_fake_mode: got unexpected type {type(obj)}"


class MaybeUnwrapKeysWeakKeyDict:
    def __init__(self, data=None):
        self.data: WeakTensorKeyDictionary[torch.Tensor, int] = (
            WeakTensorKeyDictionary()
        )
        if data:
            self.update(data)

    def _unwrap(self, tensor):
        from torch._subclasses.functional_tensor import mb_unwrap_functional_tensor

        return mb_unwrap_functional_tensor(tensor)

    def get(self, tensor, default=None):
        tensor = self._unwrap(tensor)
        return self.data.get(tensor, default)

    def __getitem__(self, tensor):
        tensor = self._unwrap(tensor)
        if tensor not in self.data:
            raise KeyError(tensor)
        return self.data[tensor]

    def __setitem__(self, tensor, cache_id):
        tensor = self._unwrap(tensor)
        self.data[tensor] = cache_id

    def __delitem__(self, tensor):
        tensor = self._unwrap(tensor)
        del self.data[tensor]

    def copy(self):
        return MaybeUnwrapKeysWeakKeyDict(self.data.copy())

    def __contains__(self, tensor):
        tensor = self._unwrap(tensor)
        return tensor in self.data

    def update(self, other):
        if isinstance(other, MaybeUnwrapKeysWeakKeyDict):
            self.data.update(other.data)
        else:
            for k, v in other.items():
                self[k] = v


def assert_not_fake(t):
    from torch._subclasses.fake_tensor import FakeTensor
    from torch._subclasses.functional_tensor import mb_unwrap_functional_tensor

    t = mb_unwrap_functional_tensor(t)
    assert not isinstance(t, FakeTensor)
