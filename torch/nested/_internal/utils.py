import torch


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


def assert_not_fake(t):
    from torch._subclasses.fake_tensor import FakeTensor
    from torch._subclasses.functional_tensor import mb_unwrap_functional_tensor

    t = mb_unwrap_functional_tensor(t)
    assert not isinstance(t, FakeTensor)
