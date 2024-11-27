import torch


def _try_get_val(t, attr):
    from torch._subclasses.fake_tensor import FakeTensor
    from torch._subclasses.functional_tensor import mb_unwrap_functional_tensor
    from torch.nested._internal.cached_tensor import CachedTensor
    from torch.nested._internal.offload_tensor import OffloadTensor

    if t is None:
        return None
    if isinstance(t, CachedTensor):
        for v in t.metadata.values():
            if val := _try_get_val(v, attr):
                return val
    if isinstance(t, OffloadTensor):
        if val := _try_get_val(t.host_tensor, attr):
            return val
        if val := _try_get_val(t.device_tensor, attr):
            return val
    if isinstance(t, torch.Tensor):
        if isinstance((t := mb_unwrap_functional_tensor(t)), FakeTensor):
            return getattr(t, attr)
    else:
        return None


def _try_get_source(t):
    return _try_get_val(t, "source")


def _try_get_fake_mode(t):
    return _try_get_val(t, "fake_mode")
