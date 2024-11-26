import torch


def _try_get_fake_mode(t):
    from torch._subclasses.fake_tensor import FakeTensor
    from torch._subclasses.functional_tensor import mb_unwrap_functional_tensor
    from torch.nested._internal.cached_tensor import CachedTensor
    from torch.nested._internal.offload_tensor import OffloadTenosr

    if t is None:
        return None
    if isinstance(t, CachedTensor):
        for v in t.metadata.values():
            if fake_mode := _try_get_fake_mode(v):
                return fake_mode
    if isinstance(t, OffloadTenosr):
        if fake_mode := _try_get_fake_mode(t.host_tensor):
            return fake_mode
        if fake_mode := _try_get_fake_mode(t.device_tensor):
            return fake_mode
    if isinstance(t, torch.Tensor):
        if isinstance((t := mb_unwrap_functional_tensor(t)), FakeTensor):
            return t.fake_mode
    else:
        return None
