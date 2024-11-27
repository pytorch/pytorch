import torch


# Utility for nested subclass recursion with special handling for CachedTensor
def apply_func(t, func, *, only_source_fields):
    from torch._subclasses.fake_tensor import FakeTensor
    from torch._subclasses.functional_tensor import (
        FunctionalTensor,
        mb_unwrap_functional_tensor,
    )
    from torch.nested._internal.cached_tensor import CachedTensor
    from torch.nested._internal.offload_tensor import OffloadTensor
    from torch.utils._python_dispatch import is_traceable_wrapper_subclass

    def recurse(t):
        if isinstance(t, CachedTensor):
            for k, v in t.metadata.items():
                fields = (
                    t.source_fields
                    if only_source_fields
                    else t.source_fields + t.extra_fields
                )
                if k in fields:
                    recurse(v)
        elif isinstance(t, OffloadTensor):
            recurse(t.host_tensor)
            recurse(t.device_tensor)
        elif isinstance(t, FunctionalTensor):
            recurse(mb_unwrap_functional_tensor(t))
        # Treat everything subclass as leaf tensors?
        elif is_traceable_wrapper_subclass(t) and not isinstance(t, (FakeTensor)):
            raise RuntimeError("Unsupported traceable wrapper subclass", type(t))
        elif isinstance(t, torch.Tensor):
            func(t)
        else:
            return None

    recurse(t)


def _try_get_fake_mode(t):
    from torch._subclasses.fake_tensor import FakeTensor

    out = []

    def func(t):
        if isinstance(t, FakeTensor) and t.fake_mode is not None:
            out.append(t.fake_mode)

    apply_func(t, func, only_source_fields=False)
    if len(out) == 0:
        return None
    assert all(out[0] is o for o in out)
    return out[0]


def _try_get_source(t):
    from torch._subclasses.fake_tensor import FakeTensor

    out = []

    def func(t):
        if isinstance(t, FakeTensor) and t.source is not None:
            out.append(t.source)

    apply_func(t, func, only_source_fields=True)
    if len(out) == 0:
        return None
    return out[0]
