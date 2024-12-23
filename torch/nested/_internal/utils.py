from typing import *  # noqa: F403

import torch


# Utility for nested subclass recursion with special handling for CachedTensor
def apply_func(
    t: Optional[torch.Tensor],
    func: Callable[[torch.Tensor], None],
    *,
    only_source_fields: bool,
) -> None:
    from torch._subclasses.fake_tensor import FakeTensor
    from torch._subclasses.functional_tensor import (
        FunctionalTensor,
        mb_unwrap_functional_tensor,
    )
    from torch.nested._internal.cached_tensor import CachedTensor
    from torch.nested._internal.nested_tensor import EXTRA_FIELDS, SOURCE_FIELDS
    from torch.utils._python_dispatch import is_traceable_wrapper_subclass

    def recurse(t: Optional[torch.Tensor]) -> None:
        if isinstance(t, CachedTensor):
            for k, v in t.metadata.items():
                fields = (
                    SOURCE_FIELDS
                    if only_source_fields
                    else SOURCE_FIELDS + EXTRA_FIELDS
                )
                if k in fields:
                    recurse(v)
        elif isinstance(t, FunctionalTensor):
            recurse(mb_unwrap_functional_tensor(t))
        # Treat everything subclass as leaf tensors?
        elif is_traceable_wrapper_subclass(t) and not isinstance(t, (FakeTensor)):
            raise RuntimeError("Unsupported traceable wrapper subclass", type(t))
        elif isinstance(t, torch.Tensor):
            func(t)
        else:
            return

    recurse(t)


def _try_get_fake_mode(t: torch.Tensor) -> Any:
    from torch._subclasses.fake_tensor import FakeTensor

    out = []

    def func(t: torch.Tensor) -> None:
        if isinstance(t, FakeTensor) and t.fake_mode is not None:
            out.append(t.fake_mode)

    apply_func(t, func, only_source_fields=False)
    if len(out) == 0:
        return None
    assert all(out[0] is o for o in out)
    return out[0]
