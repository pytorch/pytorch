# Utility for nested subclass recursion with special handling for CachedTensor
from typing import *  # noqa: F403
from typing import Callable, List, Optional

import torch


# Recurse across all tensors in a nested structure, applying `func` to each one
def _apply_func_to_tensor(
    t: Optional[torch.Tensor],
    func: Callable[[torch.Tensor], None],
    unpack_func: Callable[
        [Optional[torch.Tensor]], Optional[List[Optional[torch.Tensor]]]
    ],
) -> None:
    def recurse(x: Optional[torch.Tensor]) -> None:
        if (inner := unpack_func(x)) is not None:
            for i in inner:
                recurse(i)
        elif isinstance(x, torch.Tensor):
            func(x)
        else:
            return

    recurse(t)


# Apply `func` to all tensors in NestedTensor's metadata.
def nested_metadata_apply_func(
    t: Optional[torch.Tensor],
    func: Callable[[torch.Tensor], None],
    *,
    only_source_fields: bool,
    unpack_functional_tensor: bool,
) -> None:
    from torch._subclasses.functional_tensor import (
        FunctionalTensor,
        mb_unwrap_functional_tensor,
    )
    from torch.nested._internal.cached_tensor import CachedTensor
    from torch.nested._internal.nested_tensor import EXTRA_FIELDS, SOURCE_FIELDS
    from torch.utils._python_dispatch import is_traceable_wrapper_subclass

    def nested_unpack(
        x: Optional[torch.Tensor],
    ) -> Optional[List[Optional[torch.Tensor]]]:
        if isinstance(x, CachedTensor):
            fields = (
                SOURCE_FIELDS if only_source_fields else SOURCE_FIELDS + EXTRA_FIELDS
            )
            return [v for k, v in x.metadata.items() if k in fields]
        elif isinstance(x, FunctionalTensor):
            return (
                [mb_unwrap_functional_tensor(x)] if unpack_functional_tensor else None
            )
        elif is_traceable_wrapper_subclass(x):
            raise RuntimeError("Unsupported traceable wrapper subclass", type(x))
        else:
            return None

    _apply_func_to_tensor(t, func, nested_unpack)
