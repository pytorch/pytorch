from typing import Callable

import torch


# Apply `func` to all tensors in NestedTensor's metadata.
def apply_to_nested_metadata(
    t: torch.Tensor,
    func: Callable[[torch.Tensor], None],
    *,
    only_source_fields: bool,
    unwrap_functional_tensor: bool,
) -> None:
    from torch.nested._internal.cached_tensor import CachedTensor
    from torch.nested._internal.nested_tensor import EXTRA_FIELDS, SOURCE_FIELDS
    from torch.utils._python_dispatch import _apply_to_subclass

    def filter_fn(t_name: str, subclass_cls: object) -> bool:
        if subclass_cls is CachedTensor:
            fields = (
                SOURCE_FIELDS if only_source_fields else SOURCE_FIELDS + EXTRA_FIELDS
            )
            return t_name in fields
        else:
            raise RuntimeError("Unsupported traceable wrapper subclass", subclass_cls)

    _apply_to_subclass(
        t, func, filter_fn=filter_fn, unwrap_functional_tensor=unwrap_functional_tensor
    )
