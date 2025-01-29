import weakref
from typing import Any, Callable, Dict

import torch


# Makes it possible to take a weakref
class Dct(dict):
    pass


def _flatten_subclass_to_dict(
    t: torch.Tensor,
    *,
    unwrap_functional_tensor: bool,
    filter_fn: Callable = lambda a, b: True,
) -> Dict[tuple[str, ...], torch.Tensor]:
    from torch._subclasses.functional_tensor import FunctionalTensor
    from torch.utils._python_dispatch import is_traceable_wrapper_subclass

    res = Dct()

    def recurse(t: Any, path: tuple[str, ...] = ()) -> None:
        # Avoid a refcycle. Refcycle can be extra problematic when a dead cycle
        # holds the last reference to a resurrectable Tensor, in which case
        # resurrection won't actually prevent its weakrefs from being cleared!
        # See https://github.com/pytorch/pytorch/issues/145253
        res_p = weakref.proxy(res)
        if is_traceable_wrapper_subclass(t):
            inner_names, _ = t.__tensor_flatten__()
            for name in inner_names:
                if filter_fn(name, type(t)):
                    new_path = path + (name,)
                    recurse(getattr(t, name), new_path)
        elif isinstance(t, FunctionalTensor):
            if unwrap_functional_tensor:
                t = torch._from_functional_tensor(t.elem)
                recurse(t, path)
            else:
                res_p[path] = t
        else:
            if isinstance(t, torch.Tensor):
                res_p[path] = t

    recurse(t)
    return dict(res)


# Apply `func` to all tensors in NestedTensor's metadata.
def flatten_nested_metadata_to_dict(
    t: torch.Tensor,
    *,
    only_source_fields: bool,
    unwrap_functional_tensor: bool,
) -> Dict[tuple[str, ...], torch.Tensor]:
    from torch.nested._internal.dict_tensor import DictTensor
    from torch.nested._internal.nested_tensor import EXTRA_FIELDS, SOURCE_FIELDS

    def filter_fn(t_name: str, subclass_cls: object) -> bool:
        if subclass_cls is DictTensor:
            fields = (
                SOURCE_FIELDS if only_source_fields else SOURCE_FIELDS + EXTRA_FIELDS
            )
            return t_name in fields
        else:
            raise RuntimeError("Unsupported traceable wrapper subclass", subclass_cls)

    return _flatten_subclass_to_dict(
        t,
        unwrap_functional_tensor=unwrap_functional_tensor,
        filter_fn=filter_fn,
    )
