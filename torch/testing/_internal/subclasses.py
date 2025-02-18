# mypy: ignore-errors
from typing import Any, Optional

import torch
import torch.utils._pytree as pytree
from torch._subclasses.base import BaseTensorSubclass, tensor_kwargs_from
from torch.testing._internal.two_tensor import TwoTensor


class WrapperSubclass(BaseTensorSubclass):
    INNER_TENSORS = ["a"]

    @staticmethod
    def __new__(cls, a, outer_size=None, outer_stride=None):
        if outer_size is None:
            outer_size = a.size()
        if outer_stride is None:
            outer_stride = a.stride()

        kwargs = {}
        kwargs["strides"] = a.stride()
        kwargs["storage_offset"] = a.storage_offset()
        kwargs["device"] = a.device
        kwargs["layout"] = a.layout
        kwargs["requires_grad"] = a.requires_grad
        kwargs["dtype"] = a.dtype
        out = torch.Tensor._make_wrapper_subclass(cls, a.size(), **kwargs)

        return out

    def __init__(self, a, outer_size=None, outer_stride=None):
        self.a = a

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        if kwargs is None:
            kwargs = {}
        out_a = cls.func_args_kwargs_attr(func, args, kwargs, "a")
        out_a_flat, spec = pytree.tree_flatten(out_a)
        out_flat = [
            cls(o_a) if isinstance(o_a, torch.Tensor) else o_a for o_a in out_a_flat
        ]
        return cls._return(func, args, kwargs, out)

    def __coerce_same_metadata_as_tangent__(
        self, expected_metadata: Any, expected_type: Optional[type] = None
    ):
        if expected_type == type(self.a):
            return self.a
        elif expected_type is TwoTensor:
            return TwoTensor(self.a, self.a.clone())

        return None


class LogTensor(BaseTensorSubclass):
    INNER_TENSORS = ["a"]

    @staticmethod
    def __new__(cls, a, outer_size=None, outer_stride=None):
        return torch.Tensor._make_wrapper_subclass(
            cls, outer_size or a.size(), **tensor_kwargs_from(a, outer_stride)
        )

    def __init__(self, a, outer_size=None, outer_stride=None):
        self.a = a

    @classmethod
    def torch_function_prologue(cls, func, types, args, kwargs) -> Optional[Any]:
        print(f"{cls} torch_function_prologue {func}")
        return None

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        print(f"{cls} torch_dispatch {func}")
        out = pytree.tree_map_only(
            torch.Tensor,
            lambda x: cls(x),
            cls.func_args_kwargs_attr(func, args, kwargs or {}, "a"),
        )
        return cls._return(func, args, kwargs, out)
